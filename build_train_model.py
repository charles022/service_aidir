import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math

# --- CONFIG ---
class DCLTConfig:
    def __init__(self, num_locs, num_events, num_lanes):
        self.num_locs = num_locs
        self.num_events = num_events
        self.num_lanes = num_lanes
        self.cont_dim = 32
        self.d_model = 128
        self.nhead = 4
        self.num_layers = 3
        self.dropout = 0.2

# --- COMPONENTS ---
class CyclicalTimeEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.w_hour = nn.Parameter(torch.randn(out_dim // 4)) 
        self.w_day = nn.Parameter(torch.randn(out_dim // 4)) 
    def forward(self, hours, days):
        h_emb = torch.cat([torch.sin(2 * math.pi * hours.unsqueeze(-1) * self.w_hour), torch.cos(2 * math.pi * hours.unsqueeze(-1) * self.w_hour)], dim=-1)
        d_emb = torch.cat([torch.sin(2 * math.pi * days.unsqueeze(-1) * self.w_day), torch.cos(2 * math.pi * days.unsqueeze(-1) * self.w_day)], dim=-1)
        return torch.cat([h_emb, d_emb], dim=-1)

class ContextProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
    def forward(self, x): return self.net(x)

# --- MODEL (SUPERVISED RISK PREDICTOR) ---
class DeepContextualLogisticsTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Embeddings
        self.emb_loc = nn.Embedding(config.num_locs, 64)
        self.emb_event = nn.Embedding(config.num_events, 32)
        self.emb_lane = nn.Embedding(config.num_lanes, 32)
        
        # Physics
        self.time_encoder = CyclicalTimeEncoder(out_dim=64)
        self.context_proj = ContextProjector(in_dim=4, out_dim=64)
        
        # Fusion & Transformer
        self.fusion = nn.Linear(256, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 128, config.d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.nhead, dim_feedforward=512, dropout=config.dropout, batch_first=True, activation='gelu', norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=config.num_layers)
        
        # HEADS
        # 1. Step-wise Risk (The "Root Cause" Detector)
        self.step_head = nn.Sequential(nn.Linear(config.d_model, 128), nn.GELU(), nn.Linear(128, 1), nn.Sigmoid())
        
        # 2. Global Outcome (Auxiliary)
        self.global_head = nn.Sequential(nn.Linear(config.d_model, 1), nn.Sigmoid())

    def forward(self, batch):
        x = torch.cat([
            self.emb_loc(batch['locs']), self.emb_event(batch['events']), self.emb_lane(batch['lanes']),
            self.time_encoder(batch['hours'], batch['days']), self.context_proj(batch['context'])
        ], dim=-1)
        x = self.fusion(x)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.dropout(x)
        
        out = self.transformer(x, src_key_padding_mask=batch['mask'])
        
        step_risk = self.step_head(out).squeeze(-1)
        
        # Global pooling (Max Risk strategy)
        masked_out = out * (~batch['mask'].unsqueeze(-1))
        global_pool, _ = torch.max(masked_out, dim=1)
        global_risk = self.global_head(global_pool).squeeze(-1)
        
        return step_risk, global_risk

# --- SUPERVISED LOSS (Risk Velocity) ---
def risk_velocity_loss(step_risk, global_risk, target, mask):
    # 1. Global Accuracy
    loss_global = F.binary_cross_entropy(global_risk, target)
    
    # 2. Weak Supervision (Max step risk should match target)
    valid_step = step_risk * (~mask)
    max_step, _ = torch.max(valid_step, dim=1)
    loss_step_max = F.binary_cross_entropy(max_step, target)
    
    # 3. Monotonicity (Risk shouldn't drop without reason)
    loss_mono = torch.mean(F.relu(-(valid_step[:, 1:] - valid_step[:, :-1])))
    
    return loss_global + loss_step_max + (0.5 * loss_mono)

# --- DATASET ---
class LogisticsDataset(Dataset):
    def __init__(self, packages_df, scans_df, calendar_df, maps, max_seq_len=64):
        self.max_seq_len = max_seq_len
        self.scans = scans_df.sort_values(['PackageID', 'ScanTime']).groupby('PackageID')
        self.package_ids = list(self.scans.groups.keys())
        self.maps = maps
        self.labels = packages_df.set_index('PackageID')['FailedService'].to_dict()

    def __len__(self): return len(self.package_ids)

    def __getitem__(self, idx):
        pkg_id = self.package_ids[idx]
        journey = self.scans.get_group(pkg_id)
        
        locs, events, lanes, hours, days, ctx = [], [], [], [], [], []
        prev_time = None
        
        for _, row in journey.iterrows():
            curr = pd.to_datetime(row['ScanTime'])
            locs.append(self.maps['loc'].get(row['LocationID'], 0))
            events.append(self.maps['event'].get(row['Event'], 0))
            lanes.append(self.maps['lane'].get(row.get('LaneID', 'NONE'), 0))
            hours.append(curr.hour + curr.minute/60.0); days.append(curr.weekday())
            
            delta = (curr - prev_time).total_seconds()/3600.0 if prev_time else 0.0
            prev_time = curr
            
            # Physics Features
            ttc = row.get('Time_To_Cut_Mins', 0.0) / 60.0
            ctx.append([delta, ttc, 1.0 if ttc < 0 else 0.0, row.get('Facility_Load', 0.8)])

        # Padding
        pad = self.max_seq_len - len(locs)
        if pad > 0:
            locs += [0]*pad; events += [0]*pad; lanes += [0]*pad; hours += [0.]*pad; days += [0.]*pad; ctx += [[0.]*4]*pad
            mask = [False]*(len(locs)-pad) + [True]*pad
        else:
            limit=self.max_seq_len
            locs=locs[:limit]; events=events[:limit]; lanes=lanes[:limit]; hours=hours[:limit]; days=days[:limit]; ctx=ctx[:limit]; mask=[False]*limit

        return {
            'locs': torch.tensor(locs), 'events': torch.tensor(events), 'lanes': torch.tensor(lanes),
            'hours': torch.tensor(hours), 'days': torch.tensor(days), 'context': torch.tensor(ctx),
            'mask': torch.tensor(mask, dtype=torch.bool), 'label': torch.tensor(self.labels[pkg_id], dtype=torch.float32)
        }

# --- SETUP SYSTEM (SUPERVISED) ---
def setup_system(packages_df, scans_df, calendar_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    maps = {
        'loc': {l:i for i,l in enumerate(scans_df['LocationID'].unique())},
        'event': {e:i for i,e in enumerate(scans_df['Event'].unique())},
        'lane': {l:i for i,l in enumerate(scans_df['LaneID'].fillna('NONE').unique())}
    }
    
    # Temporal Split: Train on first 11 days (0-10), Val on rest (11+)
    # This requires 'StartDay' in packages_df
    split_day = 11
    train_df = packages_df[packages_df['StartDay'] < split_day]
    val_df = packages_df[packages_df['StartDay'] >= split_day]
    
    print(f"Temporal Split Strategy: Train (Days 0-{split_day-1}) vs Val (Days {split_day}+)")
    print(f"Training Packages: {len(train_df)} | Validation Packages: {len(val_df)}")
    
    train_ids = train_df['PackageID'].unique()
    val_ids = val_df['PackageID'].unique()
    
    train_ds = LogisticsDataset(train_df, scans_df[scans_df['PackageID'].isin(train_ids)], None, maps)
    val_ds = LogisticsDataset(val_df, scans_df[scans_df['PackageID'].isin(val_ids)], None, maps)
    
    loaders = {'train': DataLoader(train_ds, batch_size=128, shuffle=True), 'val': DataLoader(val_ds, batch_size=128)}
    
    config = DCLTConfig(len(maps['loc']), len(maps['event']), len(maps['lane']))
    model = DeepContextualLogisticsTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    return model, optimizer, loaders, device, scheduler
