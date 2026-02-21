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
        self.context_in_dim = 2
        self.d_model = 256
        self.nhead = 8
        self.num_layers = 6
        self.dropout = 0.1

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
        self.emb_dest = nn.Embedding(config.num_locs, 64)

        # Time & Context
        self.time_encoder = CyclicalTimeEncoder(out_dim=64)
        self.context_proj = ContextProjector(in_dim=config.context_in_dim, out_dim=64)

        # Fusion & Transformer (64+32+32+64+64+64 = 320)
        self.fusion = nn.Linear(320, config.d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 128, config.d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.nhead, dim_feedforward=1024, batch_first=True, activation='gelu', norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=config.num_layers)

        # HEADS
        self.step_head = nn.Sequential(nn.Linear(config.d_model, 128), nn.GELU(), nn.Linear(128, 1), nn.Sigmoid())
        self.global_head = nn.Sequential(nn.Linear(config.d_model, 1), nn.Sigmoid())

    def forward(self, batch):
        x = torch.cat([
            self.emb_loc(batch['locs']),
            self.emb_event(batch['events']),
            self.emb_lane(batch['lanes']),
            self.emb_dest(batch['dests']),
            self.time_encoder(batch['hours'], batch['days']),
            self.context_proj(batch['context'])
        ], dim=-1)
        x = self.fusion(x)
        x = x + self.pos_encoder[:, :x.size(1), :]

        out = self.transformer(x, src_key_padding_mask=batch['mask'])

        step_risk = self.step_head(out).squeeze(-1)

        # Global pooling (Max Risk strategy)
        masked_out = out * (~batch['mask'].unsqueeze(-1))
        global_pool, _ = torch.max(masked_out, dim=1)
        global_risk = self.global_head(global_pool).squeeze(-1)

        return step_risk, global_risk

# --- SUPERVISED LOSS ---
def risk_velocity_loss(step_risk, global_risk, target, mask, step_targets):
    # 1. Global BCE
    loss_global = F.binary_cross_entropy(global_risk, target)

    # 2. Explicit step-level BCE (masked)
    valid_mask = (~mask).float()
    num_valid = valid_mask.sum().clamp(min=1.0)
    step_bce = F.binary_cross_entropy(step_risk, step_targets, reduction='none')
    loss_step = (step_bce * valid_mask).sum() / num_valid

    # 3. Monotonicity (risk shouldn't drop)
    valid_step = step_risk * valid_mask
    loss_mono = torch.mean(F.relu(-(valid_step[:, 1:] - valid_step[:, :-1])))

    return loss_global + loss_step + (0.5 * loss_mono)

# --- DATASET ---
class LogisticsDataset(Dataset):
    def __init__(self, packages_df, scans_df, trigger_info, maps, max_seq_len=64):
        self.max_seq_len = max_seq_len
        self.scans = scans_df.sort_values(['PackageID', 'ScanTime']).groupby('PackageID')
        self.package_ids = list(self.scans.groups.keys())
        self.maps = maps
        self.labels = packages_df.set_index('PackageID')['FailedService'].to_dict()
        self.commit_dates = packages_df.set_index('PackageID')['Commit_Date'].to_dict()
        self.destinations = packages_df.set_index('PackageID')['Destination'].to_dict()

        # Trigger times for step-level labels
        if trigger_info is not None and len(trigger_info) > 0:
            self.trigger_times = trigger_info.set_index('PackageID')['Trigger_Time'].to_dict()
        else:
            self.trigger_times = {}

    def __len__(self): return len(self.package_ids)

    def __getitem__(self, idx):
        pkg_id = self.package_ids[idx]
        journey = self.scans.get_group(pkg_id)

        pkg_label = self.labels[pkg_id]
        commit_date = self.commit_dates[pkg_id]
        dest_id = self.maps['loc'].get(self.destinations[pkg_id], 0)

        trigger_time = self.trigger_times.get(pkg_id, None)
        if trigger_time is not None:
            trigger_time = pd.to_datetime(trigger_time)

        locs, events, lanes, dests, hours, days, ctx, step_labels = [], [], [], [], [], [], [], []
        prev_time = None

        for _, row in journey.iterrows():
            curr = pd.to_datetime(row['ScanTime'])
            locs.append(self.maps['loc'].get(row['LocationID'], 0))
            events.append(self.maps['event'].get(row['Event'], 0))
            lanes.append(self.maps['lane'].get(row.get('LaneID', 'NONE'), 0))
            dests.append(dest_id)
            hours.append(curr.hour + curr.minute / 60.0)
            days.append(curr.weekday())

            # Context features
            delta = (curr - prev_time).total_seconds() / 3600.0 if prev_time else 0.0
            prev_time = curr
            days_until_commit = (commit_date - curr.date()).days
            ctx.append([delta, float(days_until_commit)])

            # Step-level labels
            if pkg_label == 0:
                step_labels.append(0.0)
            else:
                if trigger_time is not None:
                    step_labels.append(1.0 if curr >= trigger_time else 0.0)
                else:
                    step_labels.append(1.0)

        # Padding
        seq_len = len(locs)
        pad = self.max_seq_len - seq_len
        if pad > 0:
            locs += [0] * pad
            events += [0] * pad
            lanes += [0] * pad
            dests += [0] * pad
            hours += [0.0] * pad
            days += [0.0] * pad
            ctx += [[0.0] * 2] * pad
            step_labels += [0.0] * pad
            mask = [False] * seq_len + [True] * pad
        else:
            limit = self.max_seq_len
            locs = locs[:limit]
            events = events[:limit]
            lanes = lanes[:limit]
            dests = dests[:limit]
            hours = hours[:limit]
            days = days[:limit]
            ctx = ctx[:limit]
            step_labels = step_labels[:limit]
            mask = [False] * limit

        return {
            'locs': torch.tensor(locs),
            'events': torch.tensor(events),
            'lanes': torch.tensor(lanes),
            'dests': torch.tensor(dests),
            'hours': torch.tensor(hours),
            'days': torch.tensor(days),
            'context': torch.tensor(ctx),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'label': torch.tensor(self.labels[pkg_id], dtype=torch.float32),
            'step_targets': torch.tensor(step_labels, dtype=torch.float32),
        }

# --- SETUP SYSTEM (SUPERVISED) ---
def setup_system(packages_df, scans_df, trigger_info=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    maps = {
        'loc': {l: i for i, l in enumerate(scans_df['LocationID'].unique())},
        'event': {e: i for i, e in enumerate(scans_df['Event'].unique())},
        'lane': {l: i for i, l in enumerate(scans_df['LaneID'].fillna('NONE').unique())}
    }

    pids = packages_df['PackageID'].unique()
    split = int(len(pids) * 0.8)
    train_ids, val_ids = pids[:split], pids[split:]

    print(f"Supervised Training on {len(train_ids)} packages (Mixed Success/Failure)")

    train_ds = LogisticsDataset(
        packages_df[packages_df['PackageID'].isin(train_ids)],
        scans_df[scans_df['PackageID'].isin(train_ids)],
        trigger_info, maps
    )
    val_ds = LogisticsDataset(
        packages_df[packages_df['PackageID'].isin(val_ids)],
        scans_df[scans_df['PackageID'].isin(val_ids)],
        trigger_info, maps
    )

    loaders = {
        'train': DataLoader(train_ds, batch_size=32, shuffle=True),
        'val': DataLoader(val_ds, batch_size=32)
    }

    config = DCLTConfig(len(maps['loc']), len(maps['event']), len(maps['lane']))
    model = DeepContextualLogisticsTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    return model, optimizer, loaders, device
