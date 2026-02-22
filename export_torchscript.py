"""
Export DeepContextualLogisticsTransformer to TorchScript.
This script creates a wrapper that's compatible with TorchScript.
"""

import os
import torch
import torch.nn as nn
import math
from build_train_model import DeepContextualLogisticsTransformer, DCLTConfig


class DCLTForExport(nn.Module):
    """
    TorchScript-compatible version of DCLT.
    Uses explicit tensor inputs instead of dict.
    """
    def __init__(self, config):
        super().__init__()
        # Embeddings
        self.emb_loc = nn.Embedding(config.num_locs, 64, padding_idx=0)
        self.emb_event = nn.Embedding(config.num_events, 32, padding_idx=0)
        self.emb_lane = nn.Embedding(config.num_lanes, 32, padding_idx=0)
        
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
        self.step_head = nn.Sequential(nn.Linear(config.d_model, 128), nn.GELU(), nn.Linear(128, 1), nn.Sigmoid())
        self.global_head = nn.Sequential(nn.Linear(config.d_model, 1), nn.Sigmoid())

    def forward(self, locs, events, lanes, hours, days, context, mask):
        # Embeddings
        loc_emb = self.emb_loc(locs)
        event_emb = self.emb_event(events)
        lane_emb = self.emb_lane(lanes)
        time_emb = self.time_encoder(hours, days)
        ctx_emb = self.context_proj(context)
        
        # Concatenate all features
        x = torch.cat([loc_emb, event_emb, lane_emb, time_emb, ctx_emb], dim=-1)
        x = self.fusion(x)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.dropout(x)
        
        # Transformer
        out = self.transformer(x, src_key_padding_mask=mask)
        
        # Step-wise risk
        step_risk = self.step_head(out).squeeze(-1)
        
        # Global pooling (Max Risk strategy)
        masked_out = out.masked_fill(mask.unsqueeze(-1), -1e9)
        global_pool, _ = torch.max(masked_out, dim=1)
        global_risk = self.global_head(global_pool).squeeze(-1)
        
        return step_risk, global_risk


class CyclicalTimeEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.w_hour = nn.Parameter(torch.randn(out_dim // 4)) 
        self.w_day = nn.Parameter(torch.randn(out_dim // 4)) 
    
    def forward(self, hours: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        h_emb = torch.cat([
            torch.sin(2 * math.pi * hours.unsqueeze(-1) * self.w_hour), 
            torch.cos(2 * math.pi * hours.unsqueeze(-1) * self.w_hour)
        ], dim=-1)
        d_emb = torch.cat([
            torch.sin(2 * math.pi * days.unsqueeze(-1) * self.w_day), 
            torch.cos(2 * math.pi * days.unsqueeze(-1) * self.w_day)
        ], dim=-1)
        return torch.cat([h_emb, d_emb], dim=-1)


class ContextProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), 
            nn.LayerNorm(out_dim), 
            nn.GELU(), 
            nn.Linear(out_dim, out_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def extract_vocab_sizes_from_scans(scans_df):
    """Extract vocabulary sizes from scans dataframe"""
    maps = {
        'loc': {l: i for i, l in enumerate(scans_df['LocationID'].unique(), start=1)},
        'event': {e: i for i, e in enumerate(scans_df['Event'].unique(), start=1)},
        'lane': {l: i for i, l in enumerate(scans_df['LaneID'].fillna('NONE').unique(), start=1)}
    }
    return len(maps['loc']), len(maps['event']), len(maps['lane']), maps


def load_trained_weights(model, checkpoint_path, device='cpu'):
    """Load weights from trained model into export model"""
    import generate_data
    
    # Generate sample data to extract vocab sizes
    packages_df, scans_df, calendar_df, maps = generate_data.run_simulation()
    num_locs, num_events, num_lanes, _ = extract_vocab_sizes_from_scans(scans_df)
    
    # Create original model and load weights
    original_config = DCLTConfig(num_locs, num_events, num_lanes)
    original_model = DeepContextualLogisticsTransformer(original_config)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            original_model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict):
            original_model.load_state_dict(checkpoint)
        else:
            original_model.load_state_dict(checkpoint)
    
    # Copy weights to export model
    model.load_state_dict(original_model.state_dict())
    return model


def main():
    config_path = "training_logs/run_20260221_202559/model.pt"
    output_path = "dclt_model.ts"
    
    print(f"Exporting model to {output_path}")
    
    # Use default vocab sizes (extracted from simulation)
    num_locs = 100
    num_events = 2
    num_lanes = 91
    d_model = 128
    nhead = 4
    num_layers = 3
    dropout = 0.2
    
    config = type('Config', (), {
        'num_locs': num_locs,
        'num_events': num_events,
        'num_lanes': num_lanes,
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_layers,
        'dropout': dropout
    })()
    
    # Create export model
    model = DCLTForExport(config)
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Try to load trained weights
    try:
        model = load_trained_weights(model, config_path)
        print("Loaded trained weights")
    except Exception as e:
        print(f"Could not load trained weights: {e}")
        print("Using untrained model")
    
    model.eval()
    
    # Export to TorchScript
    print("Exporting to TorchScript...")
    with torch.no_grad():
        dummy_batch = {
            'locs': torch.randint(1, num_locs, (1, 64)),
            'events': torch.randint(1, num_events, (1, 64)),
            'lanes': torch.randint(1, num_lanes, (1, 64)),
            'hours': torch.rand(1, 64),
            'days': torch.rand(1, 64),
            'context': torch.rand(1, 64, 4),
            'mask': torch.zeros(1, 64, dtype=torch.bool)
        }
        
        # Tracing (works better than scripting for dict models)
        traced_model = torch.jit.trace(model, (
            dummy_batch['locs'],
            dummy_batch['events'],
            dummy_batch['lanes'],
            dummy_batch['hours'],
            dummy_batch['days'],
            dummy_batch['context'],
            dummy_batch['mask']
        ))
        
        traced_model.save(output_path)
        print(f"TorchScript saved to: {output_path}")
        
        # Verify
        step_risk, global_risk = traced_model(
            dummy_batch['locs'],
            dummy_batch['events'],
            dummy_batch['lanes'],
            dummy_batch['hours'],
            dummy_batch['days'],
            dummy_batch['context'],
            dummy_batch['mask']
        )
        print(f"Verification - step_risk: {step_risk.shape}, global_risk: {global_risk.shape}")


if __name__ == "__main__":
    main()
