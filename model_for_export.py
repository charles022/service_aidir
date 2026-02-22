"""
Model wrapper for TorchScript export that supports both dict and tensor inputs.
This allows exporting to TensorRT which requires fixed input signatures.
"""

import os
import torch
import torch.nn as nn
import math
from build_train_model import DeepContextualLogisticsTransformer, DCLTConfig


class TensorRTCompatibleModel(nn.Module):
    """
    Wrapper that converts dict input to tensor arguments for TensorRT compatibility.
    """
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        
    def forward(self, locs, events, lanes, hours, days, context, mask):
        """Forward with tensor arguments for TensorRT compatibility"""
        batch = {
            'locs': locs,
            'events': events,
            'lanes': lanes,
            'hours': hours,
            'days': days,
            'context': context,
            'mask': mask
        }
        return self.original_model(batch)


def extract_vocab_sizes_from_scans(scans_df):
    """Extract vocabulary sizes from scans dataframe"""
    maps = {
        'loc': {l: i for i, l in enumerate(scans_df['LocationID'].unique(), start=1)},
        'event': {e: i for i, e in enumerate(scans_df['Event'].unique(), start=1)},
        'lane': {l: i for i, l in enumerate(scans_df['LaneID'].fillna('NONE').unique(), start=1)}
    }
    return len(maps['loc']), len(maps['event']), len(maps['lane']), maps


def create_exportable_model(checkpoint_path, device='cuda'):
    """
    Load model and wrap it for export to TorchScript/TensorRT.
    Returns both the original model (for dict input) and TRT-compatible model.
    """
    import generate_data
    
    # Generate sample data to extract vocab sizes
    print("Loading data to extract vocabulary sizes...")
    packages_df, scans_df, calendar_df, maps = generate_data.run_simulation()
    
    num_locs, num_events, num_lanes, _ = extract_vocab_sizes_from_scans(scans_df)
    
    print(f"Vocabulary sizes - locs: {num_locs}, events: {num_events}, lanes: {num_lanes}")
    
    model_config = DCLTConfig(
        num_locs=num_locs,
        num_events=num_events,
        num_lanes=num_lanes
    )
    
    model = DeepContextualLogisticsTransformer(model_config)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
    
    model.eval()
    
    # Wrap for TensorRT compatibility
    trt_model = TensorRTCompatibleModel(model)
    
    return model, trt_model, num_locs, num_events, num_lanes


if __name__ == "__main__":
    # Test the exportable model
    try:
        model, trt_model, num_locs, num_events, num_lanes = create_exportable_model(
            "training_logs/run_20260221_202559/model.pt"
        )
        print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        print(f"Error: {e}")
        print("Using default vocab sizes for export...")
        num_locs, num_events, num_lanes = 100, 5, 90
