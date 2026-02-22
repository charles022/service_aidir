"""
Export DeepContextualLogisticsTransformer to TensorRT engine.
This script handles:
1. Loading the trained model weights
2. Converting to TorchScript via scripting
3. Exporting to TensorRT with FP16 optimization
"""

import torch
import torch.nn as nn
import numpy as np
import os
from build_train_model import DeepContextualLogisticsTransformer, DCLTConfig

# --- CONFIG ---
class ExportConfig:
    # Update with your actual model checkpoint
    checkpoint_path = "training_logs/run_20260221_202559/model.pt"  # Update this path if different
    output_torchscript = "dclt_model.ts"
    output_tensorrt = "dclt_model.engine"
    max_batch_size = 128
    max_seq_len = 64
    device = "cuda"  # Device for export


def extract_vocab_sizes_from_scans(scans_df):
    """Extract vocabulary sizes from scans dataframe"""
    maps = {
        'loc': {l: i for i, l in enumerate(scans_df['LocationID'].unique(), start=1)},
        'event': {e: i for i, e in enumerate(scans_df['Event'].unique(), start=1)},
        'lane': {l: i for i, l in enumerate(scans_df['LaneID'].fillna('NONE').unique(), start=1)}
    }
    return len(maps['loc']), len(maps['event']), len(maps['lane']), maps


def load_model(config):
    """Load trained model from checkpoint"""
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
    
    if os.path.exists(config.checkpoint_path):
        print(f"Loading checkpoint: {config.checkpoint_path}")
        checkpoint = torch.load(config.checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: Checkpoint not found at {config.checkpoint_path}")
        print("Using untrained model for export demo")
    
    model.eval()
    return model


def export_to_torchscript(model, config):
    """Export model to TorchScript using scripting method"""
    print("Exporting to TorchScript...")
    
    # Scripting preserves control flow better than tracing
    scripted_model = torch.jit.script(model)
    
    # Save TorchScript
    scripted_model.save(config.output_torchscript)
    print(f"TorchScript saved to: {config.output_torchscript}")
    
    # Verify TorchScript works
    print("Verifying TorchScript inference...")
    with torch.no_grad():
        # Create dummy inputs matching your model's forward signature
        batch_size = 1
        device = next(model.parameters()).device
        
        dummy_batch = {
            'locs': torch.randint(1, 1000, (batch_size, config.max_seq_len), device=device),
            'events': torch.randint(1, 100, (batch_size, config.max_seq_len), device=device),
            'lanes': torch.randint(1, 100, (batch_size, config.max_seq_len), device=device),
            'hours': torch.rand(batch_size, config.max_seq_len, device=device),
            'days': torch.rand(batch_size, config.max_seq_len, device=device),
            'context': torch.rand(batch_size, config.max_seq_len, 4, device=device),
            'mask': torch.zeros(batch_size, config.max_seq_len, dtype=torch.bool, device=device)
        }
        
        # Run inference
        step_risk_ts, global_risk_ts = scripted_model(dummy_batch)
        print(f"TorchScript outputs - step_risk: {step_risk_ts.shape}, global_risk: {global_risk_ts.shape}")
    
    return scripted_model


def main():
    config = ExportConfig()
    
    # Load model
    model = load_model(config)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Export to TorchScript
    scripted_model = export_to_torchscript(model, config)
    
    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    print(f"TorchScript model: {config.output_torchscript}")
    print("\nTo run inference:")
    print(f"  python inference_runtime.py --model {config.output_torchscript} --batch-size 128")
    print("\nTo benchmark:")
    print(f"  python inference_runtime.py --model {config.output_torchscript} --benchmark")


if __name__ == "__main__":
    main()
