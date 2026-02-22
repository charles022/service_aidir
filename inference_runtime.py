"""
TorchScript inference runtime for DeepContextualLogisticsTransformer.
"""

import torch
import numpy as np
import os
import argparse


class TorchScriptInference:
    """Load and run inference on TorchScript model"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        print(f"Loaded TorchScript model from {model_path}")
        print(f"Running on {self.device}")
    
    def inference(self, locs, events, lanes, hours, days, context, mask):
        """Run inference with tensor arguments"""
        with torch.no_grad():
            # Move inputs to device
            locs = locs.to(self.device)
            events = events.to(self.device)
            lanes = lanes.to(self.device)
            hours = hours.to(self.device)
            days = days.to(self.device)
            context = context.to(self.device)
            mask = mask.to(self.device)
            
            step_risk, global_risk = self.model(locs, events, lanes, hours, days, context, mask)
            return step_risk.cpu(), global_risk.cpu()
    
    def warmup(self, batch_size=1, seq_len=64, num_locs=100, num_events=2, num_lanes=91, num_iterations=10):
        """Warmup the model for consistent performance"""
        print(f"Warming up with {num_iterations} iterations...")
        
        dummy_batch = {
            'locs': torch.randint(1, num_locs, (batch_size, seq_len), device=self.device),
            'events': torch.randint(1, num_events, (batch_size, seq_len), device=self.device),
            'lanes': torch.randint(1, num_lanes, (batch_size, seq_len), device=self.device),
            'hours': torch.rand(batch_size, seq_len, device=self.device),
            'days': torch.rand(batch_size, seq_len, device=self.device),
            'context': torch.rand(batch_size, seq_len, 4, device=self.device),
            'mask': torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
        }
        
        for _ in range(num_iterations):
            self.inference(**dummy_batch)
        
        print("Warmup complete!")


def run_benchmark(model_path, batch_sizes=[1, 8, 32, 128], seq_len=64, num_warmup=50, num_iterations=200):
    """Benchmark inference performance across different batch sizes"""
    print("\n" + "="*60)
    print("INFERENCE BENCHMARK")
    print("="*60)
    
    try:
        inferencer = TorchScriptInference(model_path, device='cuda')
        
        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}")
            
            # Create dummy batch with correct vocab sizes
            dummy_batch = {
                'locs': torch.randint(1, 100, (batch_size, seq_len), device='cuda'),
                'events': torch.randint(1, 2, (batch_size, seq_len), device='cuda'),
                'lanes': torch.randint(1, 91, (batch_size, seq_len), device='cuda'),
                'hours': torch.rand(batch_size, seq_len, device='cuda'),
                'days': torch.rand(batch_size, seq_len, device='cuda'),
                'context': torch.rand(batch_size, seq_len, 4, device='cuda'),
                'mask': torch.zeros(batch_size, seq_len, dtype=torch.bool, device='cuda')
            }
            
            # Warmup
            for _ in range(num_warmup):
                inferencer.inference(**dummy_batch)
            
            # Benchmark
            import time
            torch.cuda.synchronize()
            start = time.time()
            
            for _ in range(num_iterations):
                inferencer.inference(**dummy_batch)
            
            torch.cuda.synchronize()
            end = time.time()
            
            total_time = end - start
            avg_time = total_time / num_iterations * 1000  # ms
            throughput = batch_size / (total_time / num_iterations)
            
            print(f"  Avg latency: {avg_time:.2f} ms")
            print(f"  Throughput: {throughput:.1f} samples/sec")
    
    except Exception as e:
        print(f"Benchmark failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on exported model')
    parser.add_argument('--model', type=str, required=True, help='Path to TorchScript model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--seq-len', type=int, default=64, help='Sequence length')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark across batch sizes')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on (cpu/cuda)')
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_benchmark(args.model)
    else:
        inferencer = TorchScriptInference(args.model, device=args.device)
        # Use correct vocab sizes (100 locs, 2 events, 91 lanes)
        inferencer.warmup(batch_size=args.batch_size, seq_len=args.seq_len, num_locs=100, num_events=2, num_lanes=91)
        
        # Create dummy batch with correct vocab sizes
        dummy_batch = {
            'locs': torch.randint(1, 100, (args.batch_size, args.seq_len), device=inferencer.device),
            'events': torch.randint(1, 2, (args.batch_size, args.seq_len), device=inferencer.device),
            'lanes': torch.randint(1, 91, (args.batch_size, args.seq_len), device=inferencer.device),
            'hours': torch.rand(args.batch_size, args.seq_len, device=inferencer.device),
            'days': torch.rand(args.batch_size, args.seq_len, device=inferencer.device),
            'context': torch.rand(args.batch_size, args.seq_len, 4, device=inferencer.device),
            'mask': torch.zeros(args.batch_size, args.seq_len, dtype=torch.bool, device=inferencer.device)
        }
        
        step_risk, global_risk = inferencer.inference(**dummy_batch)
        print(f"\nInference results:")
        print(f"  Step risk shape: {step_risk.shape}")
        print(f"  Global risk shape: {global_risk.shape}")
