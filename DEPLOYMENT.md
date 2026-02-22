# DeepContextualLogisticsTransformer Deployment Guide

## Overview

This guide covers deploying your trained model for both **batch jobs** and **online inference** on GPU servers.

---

## Prerequisites

- Python 3.8+
- PyTorch 2.0+ (for TorchScript)
- CUDA 11.0+ (for GPU inference)
- TensorRT (optional, for maximum inference performance)

---

## Model Files

After running `export_to_tensorrt.py`, you'll have:

- `dclt_model.ts` - TorchScript model (portable, no Python dependency)
- `dclt_model.engine` - TensorRT engine (maximum performance, GPU-specific)

---

## Batch Jobs (Training, Offline Processing)

### Best Approach: `torch.compile()`

For batch processing with minimal changes to your code:

```python
import torch
from build_train_model import DeepContextualLogisticsTransformer, DCLTConfig

# Load model
config = DCLTConfig(num_locs=1000, num_events=50, num_lanes=20)
model = DeepContextualLogisticsTransformer(config)
model.load_state_dict(torch.load("training_logs/checkpoints/best_model.pt"))
model.eval()

# Compile for batch processing
model = torch.compile(model, mode="reduce-overhead")

# Run batch inference
with torch.no_grad():
    step_risk, global_risk = model(batch)
```

**Benefits:**
- Zero code changes (just add `torch.compile()`)
- Automatic kernel fusion via TorchInductor
- No additional dependencies

---

## Online Inference (Low-Latency Serving)

### Option 1: TorchScript (Recommended)

**Export:**
```bash
python export_to_tensorrt.py
```

**Inference:**
```python
from inference_runtime import TorchScriptInference

inferencer = TorchScriptInference("dclt_model.ts", device='cuda')
step_risk, global_risk = inferencer.inference(batch)
```

**Benefits:**
- No Python dependency (can use C++ runtime with libtorch)
- Significant speedup over raw PyTorch
- Easy to deploy

**C++ Deployment:**
```cpp
#include <torch/script.h>

int main() {
    torch::jit::script::Module module = torch::jit::load("dclt_model.ts");
    // Run inference...
}
```

---

### Option 2: TensorRT (Maximum Performance)

**Install TensorRT:**
```bash
# Download from NVIDIA
pip install tensorrt torch2trt
```

**Convert TorchScript to TensorRT:**
```python
# Requires modifying model.forward() to accept tensor args instead of dict
# See export_to_tensorrt.py for details
```

**Inference:**
```python
from inference_runtime import TensorRTInference

inferencer = TensorRTInference("dclt_model.engine", device='cuda')
step_risk, global_risk = inferencer.inference(batch)
```

**Benefits:**
- 2-5x faster than TorchScript
- Kernel fusion and precision calibration (FP16/INT8)
- Ideal for high-throughput scenarios

---

## Benchmarking

Run performance benchmarks:

```bash
python inference_runtime.py --model dclt_model.ts --benchmark
```

Test different batch sizes:
```bash
python inference_runtime.py --model dclt_model.ts --batch-size 128 --seq-len 64
```

---

## Performance Comparison

| Approach | Latency | Throughput | Complexity | Use Case |
|----------|---------|------------|------------|----------|
| Raw PyTorch | High | Low | None | Development |
| TorchCompile | Medium | Medium | Low | Batch jobs |
| TorchScript | Low | Medium | Medium | General serving |
| TensorRT | Very Low | Very High | High | High-scale production |

---

## Recommended Production Setup

**For Batch Processing:**
```
torch.compile(model, mode="reduce-overhead")
→ Run on GPU with PyTorch 2.0+
```

**For Online Inference:**
```
1. Export: python export_to_tensorrt.py
2. Deploy: TorchScript (easier) or TensorRT (faster)
3. Monitor: Use nvidia-smi or Prometheus metrics
```

---

## Troubleshooting

**Model fails to export:**
- Ensure `model.eval()` is called before export
- Check all input tensors are on same device

**TensorRT conversion fails:**
- Your model uses dict input - modify `forward()` to accept tensor args
- Or stick with TorchScript (no conversion needed)

**Performance not improved:**
- Verify GPU is being used (`device='cuda'`)
- Try different `torch.compile()` modes: "default", "reduce-overhead", "max-autotune"

---

## Next Steps

1. Run `python export_to_tensorrt.py` to export your trained model
2. Test TorchScript inference with `python inference_runtime.py --model dclt_model.ts`
3. If you need maximum performance, modify `forward()` to accept tensor arguments and try TensorRT
4. Set up monitoring with `nvidia-smi` or Prometheus + GPU exporter
