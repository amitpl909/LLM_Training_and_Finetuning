# GPU Availability Fix - April 7, 2026

## Problem Diagnosed

The system had a **PyTorch-to-NVIDIA Driver version mismatch**:

- **NVIDIA Driver**: 545.23.08 (supports CUDA 12.3)
- **PyTorch initially installed**: Compiled for CUDA 13.0
- **Error**: "NVIDIA driver on your system is too old (found version 12030)"

PyTorch 13.0 was incompatible with the available CUDA 12.3 drivers, causing `torch.cuda.is_available()` to return `False`.

## Root Cause

When pip installed PyTorch without specifying a CUDA version, it defaulted to the latest PyTorch with CUDA 13.0. However, the UTSA HPC system's NVIDIA driver only supports CUDA up to 12.3.

## Solution Applied

**Reinstalled PyTorch for CUDA 12.1** (compatible with CUDA 12.3 drivers):

```bash
# Remove old PyTorch
pip uninstall -y torch torchvision torchaudio

# Install compatible version for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Verification

After fix, GPU is fully operational:

```
✅ CUDA Available: True
   GPU: Tesla V100S-PCIE-32GB (34.08GB memory)
   CUDA Version: 12.1
   GPU Device Count: 1

✅ Model Loading Test:
   - Phi-3.5-mini-instruct loaded successfully
   - 4-bit quantization working
   - LoRA adapters applied
   - Trainable parameters: 3,145,728 (0.0823% of 3.8B)
```

## Setup Instructions for Future Use

To reproduce this environment:

1. **Create conda environment:**
   ```bash
   conda create -n llm_env python=3.10
   conda activate llm_env
   ```

2. **Install PyTorch with CUDA 12.1 support:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install other dependencies:**
   ```bash
   pip install transformers peft trl datasets bitsandbytes accelerate openai evaluate rouge_score bert_score pyyaml
   ```

## Important Notes

- **Always specify CUDA version** when installing PyTorch on HPC systems
- The `--index-url` flag tells pip to use PyTorch's official wheel repository
- CUDA 12.1 wheels are fully compatible with CUDA 12.3 drivers (backward compatible)
- Never rely on pip's default PyTorch version on systems with specific driver versions

## Timeline

- **April 6**: Initial training run (Job 724031) fell back to CPU due to GPU unavailability
- **April 7**: Diagnosed version mismatch and fixed by reinstalling PyTorch for CUDA 12.1
- **April 7**: GPU fully operational and ready for training

## Next Steps

Now that GPU is available, you can:
1. Resubmit training job to SLURM: `sbatch slurm/stage1_alpaca.sh`
2. Training will now use GPU (100x+ faster than CPU)
3. Complete evaluation and submission before deadline
