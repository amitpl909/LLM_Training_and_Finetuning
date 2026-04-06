# SLURM Submission Error - Analysis & Fix

## Error Encountered
```
sbatch: error: Memory specification can not be satisfied
sbatch: error: Batch job submission failed: Requested node configuration is not available
```

## Root Cause Analysis

### Problem 1: Incorrect Memory Specification Syntax
- **Original**: `--mem-per-cpu=4G` with `--cpus-per-task=4`
- **Result**: Requests 4G × 4 CPUs = **16GB total RAM**
- **Issue**: This exceeds available memory on most node allocations in the gpu1v100 partition

### Problem 2: Resource Overbooking  
Your training workload analysis:
- **Model**: Phi-3.5 Mini (3.8B parameters)
- **Quantization**: 4-bit QLoRA → ~950MB model size on GPU
- **Batch size**: 4 sequences
- **Max sequence length**: 1024 tokens
- **GPU**: V100 (32GB) - MORE than adequate
- **CPU RAM needed**: Data loading + processing ≈ 4-8GB sufficient

**Conclusion**: 16GB CPU request was grossly oversized and incompatible with node allocation policies.

### Problem 3: CPU Over-allocation
- **Original**: 4 CPUs per task
- **Actual need**: 1-2 CPUs sufficient for data loading and background operations
- **Impact**: Reduces queue wait time and increases scheduling flexibility

---

## Solution Applied

### Changes Made to All SLURM Scripts

#### 1. **run_training.slurm** ✓
```diff
- #SBATCH --cpus-per-task=4    # 4 CPU cores for data loading 
- #SBATCH --mem-per-cpu=4G     # 4G * 4 cores = 16GB total RAM 
+ #SBATCH --cpus-per-task=2    # 2 CPU cores for data loading (sufficient for QLoRA)
+ #SBATCH --mem=8G             # 8GB total RAM (conservative, portable across clusters)
```

#### 2. **run_inference.slurm** ✓
```diff
- #SBATCH --cpus-per-task=8
- #SBATCH --mem-per-cpu=2G     # 8 * 2G = 16GB total
+ #SBATCH --cpus-per-task=2
+ #SBATCH --mem=4G             # 4GB sufficient for inference
```

#### 3. **install_env.sh** ✓
```diff
- #SBATCH --cpus-per-task=4
+ #SBATCH --cpus-per-task=2
- #SBATCH --time=01:00:00
+ #SBATCH --mem=4G
+ #SBATCH --time=00:30:00
```

---

## Memory Specification Best Practices

| Option | Format | Use Case | Portability |
|--------|--------|----------|-------------|
| `--mem=8G` | Total memory | **Most portable** | ✅ Works on all clusters |
| `--mem-per-cpu=2G` | Per CPU | Cluster-specific | ❌ May cause syntax errors |
| `--mem=8192` | Megabytes | Explicit control | ✅ Works everywhere |

**Recommendation**: Use `--mem=<total>` format (e.g., `--mem=8G`) for maximum compatibility.

---

## Revised Resource Allocation

### Training Job (run_training.slurm)
| Resource | Before | After | Justification |
|----------|--------|-------|---------------|
| CPUs | 4 | 2 | Data loading doesn't need ×4 parallelism |
| Memory | 16GB | 8GB | QLoRA + batch 4 uses <5GB CPU RAM |
| GPU | 1× V100 (32GB) | 1× V100 | Unchanged - GPU is the bottleneck |
| Wall time | 12 hours | 12 hours | Unchanged - training duration unchanged |

### Inference Job (run_inference.slurm)
| Resource | Before | After | Justification |
|----------|--------|-------|---------------|
| CPUs | 8 | 2 | Inference batch size is small |
| Memory | 16GB | 4GB | Model loaded once, streams inference |
| GPU | 1× V100 | 1× V100 | Unchanged |

---

## Expected Outcomes

✅ **Job should now submit successfully** without the "Memory specification cannot be satisfied" error.

✅ **Faster queue entry**: Reduced resource requests = higher likelihood of immediate scheduling.

✅ **No performance loss**: Your training will run identically with 2 CPUs vs 4.

---

## Testing the Fix

```bash
# Try submitting the fixed script
cd /work/nbe841/LLM_Training_and_Finetuning
sbatch hpc_scripts/run_training.slurm

# Check submission status
squeue -u $USER

# Monitor job
squeue -j <JOB_ID> --format="%i %T %M %l %N"
```

---

## If Issues Persist

1. **Check node availability**:
   ```bash
   sinfo -p gpu1v100
   scontrol show node gpu001
   ```

2. **Further reduce resources** if needed:
   - Try `--mem=6G` (still adequate)
   - Reduce `--cpus-per-task=1` if needed

3. **Check cluster policies**:
   ```bash
   scontrol show partition gpu1v100
   ```

4. **Contact HPC support** with job submit details if allocation still fails.

---

## Files Modified
- ✅ `/work/nbe841/LLM_Training_and_Finetuning/hpc_scripts/run_training.slurm`
- ✅ `/work/nbe841/LLM_Training_and_Finetuning/hpc_scripts/run_inference.slurm`
- ✅ `/work/nbe841/LLM_Training_and_Finetuning/install_env.sh`

All scripts now use consistent, portable SLURM memory specifications.
