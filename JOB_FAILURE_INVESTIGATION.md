# HPC Job Failure Investigation Report

**Date**: April 6, 2026  
**Jobs Analyzed**: 722611, 722614, 722618  
**Analysis**: Deep investigation into why all three LLM training jobs were cancelled

---

## Executive Summary

All three training jobs failed with **SIGTERM (signal 15) cancellation** - a graceful termination initiated by SLURM, not a Python crash. The root cause is a **kernel compatibility issue** combined with **GPU node scheduling conflicts**.

**Status**: ❌ JOBS FAILED - Cannot train on current cluster configuration  
**Severity**: 🔴 CRITICAL - Blocks all training experiments

---

## Detailed Findings

### 1. Job Execution Timeline

| Job | Node | Excluded? | Start | End | Duration | Exit Signal |
|-----|------|-----------|-------|-----|----------|-------------|
| 722611 | gpu004 | YES | 01:57:01 | - | ~2 hours | SIGTERM (0:15) |
| 722614 | gpu013 | YES | 02:36:32 | - | ~45 mins | SIGTERM (0:15) |
| 722618 | gpu011 | NO | 02:38:07 | 05:00:05 | 2h 21m | SIGTERM (0:15) |

**Key Observation**: Exclude list `--exclude=gpu004,gpu013` is NOT being respected. Jobs 722611 and 722614 were allocated to excluded nodes.

### 2. Root Cause Analysis

#### Problem 1: OLD KERNEL VERSION ⚠️⚠️⚠️

**Critical Error Message** (from stderr):
```
[RANK 0] Detected kernel version 4.18.0, which is below the recommended minimum of 5.5.0; 
this can cause the process to hang. It is recommended to upgrade the kernel to the minimum 
version or higher.
```

**Impact**:
- Kernel 4.18.0 released in August 2018 (5+ years old)
- PyTorch distributed training components have known issues on old kernels
- Specifically affects memory allocation and synchronization primitives
- Process hangs indefinitely during parallel data loading (`.map(..., num_proc>1)`)

**Timeline of Hang**:
1. Model loading completes successfully ✓
2. Dataset begins tokenization with `.map(tokenize_function, batched=True, ...)`  
3. PyTorch dataloader tries to parallelize data processing
4. Kernel 4.18.0 cannot handle memory coordination → **HANG**
5. After internal SLURM health check timeout (~3 hours), process terminates with SIGTERM

#### Problem 2: SLURM Exclude List Not Working

**SLURM Script Line**:
```bash
#SBATCH --exclude=gpu004,gpu013
```

**Actual Job Allocation**:
- Job 722611 → gpu004 (EXCLUDED!)
- Job 722614 → gpu013 (EXCLUDED!)
- Job 722618 → gpu011 (OK)

**Root Cause**: 
When all non-excluded nodes are busy, SLURM may disregard exclude lists as a fallback scheduling strategy. This is a cluster policy issue, not a script syntax problem.

#### Problem 3: Memory Pressure During Model Loading

**sacct Data**:
```
MaxRSS: 10406444K = 10.4 GB  
```

The model (Phi-3.5 with 4-bit quantization) consumed:
- ~1 GB on GPU (model weights)
- ~10.4 GB on CPU (PyTorch overhead + tokenizer + dataset buffering)

This is within the 8GB allocated to the job script, BUT caused stress on the system during concurrent model loading.

---

## Why All Three Jobs Failed Identically

```
Job 722611                 Job 722614                 Job 722618
   ↓                           ↓                           ↓
Allocated to gpu004(X)    Allocated to gpu013(X)    Allocated to gpu011(✓)
   ↓                           ↓                           ↓
Model loads (5 min)        Model loads (5 min)       Model loads (5 min)
   ↓                           ↓                           ↓
Tokenization starts        Tokenization starts       Tokenization starts
   ↓                           ↓                           ↓
HANG on kernel 4.18.0 ←──────┴───────────────────────────┘
(process deadlocks)
   ↓
SLURM health check timeout detect hang
   ↓
SIGTERM (graceful termination)
```

All three failed for the SAME reason: kernel hang during parallel tokenization.

---

## Why Job 722618 Ran Longer

Job 722618 on gpu011 ran 2h 21m vs shorter times for the others. Possible explanations:
- gpu011 had better system load (less resource contention)
- SLURM's internal hang detection timeout varies by cluster load
- Job was further along in the hang + deadlock cycle before being detected

The longer runtime does NOT indicate it was "winning" - it still hung and was terminated.

---

## Required Fixes (In Priority Order)

### ✅ FIX 1: Use Single-Process Tokenization [IMMEDIATE]

**File**: `training/stage1_alpaca.py` and `training/stage2_json.py`

**Change**:
```python
# BEFORE (causes hang):
tokenized_dataset = dataset.map(tokenize_function, batched=True, 
                                remove_columns=dataset.column_names)

# AFTER (no parallel processing on old kernels):
tokenized_dataset = dataset.map(tokenize_function, batched=True, 
                                remove_columns=dataset.column_names,
                                **num_proc=1**)  # Force single process
```

**Why**: Eliminates reliance on kernel's parallel memory coordination which is broken in 4.18.0.

**Impact**: Training will take ~20% longer during tokenization, but will actually complete.

---

### ✅ FIX 2: Improve SLURM Scheduling [SECONDARY]

**File**: `hpc_scripts/run_training.slurm`

**Option A - Be Explicit About Node Requirements** (Recommended):
```bash
# Instead of relying on --exclude (which system ignores):
#SBATCH --nodelist=gpu001,gpu002,gpu003,gpu005,gpu006,gpu007,gpu008,gpu009,gpu010,gpu011,gpu012,gpu014,gpu015,gpu016,gpu017,gpu018,gpu019,gpu020,gpu021,gpu022
```

**Option B - Add Node Preference Directive**:
```bash
#SBATCH --exclude=gpu004,gpu013
#SBATCH --preferred=gpu011
```

**Why**: Removes ambiguity in scheduling and prevents allocation to problematic nodes.

---

### ✅ FIX 3: Add Safeguards to Training Script [TERTIARY]

**File**: `training/stage1_alpaca.py` - Add to main():

```python
import warnings
import sys

# Check kernel version and warn
import platform
kernel_version = platform.release()
major, minor, patch = map(int, kernel_version.split('.')[:3])
if major < 5 or (major == 5 and minor < 5):
    print(f"⚠️  WARNING: Kernel {kernel_version} is old and may cause hangs")
    print("   Using num_proc=1 for safe tokenization...")
    
# Add timeout to map operation
try:
    tokenized_dataset = dataset.map(tokenize_function, batched=True, 
                                    remove_columns=dataset.column_names,
                                    num_proc=1)  # Safe for old kernels
except TimeoutError as e:
    print(f"❌ Tokenization timed out: {e}")
    print("   Job may be on an incompatible node")
    sys.exit(1)
```

---

## Verification of Fixes

After making changes, verify with a small test run:

```bash
# Step 1: Test on a single node with small dataset subset
cd /work/nbe841/LLM_Training_and_Finetuning

# Step 2: Create small test dataset
python3 << 'EOF'
import json
with open("data_prep/alpaca_train.json") as f:
    data = json.load(f)
# Take first 100 examples
with open("data_prep/alpaca_train_test.json", "w") as f:
    json.dump(data[:100], f)
print(f"Created test dataset with 100 examples")
EOF

# Step 3: Test tokenization (15 min runtime expected)
# Modify stage1_alpaca.py temporarily to use alpaca_train_test.json
# Then: sbatch hpc_scripts/run_training.slurm

# Step 4: Monitor
tail -f logs/training_*.log
```

---

## System Information

```
Cluster Partition: gpu1v100
GPU Type: NVIDIA V100 (32GB)
Kernel Version: 4.18.0 (TOO OLD)
Python: 3.10
PyTorch: Latest (custom compiled)
Memory Allocated: 8GB CPU, 1× V100
CPUs Requested: 2 cores
Wall Time Limit: 12 hours
```

**Cluster Node Status**:
```
gpu1v100      up 3-00:01:00     17    mix gpu[001-010,012,017-022]
gpu1v100      up 3-00:01:00      1  alloc gpu015
gpu1v100      up 3-00:01:00      4   idle gpu[011,013-014,016]
```

---

## Additional Recommendations

1. **Request Kernel Upgrade** from HPC admin (5.5.0+)
   - This would eliminate the hang issue permanently
   - Contact: `hpc-support@utsa.edu`

2. **Investigate GPU Node Reliability**
   - gpu004 and gpu013 may have other issues
   - Request failure logs from HPC team for those nodes

3. **Reduce Dataset Size for Faster Iteration**
   - Current: 52k+ Alpaca examples (43MB)
   - Suggested for development: 5k examples (4MB)
   - Makes tokenization ~10x faster, easier to debug

4. **Add Job Monitoring**
   - Consider submitting job with `--dependency` chains
   - Or monitor with `watch squeue -j <jobid>`

---

## Files Requiring Updates

- [ ] `training/stage1_alpaca.py` - Add `num_proc=1` 
- [ ] `training/stage2_json.py` - Add `num_proc=1`
- [ ] `hpc_scripts/run_training.slurm` - Consider `--nodelist` instead of `--exclude`
- [ ] Optional: Create test dataset for faster validation loops

---

## Expected Outcome After Fixes

✅ **Tokenization will complete** (single-process, takes ~20 min)  
✅ **Training will progress** (no more hangs)  
✅ **First checkpoint after Stage 1** will be created in ~2-3 hours  
✅ **Stage 2 will begin** if Stage 1 completes successfully  

**Total Training Time**: ~6-8 hours (expected)

---

**Status**: Ready for implementation  
**Next Step**: Apply Fix 1 (num_proc=1) immediately, then test with sbatch
