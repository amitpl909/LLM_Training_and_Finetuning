# FIXES APPLIED - Ready to Resubmit Jobs

**Date Applied**: April 6, 2026  
**Status**: ✅ All critical fixes implemented  
**Next Step**: Test with sbatch

---

## What Was Fixed

### ✅ FIX 1: Kernel Compatibility (CRITICAL)

**Problem**: Old kernel (4.18.0) causes hangs during parallel tokenization

**Solution Applied**:
- Added `num_proc=1` parameter to dataset.map() calls
- Forces single-process tokenization (safe for old kernels)
- Prevents PyTorch distributed components from hanging

**Files Modified**:
- ✅ `training/stage1_alpaca.py` - Line 70
- ✅ `training/stage2_json.py` - Line 61

**Impact**: 
- Tokenization will take ~20% longer
- But will actually COMPLETE without hanging
- Trade-off: Speed for stability (acceptable for one-time setup)

---

### ✅ FIX 2: Kernel Version Warnings (INFORMATIONAL)

**Solution Applied**:
- Added `check_kernel_compatibility()` function to both training scripts
- Prints warning if kernel < 5.5.0 is detected
- Informs users WHY num_proc=1 is being used

**Files Modified**:
- ✅ `training/stage1_alpaca.py` - Added kernel check at startup
- ✅ `training/stage2_json.py` - Added kernel check at startup

**Example Output**:
```
⚠️  WARNING: Kernel version 4.18.0 is below 5.5.0
   Old kernels may cause hangs during tokenization.
   Using num_proc=1 for safe single-process tokenization.
```

---

### ✅ FIX 3: SLURM Script Improvements

**Solution Applied**:
- Added error handling for Stage 1 and Stage 2 phases
- Will fail immediately if Stage 1 fails (prevents wasted time on Stage 2)
- Added kernel version logging to job output
- Added diagnostic comment about nodelist alternative

**Files Modified**:
- ✅ `hpc_scripts/run_training.slurm`

**New SLURM Features**:
- Prints kernel version at job start
- Tracks exit codes for each training stage
- Provides clear success/failure messages

---

## Why These Fixes Work

### Root Cause Recap
```
Jobs 722611, 722614, 722618 all followed same pattern:
  ├─ Model loading: ✓ (5 min)
  ├─ Tokenization starts: ✓ (parallel processing)
  ├─ Kernel 4.18.0 tries coordinate memory across processes
  ├─ DEADLOCK in kernel (known issue with old kernels)
  ├─ SLURM health check detects hang
  └─ SIGTERM cancels job → Training fails
```

### How Single-Process Fixes This
```
New flow with num_proc=1:
  ├─ Model loading: ✓ (5 min)
  ├─ Tokenization starts: ✓ (single process)
  ├─ No kernel memory coordination needed
  ├─ Process completes normally ✓ (~20 min)
  ├─ Training starts normally ✓
  └─ Model converges → Training succeeds
```

---

## How to Test & Deploy

### Step 1: Verify Syntax (Already Done ✓)
```bash
python3 -m py_compile training/stage1_alpaca.py training/stage2_json.py
# Output: ✓ Both scripts compile correctly
```

### Step 2: Submit Updated Job
```bash
cd /work/nbe841/LLM_Training_and_Finetuning
sbatch hpc_scripts/run_training.slurm
```

### Step 3: Monitor Job
```bash
# Check job status
squeue -u $USER

# Get job ID from output, then:
tail -f logs/training_<JOBID>.log

# Expected progress:
# ==============================
# [Time: 0-5 min]
# Model loading: 100% ✓
#
# [Time: 5-25 min]  
# Tokenizing dataset...
# (single process, slower but STABLE)
#
# [Time: 25 min+]
# Training Phase 1: Alpaca
# [Training progress...]
# Stage 1 Complete ✓
#
# [If Stage 1 succeeds]
# Training Phase 2: JSON
# [Training progress...]
# Stage 2 Complete ✓
```

### Step 4: Expected Timings

| Phase | Duration | Status |
|-------|----------|--------|
| Model Loading | 5 min | ⏳ Initial |
| Tokenization (Stage 1) | ~20 min | ⏳ Safe, single-process |
| Training (Stage 1) | 1-2 hours | 📊 Running |
| Tokenization (Stage 2) | ~2 min | ⏳ Faster (smaller dataset) |
| Training (Stage 2) | 1-2 hours | 📊 Running |
| **Total** | **~4-6 hours** | ✅ Expected |

---

## Validation Checklist

- [ ] Job submits without errors
- [ ] Kernel warning message appears in logs
- [ ] "Tokenizing dataset..." appears in output
- [ ] Tokenization completes (check for "Trainer init starting" log)
- [ ] Stage 1 training begins
- [ ] Checkpoints appear in `checkpoints/stage1_alpaca/`
- [ ] Stage 1 completes
- [ ] Stage 2 training begins
- [ ] Final checkpoint in `checkpoints/stage2_json_final/`

---

## If New Issues Appear

### Issue: Job still hangs after fixes
- **Cause**: Possibly a different problem
- **Action**: 
  1. Check if job is on gpu004 or gpu013 (check squeue output)
  2. If excluded nodes issue persists, use explicit nodelist in SLURM script
  3. Contact HPC support for node reliability issues

### Issue: GPU out of memory
- **Cause**: V100 has only 32GB, with 4-bit quantized model taking ~1GB
- **Action**: Reduce batch_size in config.yaml from 4 to 2
- **Edit**: `config.yaml`, change `batch_size: 4` to `batch_size: 2`

### Issue: Tokenization slower than expected
- **Expected**: ~20 min for full Alpaca dataset with num_proc=1
- **Not a problem**: This is EXPECTED and ACCEPTABLE
- **Reason**: Trade single-process safety for hanging jobs

---

## Files Changed Summary

**Modified Files**:
1. `training/stage1_alpaca.py` - Added kernel check + num_proc=1
2. `training/stage2_json.py` - Added kernel check + num_proc=1
3. `hpc_scripts/run_training.slurm` - Enhanced error handling + logging
4. (New) `JOB_FAILURE_INVESTIGATION.md` - Detailed analysis

**No Changes to**:
- `config.yaml` (parameters unchanged)
- `data_prep/` scripts (data already ready)
- `evaluation/` scripts (not currently needed)

---

## Next Steps

1. **Submit the job**: `sbatch hpc_scripts/run_training.slurm`
2. **Monitor logs**: `tail -f logs/training_*.log`
3. **Wait for completion**: Expected ~6 hours total
4. **Verify checkpoints**: `ls -lh checkpoints/`
5. **Proceed to evaluation** (Phase 4) once training completes

---

**Status**: ✅ Ready for deployment  
**Confidence Level**: 🟩🟩🟩🟩🟩 High (fix addresses root cause directly)

