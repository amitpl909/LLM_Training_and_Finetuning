# Sequential Instruction Tuning Pipeline: Small-Model Post-Training with Catastrophic Forgetting Analysis

**Graduate-Level Research Implementation** | UTSA LLM & Agentic Systems Assignment 3

**Research Question**: Does fine-tuning a small LLM first on general instruction data (Alpaca) and then on task-specific JSON outputs preserve general instruction-following ability, or does catastrophic forgetting degrade prior gains?

**Status**: ✅ Implementation complete | Training active (Job 724031) | Blog post at [REPORT.md](REPORT.md)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [System Architecture](#system-architecture)
5. [Usage Guide](#usage-guide)
6. [Configuration](#configuration)
7. [Results & Blog Post](#results--blog-post)
8. [Project Structure](#project-structure)
9. [Troubleshooting](#troubleshooting)
10. [References](#references)

---

## Quick Start

**TL;DR** for users who want to run this end-to-end:

```bash
# 1. Setup (5 min)
git clone https://github.com/amitpl909/LLM_Training_and_Finetuning.git
cd LLM_Training_and_Finetuning
module load anaconda3
conda activate llm_env
pip install -r requirements.txt

# 2. Prepare data (10 min)
python data_prep/1a_prep_alpaca.py              # ~1 min download
# JSON data pre-generated; run this if needed: python data_prep/1b_generate_json_instruct.py

# 3. Train on UTSA HPC (6 hours)
sbatch hpc_scripts/run_training.slurm
squeue -j <JOBID>                              # Monitor progress
tail -f logs/training_<JOBID>.log

# 4. Evaluate (30 min, after training completes)
python evaluation/inference.py
python evaluation/llm_judge.py

# 5. Read results
cat REPORT.md                                  # Full blog post with findings
```

---

## Requirements

### System Requirements

| Requirement | Minimum | Recommended | Notes |
|-------------|---------|-------------|-------|
| **GPU** | 16 GB | 32 GB (V100) | Required for training. 4-bit quantization enables 16GB.|
| **RAM** | 32 GB | 64 GB | For model loading and inference. |
| **Disk** | 100 GB | 200 GB | For datasets, checkpoints, and logs. |
| **Network** | Stable | High-speed | For downloading models and API calls. |

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| **Python** | 3.10+ | Runtime environment |
| **CUDA** | 11.8+ | GPU computation (for PyTorch) |
| **cuDNN** | 8.0+ | Deep learning library |
| **Conda** | 4.10+ | Package management (recommended) |
| **Git** | 2.25+ | Repository cloning |

### Python Dependencies

All are listed in `requirements.txt`. See [Installation](#installation) for setup.

```
torch>=2.0.0                    # Deep learning framework
transformers>=4.30.0            # HuggingFace models
peft>=0.4.0                     # Parameter-efficient fine-tuning (LoRA)
datasets>=2.13.0                # Dataset loading utilities
bitsandbytes>=0.39.0            # 4-bit quantization
accelerate>=0.20.0              # Distributed training
trl>=0.5.0                      # Training utilities
pyyaml>=6.0                     # Configuration parsing
tqdm>=4.62.0                    # Progress bars
openai>=0.27.0                  # LLM API access
rouge_score>=0.1.2              # ROUGE metric calculation
bert_score>=0.3.13              # BERTScore metric calculation
```

### API Access

You need API credentials for:

1. **Teacher Model** (for generating JSON Instruct dataset in Phase 1b)
   - Model: Llama-3.3-70B-Instruct (quantized)
   - Access: Provided in `config.yaml`

2. **Judge Model** (for evaluation in Phase 4)
   - Model: Llama-3.3-70B-Instruct
   - Access: Provided in `config.yaml`

These are provided in the assignment specification. See [Configuration](#configuration).

### HPC Cluster Access

- **Cluster**: UTSA HPC
- **Login nodes**: login001, login002
- **Compute nodes**: gpu001-gpu022 (V100 GPUs)
- **Required partitions**: gpu1v100

---

## Installation

### Step 1: Clone Repository

```bash
# Clone repository
git clone https://github.com/amitpl909/LLM_Training_and_Finetuning.git
cd LLM_Training_and_Finetuning

# Verify structure
ls -la
# Expected: data_prep/, training/, evaluation/, hpc_scripts/, config.yaml, etc.
```

### Step 2: Load Required Modules (UTSA HPC Only)

```bash
# Load anaconda/python module
module load anaconda3

# Verify module loaded
python --version  # Should be 3.10+
```

### Step 3: Create & Activate Conda Environment

```bash
# Create isolated Python environment
conda create -n llm_env python=3.10 -y

# Activate environment
conda activate llm_env

# Verify activation
which python  # Should show path ending in llm_env/bin/python
```

### Step 4: Install Python Dependencies

**For UTSA HPC (or systems with NVIDIA driver CUDA 12.1-12.3):**

```bash
# Install PyTorch for CUDA 12.1 compatibility
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install remaining dependencies
pip install transformers peft trl datasets bitsandbytes accelerate openai evaluate rouge_score bert_score pyyaml

# Verify installation (should complete without errors)
python -c "import torch, transformers, peft, datasets, bitsandbytes; print('✅ All imports successful')"
```

**For other systems**, you can use the standard requirements file (adjust for your CUDA version):

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU-only (slow, not recommended for large models)
pip install torch torchvision torchaudio
```

> **Note**: If training reports `CUDA Available: False`, you likely have a version mismatch. Check [GPU_FIX_EXPLANATION.md](GPU_FIX_EXPLANATION.md) for diagnosis steps.

### Step 5: Verify GPU Access (Optional but Recommended)

```bash
# Check CUDA availability and GPU info
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Expected output: CUDA available: True, GPU: NVIDIA V100-PCIE-32GB (or similar)
```

### Troubleshooting Installation

**Issue**: `ImportError: No module named 'datasets'`
- **Solution**: Ensure `pip install -r requirements.txt` completed without errors. Run again if needed.

**Issue**: `CUDA not available` when running training
- **Solution**: Ensure you're on a compute node with GPU access. CUDA is not available on login nodes. Use `sbatch` to submit jobs to GPU nodes.

**Issue**: `ModuleNotFoundError: No module named 'data_utils'`
- **Solution**: This is fixed in the updated code. If you cloned an older version, run `git pull` to get the latest fixes.

---

## System Architecture

### Overview

```
Input                 Processing              Output
─────────────         ──────────────          ──────

Alpaca Dataset  ──→  Stage 1 Training    ──→  Checkpoint 1
(51k examples)       (QLoRA, 2 epochs)        (Alpaca-Tuned Model)
                            │
                            ├─ Evaluate on Alpaca & JSON tasks
                            │
                            ▼
Teacher Model      Stage 2 Training        Checkpoint 2
+ Prompts    ──→   (QLoRA continued)   ──→  (JSON-Tuned Model)
(JSON tasks)        (2 epochs on 50          
                     JSON examples)          
                            │
                            ├─ Evaluate on Alpaca & JSON tasks
                            │
                            ▼
                    Judge Evaluation    ──→  Results Table
                    + Metrics                Forgetting Analysis
                    + Analysis               Blog Post (REPORT.md)
```

### Data Flow

```
Phase 1: Data Construction (Hours 0-1)
├── 1a_prep_alpaca.py
│   ├── Input: Hugging Face alpaca-cleaned dataset
│   └── Output: alpaca_train.json, alpaca_eval.json
│
└── 1b_generate_json_instruct.py
    ├── Input: Prompts + Teacher API (Llama 70B)
    ├── Process: Generate + validate JSON responses
    └── Output: stage2_json_instruct_train.json, _eval.json

Phase 2-3: Training (Hours 1-7, running on UTSA HPC)
├── stage1_alpaca.py
│   ├── Input: alpaca_train.json + Phi-3.5 base model
│   ├── Process: QLoRA fine-tuning for 2 epochs
│   └── Output: checkpoints/stage1_alpaca_final/
│
└── stage2_json.py
    ├── Input: stage1_alpaca_final + stage2_json_instruct_train.json
    ├── Process: Continue QLoRA fine-tuning for 2 epochs
    └── Output: checkpoints/stage2_json_final/

Phase 4: Evaluation (Hours 7-9)
├── inference.py
│   ├── Input: 3 model checkpoints + eval datasets
│   └── Output: results/inference_results.json
│
├── llm_judge.py
│   ├── Input: inference_results.json + Judge API (Llama 70B)
│   ├── Process: Pairwise comparison + metrics calculation
│   └── Output: results/judge_scores.json, metrics_table.json
│
└── forgetting_analysis.py
    ├── Input: judge_scores.json
    ├── Process: Compare Checkpoint 1 vs 2 on Alpaca tasks
    └── Output: REPORT.md (updated with findings)
```

### Component Functions

| File | Phase | Function | Input | Output |
|------|-------|----------|-------|--------|
| `data_prep/1a_prep_alpaca.py` | 1a | Download & normalize Alpaca dataset | HuggingFace API | alpaca_*.json |
| `data_prep/1b_generate_json_instruct.py` | 1b | Teacher-generated JSON examples | Prompts + Teacher API | stage2_json_instruct_*.json |
| `training/stage1_alpaca.py` | 2 | QLoRA fine-tuning on Alpaca | Base model + Alpaca data | stage1 adapter checkpoints |
| `training/stage2_json.py` | 3 | Continue QLoRA on JSON data | Stage 1 adapter + JSON data | stage2 adapter checkpoints |
| `evaluation/inference.py` | 4a | Generate responses at 3 checkpoints | All 3 models + eval prompts | inference_results.json |
| `evaluation/llm_judge.py` | 4b | Judge evaluation + automatic metrics | inference results + Judge API | judge_scores.json, metrics_table.json |
| `evaluation/forgetting_analysis.py` | 4c | Forgetting analysis & per-category breakdown | judge_scores.json | Analysis output |

---

## Usage Guide

### Phase 1a: Download & Prepare Alpaca Data

**Purpose**: Download the Stanford Alpaca dataset and split into train/evaluation sets.

**Time**: ~1-2 minutes (includes download)

```bash
# Activate environment
conda activate llm_env

# Run script
python data_prep/1a_prep_alpaca.py

# Expected output:
# Downloading Alpaca-cleaned dataset...
# alpaca_data_cleaned.json: 100%|████| 44.3M [00:01<00:00, 42.3MB/s]
# Saved 51,660 training examples to data_prep/alpaca_train.json
# Saved 100 evaluation examples to data_prep/alpaca_eval.json

# Verify files created
ls -lh data_prep/alpaca*.json
```

**What this produces**:
- `alpaca_train.json`: 51,660 general instruction-following examples (43 MB)
- `alpaca_eval.json`: 100 held-out evaluation examples (93 KB)
- Both in schema: `{instruction, input, output}`

---

### Phase 1b: Generate Teacher-Created JSON Instruct Dataset

**Purpose**: Create a specialized JSON-focused dataset using a stronger teacher model (imitation learning).

**Time**: 15-30 minutes (includes API calls to teacher model)

**Prerequisites**: Teacher API must be accessible (provided in `config.yaml`)

```bash
# Activate environment (if not already active)
conda activate llm_env

# Run script
python data_prep/1b_generate_json_instruct.py

# Expected output (example):
# Generating task variations...
# [Entity Extraction] 8/8 prompts generated
# [Schema-Constrained] 7/7 prompts generated
# [Classification] 15/15 prompts generated
# [JSON Repair] 10/10 prompts generated
# [Function Calls] 10/10 prompts generated
# 
# Calling teacher model for generation & validation...
# ✓ Example 1: valid JSON
# ✓ Example 2: valid JSON
# ...
#
# Saved 50 training examples to data_prep/stage2_json_instruct_train.json
# Saved 25 eval examples to data_prep/stage2_json_instruct_eval.json

# Verify files created
head -5 data_prep/stage2_json_instruct_train.json
```

**What this produces**:
- `stage2_json_instruct_train.json`: 50 teacher-generated examples (16 KB)
- `stage2_json_instruct_eval.json`: 25 evaluation examples (7.4 KB)
- All JSON outputs validated for correctness
- Covers 5 task types: extraction, schema, classification, repair, function calls

**Note**: This step is optional if JSON data already exists. Pre-generated data is included in the repo.

---

### Phase 2-3: Training on UTSA HPC

**Purpose**: Fine-tune the student model (Phi-3.5) in two sequential stages.

**Time**: ~6-7 hours total (Stage 1: ~2 hours, Stage 2: ~1 minute)

**Prerequisites**: Data files must exist (Phase 1a and 1b completed)

```bash
# Activate environment
conda activate llm_env

# Submit SLURM training job (runs both Stage 1 and Stage 2)
sbatch hpc_scripts/run_training.slurm

# Expected output:
# Submitted batch job 724031

# Check job status
squeue -j 724031
# Output: JOBID PARTITION NAME USER ST TIME NODES NODELIST
#         724031 gpu1v100 llm_fine user R 0:05 1 gpu005

# Monitor training in real-time
tail -f logs/training_724031.log

# Or check occasionally
tail -30 logs/training_724031.log
```

**What happens during training**:

**Stage 1 (Alpaca Fine-Tuning)**: 
- Duration: ~2 hours
- Dataset: 51,660 Alpaca examples × 2 epochs
- Model: Loads base Phi-3.5 with 4-bit quantization (~1 GB)
- Output: LoRA adapter weights saved to `checkpoints/stage1_alpaca_final/`

**Stage 2 (JSON Instruct Fine-Tuning)**:
- Duration: ~1 minute
- Dataset: 50 JSON examples × 2 epochs
- Model: Loads Stage 1 checkpoint and continues training
- Output: LoRA adapter weights saved to `checkpoints/stage2_json_final/`

**Monitoring**:
```bash
# Check real-time progress
tail -f logs/training_724031.log | grep -E "(Step|epoch|loss)"

# Check if training completed successfully
tail -20 logs/training_724031.log | grep -E "(✅|Complete|Stage 2 failed)"

# Check for errors
grep -i "error\|exception\|traceback" logs/training_724031.log

# See checkpoint sizes
du -sh checkpoints/*/
```

**Troubleshooting**:
- **Job cancelled after 30 mins**: Likely kernel hang (common on old UTSA HPC nodes). Fixed in current version with `num_proc=1` tokenization.
- **Out of memory**: Reduce `batch_size` in `config.yaml` from 4 to 2.
- **Training hangs**: Check `tail logs/training_724031.log` for details. May need to request different node.

---

### Phase 4: Evaluation

**Purpose**: Measure model quality at 3 checkpoints and analyze forgetting.

**Time**: ~30-60 minutes total

**Prerequisites**: Training must be complete (Phase 2-3 finished)

#### 4a: Generate Responses at All Checkpoints

```bash
# Activate environment
conda activate llm_env

# Generate responses from baseline, Stage 1, and Stage 2 models
python evaluation/inference.py

# Expected output:
# Loading checkpoint 0 (baseline)...
# Generating responses on Alpaca eval set... 100%|████| 100/100
# Generating responses on JSON eval set... 100%|████| 25/25
# 
# Loading checkpoint 1 (Alpaca-tuned)...
# Generating responses on Alpaca eval set... 100%|████| 100/100
# Generating responses on JSON eval set... 100%|████| 25/25
# 
# Loading checkpoint 2 (JSON-tuned)...
# Generating responses on Alpaca eval set... 100%|████| 100/100
# Generating responses on JSON eval set... 100%|████| 25/25
#
# Saved results to results/inference_results.json

# Verify results file
wc -l results/inference_results.json
```

#### 4b: Run Judge Evaluation & Metrics

```bash
# Run pairwise judge evaluations and automatic metrics
python evaluation/llm_judge.py

# Expected output:
# Loading inference results...
# 
# Running Alpaca Evaluation (Self-Instruct Protocol)
# CP0 vs CP1 comparison... 100%|████| 100/100
# CP1 vs CP2 comparison... 100%|████| 100/100
# 
# Running JSON Evaluation
# Computing validity, schema, exact match... 100%|████| 25/25
#
# Saved judge scores to results/judge_scores.json
# Saved metrics table to results/metrics_table.json
#
# Primary Finding:
# Alpaca score @ CP1: 78%
# Alpaca score @ CP2: 75%
# Forgetting delta: -3% (MILD FORGETTING OBSERVED)
```

---

## Configuration

All parameters are in `config.yaml`. Key settings:

```yaml
# Model Selection
student_model: "microsoft/Phi-3.5-mini-instruct"     # Student to be fine-tuned
teacher_model: "llama-3.3-70b-instruct-awq"          # For JSON data generation
judge_model: "llama-3.3-70b-instruct-awq"            # For evaluation

# API Endpoints (UTSA HPC provided)
teacher_api_url: "http://10.246.100.230/v1"
judge_api_url: "http://10.246.100.230/v1"
teacher_api_key: "[PROVIDED IN ASSIGNMENT]"
judge_api_key: "[PROVIDED IN ASSIGNMENT]"

# Training Parameters
max_sequence_length: 1024           # Input sequence length
batch_size: 4                       # Per-device batch size
gradient_accumulation_steps: 8      # Larger effective batch
lora_rank: 16                       # LoRA rank r
lora_alpha: 32                      # LoRA scaling
lora_dropout: 0.05                  # Regularization

# Stage 1 (Alpaca)
stage1_learning_rate: 0.00002       # 2e-5 recommended
stage1_epochs: 2                    # 2-3 epochs

# Stage 2 (JSON Instruct)
stage2_learning_rate: 0.00002       # Same as Stage 1
stage2_epochs: 2                    # 2-3 epochs

# Evaluation
num_eval_prompts: 100               # Alpaca eval set size
```

**To modify**:
```bash
# Edit config.yaml
nano config.yaml
# Or use your preferred editor
# Save changes before running training/evaluation
```

---

## Results & Blog Post

### Where to Find Results

All results and analysis are in **[REPORT.md](REPORT.md)** — the blog post that forms the core of your submission.

**REPORT.md contains**:
1. **Methodology** - Detailed explanation of approach (1 page)
2. **Experiments** - Results tables, figures, and metrics (3 pages)
3. **Analysis** - Interpretation of forgetting findings (1 page)
4. **Prompt Engineering** - How prompts were designed and iterated (½ page)
5. **Appendix** - Complete prompt templates and configuration (extras pages)

### Key Results

After evaluation completes, REPORT.md will show:

**Table 1: Three-Checkpoint Comparison**
| Checkpoint | Alpaca Judge Win % | JSON Validity % | Exact Match % |
|-----------|-------------------|-----------------|---------------|
| CP0 (Baseline) | — | — | — |
| CP1 (Alpaca) | ~75-85% | ~20% | ~10% |
| CP2 (JSON) | ~72-80%* | ~95% | ~80% |

*The key finding: Does CP2 < CP1 significantly? This indicates forgetting.

**Central Research Question Answer**: 
- **No significant forgetting**: Model gains JSON capability without losing Alpaca skills ✓
- **Mild forgetting (-5%)**: Acceptable trade-off for JSON improvement ✓
- **Severe forgetting (>-15%)**: Sequential training problematic ✗

---

## Project Structure

```
LLM_Training_and_Finetuning/
│
├── README.md                           ← You are here
├── REPORT.md                           ← Blog post with full findings
├── REPORT.md                           ← Full blog post
├── config.yaml                         ← Configuration file
├── requirements.txt                    ← Dependencies
│
├── data_prep/                          ← Phase 1: Data preparation
│   ├── 1a_prep_alpaca.py               Script to download Alpaca
│   ├── 1b_generate_json_instruct.py    Script to generate JSON data
│   ├── alpaca_train.json               (created after running 1a)
│   ├── alpaca_eval.json                (created after running 1a)
│   ├── stage2_json_instruct_train.json (created after running 1b)
│   └── stage2_json_instruct_eval.json  (created after running 1b)
│
├── training/                           ← Phase 2-3: Fine-tuning
│   ├── stage1_alpaca.py                QLoRA: Stage 1 fine-tuning
│   └── stage2_json.py                  QLoRA: Stage 2 fine-tuning
│
├── evaluation/                         ← Phase 4: Evaluation
│   ├── inference.py                    Generate responses at 3 checkpoints
│   ├── llm_judge.py                    Judge-based evaluation
│   ├── judge.py                        Judge utilities
│   ├── metrics.py                      Automatic metrics
│   ├── forgetting_analysis.py          Forgetting quantification
│   └── ablation_study.py               (optional) Ablation study
│
├── hpc_scripts/                        ← UTSA HPC job submission
│   ├── run_training.slurm              Submit Stage 1 + Stage 2 training
│   └── run_inference.slurm             Submit evaluation job
│
├── src/                                ← Utility modules
│   ├── data_utils.py                   Tokenization & formatting templates
│   └── eval_utils.py                   Evaluation helpers
│
├── prompts/                            ← Prompt templates
│   ├── teacher_prompts.json            Teacher-model prompts for JSON generation
│   └── judge_prompts.json              Judge-model prompts for evaluation
│
├── checkpoints/                        ← Model checkpoints (created during training)
│   ├── stage1_alpaca/                  Intermediate Stage 1 checkpoints
│   ├── stage1_alpaca_final/            Final Stage 1 adapter
│   └── stage2_json_final/              Final Stage 2 adapter
│
├── logs/                               ← Training logs (created during training)
│   ├── training_724031.log             Training output
│   └── training_724031.err             Training errors (if any)
│
└── results/                            ← Evaluation results (created after inference)
    ├── inference_results.json          Responses from 3 checkpoints
    ├── judge_scores.json               Judge evaluation scores
    └── metrics_table.json              Formatted results table
```

---

## Troubleshooting

### Common Issues & Solutions

**Q: Training reports "CUDA Available: False" but GPU is present**
- A: This is a PyTorch-to-NVIDIA driver version mismatch. Detected on UTSA HPC where NVIDIA driver supports CUDA 12.1-12.3 but PyTorch was compiled for CUDA 13.0. **Fix**: See [GPU_FIX_EXPLANATION.md](GPU_FIX_EXPLANATION.md) for diagnosis and solution (reinstall PyTorch for correct CUDA version with `--index-url https://download.pytorch.org/whl/cu121`).

**Q: "ModuleNotFoundError: No module named 'data_utils'"**
- A: This is fixed in the current code with absolute import paths. Run `git pull` to get latest version.

**Q: Training job gets cancelled after 30-60 minutes with SIGTERM**
- A: This is the kernel hang issue (UTSA HPC old kernel). Already fixed in current code with `num_proc=1` tokenization. If still happening, check kernel version with `uname -r`.

**Q: "CUDA out of memory" error during training**
- A: Reduce `batch_size` in `config.yaml` from 4 to 2. May slow training slightly but should fit.

**Q: API authentication error when running Phase 1b or evaluation**
- A: Verify `config.yaml` has correct API keys. They should be provided in the assignment specification.

**Q: "FileNotFoundError: alpaca_train.json not found"**
- A: Run `python data_prep/1a_prep_alpaca.py` first to download Alpaca data.

**Q: Results showing all zeros or "NaN" in metrics**
- A: Check that:
  1. Training actually completed (check logs)
  2. Checkpoints were created: `ls checkpoints/*/`
  3. Inference ran successfully: `ls results/inference_results.json`

**Q: Unsure which GPU node to use**
- A: Don't worry—SLURM automatically assigns an available V100 node. Just submit with `sbatch hpc_scripts/run_training.slurm`.

---

## References

### Key Papers

[1] Hu et al. (2021). **LoRA: Low-Rank Adaptation of Large Language Models**. ICLR 2022.

[2] Dettmers et al. (2023). **QLoRA: Efficient Finetuning of Quantized LLMs**. NeurIPS 2023.

[3] Taori et al. (2023). **Stanford Alpaca: An Instruction-following LLaMA Model**.

[4] Wang et al. (2023). **Self-Instruct: Aligning Language Models with Self-Generated Instructions**. EMNLP 2023.

[5] Rafailov et al. (2024). **From Human Preferences to Post-Training Alignment Pipelines**.

### Related Work

- **Catastrophic Forgetting**: McCloskey & Cohen (1989), Kirkpatrick et al. (2017)
- **Knowledge Distillation**: Hinton et al. (2015), Gu et al. (2024 Survey on LLM-as-Judge)
- **Imitation Learning**: Abbeel & Ng (2004), updated scope for LLMs

---

## Support & Questions

For issues or questions:

1. Check this README's [Troubleshooting](#troubleshooting) section
2. Review the detailed [REPORT.md](REPORT.md) for methodology and findings
3. Check logs: `tail logs/training_<JOBID>.log`
4. Check if data files exist: `ls data_prep/*.json`

---

**Last Updated**: April 7, 2026  
**Assignment**: UTSA LLM & Agentic Systems (Spring 2026)  
**Author**: Amit Paul  
**Repository**: https://github.com/amitpl909/LLM_Training_and_Finetuning
