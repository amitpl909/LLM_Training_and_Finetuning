# Assignment Implementation Checklist & Guidance

## Current Status Assessment

### ✅ Completed Components
1. **Project Structure**: Well-organized with separate modules for train, evaluate, data_prep
2. **Config System**: Centralized `config.yaml` with all hyperparameters
3. **SLURM Scripts**: Fixed (see SLURM_FIX_SUMMARY.md)
4. **Alpaca Data Prep**: `data_prep/1a_prep_alpaca.py` (needs review)
5. **Stage 1 Training**: `training/stage1_alpaca.py` (looks complete with QLoRA)
6. **Stage 2 Training**: `training/stage2_json.py` (looks complete)
7. **Test Data Files**: `alpaca_train.json`, `alpaca_eval.json` exist

### ⚠️ Components Needing Attention

## Phase 1: Data Construction (Assignment Section 2)

### 1a. Alpaca Data - Status Check
```python
# In data_prep/1a_prep_alpaca.py
# TODO: Verify:
- [ ] Dataset loads correctly
- [ ] Schema normalizes to (instruction, input, output)
- [ ] Held-out eval set is properly separated from training
- [ ] Data file outputs: alpaca_train.json, alpaca_eval.json
```

**Recommended**: Run this immediately to confirm data prep works:
```bash
python data_prep/1a_prep_alpaca.py
```

### 1b. Teacher-Generated JSON Dataset - **CRITICAL, MISSING**
This is the core of the assignment. You need to:

**Required Implementation**:
```python
# data_prep/1b_generate_json_instruct.py must:

1. Design 5 task types covered by teacher-generated JSON:
   ✓ JSON extraction from unstructured text
   ✓ Schema-constrained generation
   ✓ Exact-label classification with JSON output
   ✓ JSON repair/formatting correction
   ✓ Tool-call argument generation

2. Generate diverse prompts for each type (~100+ total prompts)

3. Call teacher model (Llama 3.1 70B) for each prompt:
   Teacher API: http://10.246.100.230/v1
   Model: Llama-3.1-70B-Instruct-custom
   Key: gpustack_50e00c9281422bc5_0c0696dfcb1696d7635e58a2e56d6282

4. Validate JSON correctness - CRITICAL:
   - Reject invalid JSON responses
   - Regenerate if needed
   - Log rejections

5. Save to: stage2_json_instruct_train.json
   Schema: {instruction, input, output (valid JSON)}

6. Create held-out eval set: stage2_json_instruct_eval.json (~100 prompts)
```

**This is where your assignment gets its novelty and research value.**

---

## Phase 2-3: Training (Assignment Sections 2.2-2.3)

### Current training code status: ✅ Ready to test

Your `stage1_alpaca.py` and `stage2_json.py` look properly implemented with:
- ✅ 4-bit quantization
- ✅ QLoRA adapters
- ✅ FP16 precision
- ✅ Proper checkpoint saving
- ✅ Trainer configuration

**Action Items**:
```bash
# 1. Verify training can start
sbatch hpc_scripts/run_training.slurm

# 2. Monitor logs
tail -f logs/training_<JOB_ID>.log

# 3. Check checkpoints created
ls -lh checkpoints/
```

---

## Phase 4: Evaluation - **MAJOR MISSING COMPONENT**

Your assignment **requires THREE checkpoints** to be evaluated:

### Checkpoint 0: Baseline (untuned Phi-3.5)
```python
# evaluation/inference.py must support:
output_path = "results/checkpoint_0_baseline.json"
model_path = "microsoft/Phi-3.5-mini-instruct"  # untuned
```

### Checkpoint 1: After Stage 1 (Alpaca-tuned)
```python
model_path = "checkpoints/stage1_alpaca_final/adapter"
output_path = "results/checkpoint_1_alpaca.json"
```

### Checkpoint 2: After Stage 2 (JSON-tuned)
```python
model_path = "checkpoints/stage2_json_final/adapter"
output_path = "results/checkpoint_2_json_tuned.json"
```

### Evaluation Suites Required

#### A. **Alpaca Evaluation** (Self-Instruct Protocol)
```python
# evaluation/judge_evaluation.py
# For each checkpoint, evaluate against alpaca_eval.json

Judge Model: Llama-3.1-70B-Instruct
Dimensions: instruction_following, correctness, clarity, completeness, 
            structured_output_validity, hallucination_risk

Output: judge_scores_alpaca_cp<0,1,2>.json
```

#### B. **JSON Evaluation** (Structured Output)
```python
# evaluation/json_evaluation.py
# For each checkpoint against stage2_json_instruct_eval.json

Metrics:
- json_validity_rate (% of valid JSON)
- schema_compliance_rate (% matching required schema)
- exact_match_accuracy (% perfect matches)
- field_level_f1 (entity extraction accuracy)

Output: json_scores_cp<0,1,2>.json
```

### C. **Forgetting Analysis** (Core Research Question)
```python
# evaluation/forgetting_analysis.py

Compare Checkpoint 1 vs Checkpoint 2 on Alpaca eval:
- Delta in judge win rate
- Delta in ROUGE-L, BERTScore
- Per-category breakdown
- Regression examples

This is THE critical finding for your report.
```

---

## Required Results Table (Assignment 4.1)

You must produce this table with actual measured values:

| Checkpoint | Alpaca Win Rate | ROUGE-L | BERTScore | JSON Valid | Schema Compliance | Exact Match |
|------------|-----------------|---------|-----------|------------|-------------------|------------|
| 0: Baseline | ? | ? | ? | ? | ? | ? |
| 1: Alpaca | ? | ? | ? | ? | ? | ? |
| 2: JSON-Tuned | ? | ? | ? | ? | ? | ? |

**This table is the centerpiece of your entire report.**

---

## Ablation Study (Assignment 4.5)

You must include at least ONE ablation. Recommended:

```python
# Option A: Vary Stage 2 epochs (recommended, easiest)
configs = [
    {"stage2_epochs": 1},
    {"stage2_epochs": 2},  # default
    {"stage2_epochs": 3}
]

# Measure: Alpaca retention (wins%) vs JSON accuracy (%)
# Plot: Trade-off curve
```

---

## GitHub Report Structure (Assignment Section 5)

Create `REPORT.md` with:

```markdown
# LLM Instruction Tuning: Alpaca → JSON Sequential Fine-Tuning

## 1. Methodology (1 page)
- Student model: Phi-3.5 Mini (justification)
- Alpaca data source: [specify which variant]
- Teacher model setup: Llama 3.1 70B + API config
- JSON dataset construction: 5 task types + validation
- Training: QLoRA, 4-bit, batch 4, 2 epochs per stage
- Judge model: Llama 3.1 70B
- Evaluation: Alpaca (Self-Instruct) + JSON structured metrics

## 2. Experiments (3 pages)
- Three-checkpoint comparison table (KEY RESULT)
- Alpaca judge scores by dimension
- JSON validity/compliance/accuracy metrics
- Forgetting analysis: CP1 vs CP2 Alpaca degradation
- Ablation: epochs 1/2/3 → forgetting curve
- Figures: Loss curves, metric comparisons, error distributions

## 3. Analysis (1 page)
- Did Stage 2 preserve or degrade Alpaca capabilities?
- Evidence: quantitative delta + qualitative examples
- Root cause: learning rate? data diversity? epoch count?
- Implications for sequential fine-tuning in practice

## 4. Prompt Engineering (included above)
- Teacher JSON generation prompts (5 task types)
- Judge evaluation prompts (6 dimensions)
- Iterative improvements based on failures

## Appendix: Full Prompts
- All teacher generation prompts (JSON tasks)
- All judge evaluation prompts
```

---

## Immediate Action Plan

### **Week 1: Data & Setup** 
- [ ] Fix SLURM scripts (✅ DONE - see SLURM_FIX_SUMMARY.md)
- [ ] Verify Alpaca data loads correctly
- [ ] **Create 1b_generate_json_instruct.py** (teacher generation)
  - Design 5 task types
  - Call teacher API for ~100 prompts
  - Validate JSON
  - Save stage2_json_instruct_train.json & eval set
- [ ] Create training logs directory: `mkdir -p logs results`

### **Week 2: Training**
- [ ] Run Stage 1 training: `sbatch hpc_scripts/run_training.slurm`
- [ ] Verify checkpoints saved
- [ ] Monitor GPU/CPU usage from logs

### **Week 3: Evaluation**
- [ ] Implement `inference.py` (3 checkpoints)
- [ ] Implement `judge_evaluation.py` (Alpaca metrics)
- [ ] Implement `json_evaluation.py` (structured metrics)
- [ ] Run all evals, collect results → CSV/JSON

### **Week 4: Analysis & Report**
- [ ] Run ablation study (epoch variations)
- [ ] Compute forgetting analysis (CP1 vs CP2)
- [ ] Create visualizations (loss curves, metric comparisons)
- [ ] Write `REPORT.md` (5 pages + appendix)
- [ ] Push to GitHub with clear README

---

## Critical Files to Create/Review

```
MUST EXIST (for submission):
├── 1b_generate_json_instruct.py       ← **PRIORITY - MISSING**
├── judge_evaluation.py                 ← **PRIORITY - MISSING**
├── json_evaluation.py                  ← **PRIORITY - MISSING**
├── forgetting_analysis.py              ← **PRIORITY - MISSING**
├── REPORT.md                           ← **PRIORITY - MISSING**
├── results/
│   ├── checkpoint_0_baseline.json
│   ├── checkpoint_1_alpaca.json
│   ├── checkpoint_2_json_tuned.json
│   ├── judge_scores_alpaca_*.json
│   ├── json_scores_*.json
│   └── forgetting_analysis.json
└── README.md                           ← Update with new instructions
```

---

## Common Pitfalls to Avoid

❌ **Forgetting to set aside eval splits** - Use separate prompts at eval time
❌ **Not validating teacher JSON** - Invalid JSON ruins training
❌ **Only measuring Stage 2 performance** - You MUST compare CP2 back to CP1 on Alpaca
❌ **Vague forgetting analysis** - Need exact deltas and category breakdowns
❌ **Missing ablation** - Assignment specifically requires at least one ablation
❌ **Hardcoded prompts** - All prompts should be in separate template files
❌ **No reproducibility** - Log all seeds, models, API keys (redacted), hyperparameters

---

## Questions to Answer in Your Report

1. **Central Research Question**: Did Stage 2 preserve or degrade Alpaca capabilities?
   - How much? Quantify.
   - Why? Analyze.

2. **Trade-offs**: How much JSON improvement at what cost to Alpaca performance?

3. **Generalization**: Do the findings suggest sequential fine-tuning is risky or manageable?

4. **Improvements**: What hyperparameter changes would better preserve Alpaca while improving JSON?

---

## Assignment Due: April 6th, 2026

**Submission includes**:
✅ GitHub blog post (REPORT.md)
✅ Code repository (all scripts)
✅ Reproduction instructions (README.md)
✅ All artifacts (checkpoints, eval results, logs)

**Good luck!**
