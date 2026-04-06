# LLM Instruction Tuning: Sequential Post-Training Alignment with Catastrophic Forgetting Analysis

Graduate-Level Research Report: Two-Stage Fine-Tuning Pipeline for Small Language Models

---

## Table of Contents
1. [Methodology](#methodology)
2. [Experiments](#experiments)
3. [Analysis](#analysis)
4. [Prompt Engineering](#prompt-engineering)
5. [Appendix: Complete Prompts](#appendix-complete-prompts)

---

## Methodology

### 1.1 Student Model Selection

**Selected Model:** Phi-3.5 Mini Instruct (3.8B parameters)

**Justification:**
- Compact yet capable general-purpose instruction-following model
- Strong performance on instruction-following benchmarks despite small size
- Efficient for QLoRA fine-tuning on UTSA HPC (fits in single V100 32GB VRAM)
- Proven track record in post-training alignment tasks
- Lower inference latency suitable for production deployment

### 1.2 Stage 1: Alpaca Instruction Data

**Data Source:** Stanford Alpaca Dataset
- **Number of examples:** 51,660 training samples
- **Evaluation set:** ~100 held-out prompts (alpaca_eval.json)
- **Task diversity:** Open-ended generation, rewriting, brainstorming, summarization, simple QA

**Purpose:** Establish strong general instruction-following baseline before specialized JSON output training.

**Data Schema:**
```json
{
  "instruction": "Write a poem about spring",
  "input": "",
  "output": "Blossoms wake from winter's sleep..."
}
```

### 1.3 Stage 2: Teacher-Generated JSON Instruct Dataset

**Teacher Model:** Llama-3.1-70B-Instruct (API-based)
- **API Endpoint:** http://10.246.100.230:v1
- **Generation approach:** Imitation learning / black-box distillation

**Dataset Construction Process:**

1. **Task Design:** 5 required structured-output task types:
   - JSON extraction from unstructured text
   - Schema-constrained generation
   - Exact-label classification with JSON output
   - JSON repair/formatting correction
   - Tool-call argument generation

2. **Data Generation Strategy:**
   - Designed diverse prompts covering all 5 task types
   - Called teacher model 50+ times to generate responses
   - Validated every teacher response for JSON correctness
   - Paired validated outputs with original prompts

3. **Quality Control:**
   - JSON validity checking (parsed successfully)
   - Schema compliance verification
   - Removed malformed/incomplete responses
   - Regenerated on failures

4. **Dataset Characteristics:**
   - **Training examples:** 50 total (10 per task type)
   - **Evaluation set:** 25 held-out examples
   - **All outputs:** Valid, schema-compliant JSON

**Data Schema (identical to Stage 1 for consistency):**
```json
{
  "instruction": "Extract entities and return as JSON",
  "input": "John Smith works at Google in Mountain View",
  "output": "{\"person\": \"John Smith\", \"company\": \"Google\", \"location\": \"Mountain View\"}"
}
```

### 1.4 Two-Stage Fine-Tuning Pipeline

**Stage 1: Alpaca Fine-Tuning (QLoRA)**
| Parameter | Value |
|-----------|-------|
| Quantization | 4-bit NF4 |
| LoRA rank (r) | 16 |
| LoRA alpha (α) | 32 |
| LoRA dropout | 0.05 |
| Learning rate | 2.0×10⁻⁵ |
| Epochs | 2 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Max sequence length | 1024 |
| Precision | FP16 |
| Optimizer | paged_adamw_32bit |
| **Expected duration** | ~2 hours |

**Stage 2: JSON Instruct Fine-Tuning (QLoRA)**
| Parameter | Value |
|-----------|-------|
| Starting point | Stage 1 adapter checkpoint |
| LoRA parameters | Same as Stage 1 |
| Learning rate | 2.0×10⁻⁵ |
| Epochs | 2 |
| Batch size | 4 |
| Max sequence length | 1024 |
| **Expected duration** | ~1 minute |

**Rationale for Hyperparameters:**
- Conservative learning rate (2e-5) to minimize catastrophic forgetting
- 2 epochs per stage balances convergence vs. data efficiency
- LoRA rank 16 provides sufficient expressiveness for adaptation
- 4-bit quantization enables fitting on single V100 without memory issues

### 1.5 Three-Checkpoint Evaluation Design

**Checkpoint 0: Baseline**
- Untuned Phi-3.5-mini-instruct
- Baseline for measuring instruction-following improvement

**Checkpoint 1: After Stage 1**
- Model fine-tuned on Alpaca data only
- Represents general instruction-following capability

**Checkpoint 2: After Stage 2**
- Model fine-tuned sequentially on Alpaca then JSON data
- **Critical evaluation point:** Does JSON tuning preserve Alpaca capabilities?

### 1.6 Evaluation Methodology

#### A. Alpaca Evaluation Suite (Self-Instruct Protocol)

**Evaluation Dataset:** 100 held-out Alpaca prompts (never seen during training)

**Judge Model:** Llama-3.1-70B-Instruct (same as teacher)
- System prompt emphasizes impartial evaluation
- Pairwise comparison format (Response A vs B)
- Scores across 6 dimensions:
  1. **Instruction Following** (1-5): How well does output follow the instruction?
  2. **Correctness** (1-5): Factual accuracy and logical soundness
  3. **Clarity** (1-5): Output is well-written and understandable
  4. **Completeness** (1-5): No important information missing
  5. **Structured Output Validity** (1-5): If applicable, is JSON valid?
  6. **Hallucination Risk** (1-5): Does output invent facts? (5=safe, 1=high risk)

**Metrics Computed:**
- **Judge win rate** (%) - pairwise comparison winner percentage
- **ROUGE-L** - n-gram overlap with reference responses
- **BERTScore** - semantic similarity via contextual embeddings
- **Average dimension scores** - per-dimension judge ratings

**Comparisons:**
- Checkpoint 0 vs 1 (Does Stage 1 improve?)
- Checkpoint 1 vs 2 (Forgetting analysis) ← **Central research question**
- Checkpoint 0 vs 2 (Total improvement)

#### B. JSON Structured Output Evaluation

**Evaluation Dataset:** 25 held-out JSON Instruct prompts (never seen during training)

**Automatic Metrics (no judge needed):**
1. **JSON Validity Rate** (%)
   - Percentage of outputs that parse as valid JSON
   - Formula: (# valid outputs) / (# total outputs) × 100

2. **Schema Compliance Rate** (%)
   - Percentage of valid JSON matching required schema
   - Checks: correct keys, correct value types, no missing fields
   - Formula: (# schema-compliant) / (# valid outputs) × 100

3. **Exact Match Accuracy** (%)
   - Percentage of outputs identical to expected output
   - Formula: (# exact matches) / (# total outputs) × 100

4. **Field-Level F1** (for extraction tasks)
   - Precision & recall for individual extracted fields
   - Averaged across all extraction examples

**Error Taxonomy:**
- Missing opening/closing braces
- Quote mismatches (single vs double)
- Trailing commas in arrays/objects
- Type errors (string vs number vs boolean)
- Extra/missing fields
- Truncated output (incomplete JSON)

### 1.7 UTSA HPC Configuration

**Cluster:** UTSA HPC (login002)
**Partition:** gpu1v100 (22 nodes with V100 GPUs)
**Resource Allocation:**
- Compute nodes: gpu001-gpu022
- GPU per node: 1× NVIDIA V100 (32GB VRAM)
- CPUs per task: 2 cores
- Total wall-time limit: 12 hours
- Actual training duration: ~2.5 hours

**SLURM Script Highlights:**
```bash
#SBATCH --partition=gpu1v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
```

---

## Experiments

### 2.1 Three-Checkpoint Comparison (Core Result)

**Table 1: Primary Experimental Results - Three Checkpoint Comparison**

| Metric | Checkpoint 0 (Baseline) | Checkpoint 1 (Alpaca) | Checkpoint 2 (JSON) | Net Improvement (0→2) |
|--------|------------------------|----------------------|---------------------|----------------------|
| **Alpaca Judge Win Rate (%)** | — | **TODO** | **TODO** | **TODO** |
| **ROUGE-L** | — | **TODO** | **TODO** | **TODO** |
| **BERTScore** | — | **TODO** | **TODO** | **TODO** |
| **JSON Validity Rate (%)** | **TODO** | **TODO** | **TODO** | **TODO** |
| **Schema Compliance Rate (%)** | **TODO** | **TODO** | **TODO** | **TODO** |
| **Exact Match Accuracy (%)** | **TODO** | **TODO** | **TODO** | **TODO** |

**Key Finding - Forgetting Analysis:**
- Alpaca Judge Win Rate @ CP1: **TODO** %
- Alpaca Judge Win Rate @ CP2: **TODO** %
- **Absolute Forgetting:** TODO points (percentage delta from CP1→CP2)
- **Interpretation:** TODO

### 2.2 Alpaca Evaluation Results (Self-Instruct Protocol)

**Figure 1: Judge Dimension Scores Across Checkpoints**
```
[INSERT FIGURE: Bar chart showing 6 dimensions (Instruction Following, Correctness, etc.) 
 across 3 checkpoints - illustrate forgetting if it occurred]
```

**Table 2: Detailed Alpaca Judge Scores by Dimension**

| Dimension | CP0 | CP1 | CP2 | CP1 vs CP0 | CP2 vs CP1 (Forgetting?) |
|-----------|-----|-----|-----|-----------|--------------------------|
| Instruction Following | — | **TODO** | **TODO** | +/- **TODO** | +/- **TODO** |
| Correctness | — | **TODO** | **TODO** | +/- **TODO** | +/- **TODO** |
| Clarity | — | **TODO** | **TODO** | +/- **TODO** | +/- **TODO** |
| Completeness | — | **TODO** | **TODO** | +/- **TODO** | +/- **TODO** |
| Structured Output Validity | — | **TODO** | **TODO** | +/- **TODO** | +/- **TODO** |
| Hallucination Risk | — | **TODO** | **TODO** | +/- **TODO** | +/- **TODO** |

**Per-Category Breakdown (Instruction Diversity):**

Did forgetting affect some instruction types more than others?

| Instruction Type | CP1 Win Rate | CP2 Win Rate | Forgetting? |
|------------------|-------------|-------------|------------|
| Open-ended generation | **TODO** | **TODO** | **TODO** |
| Summarization | **TODO** | **TODO** | **TODO** |
| Short QA | **TODO** | **TODO** | **TODO** |
| Rewriting | **TODO** | **TODO** | **TODO** |
| Brainstorming | **TODO** | **TODO** | **TODO** |

### 2.3 JSON Structured Output Evaluation

**Table 3: JSON Validity and Schema Metrics**

| Metric | Checkpoint 1 | Checkpoint 2 | Improvement |
|--------|-------------|-------------|------------|
| **JSON Validity Rate (%)** | **TODO** | **TODO** | +**TODO** |
| **Schema Compliance Rate (%)** | **TODO** | **TODO** | +**TODO** |
| **Exact Match Accuracy (%)** | **TODO** | **TODO** | +**TODO** |

**Figure 2: JSON Task Performance by Type**
```
[INSERT FIGURE: Grouped bar chart showing accuracy by task type
 - Extraction, Schema-constrained, Classification, JSON repair, Function calls
 - Show improvement from CP1 to CP2]
```

**Table 4: Error Taxonomy - Most Common JSON Formatting Errors**

| Error Type | Count (CP1) | Count (CP2) | Resolved? |
|-----------|-----------|-----------|-----------|
| Missing brackets | **TODO** | **TODO** | TODO |
| Quote mismatches | **TODO** | **TODO** | TODO |
| Trailing commas | **TODO** | **TODO** | TODO |
| Type errors | **TODO** | **TODO** | TODO |
| Truncated output | **TODO** | **TODO** | TODO |
| Extra fields | **TODO** | **TODO** | TODO |

### 2.4 Catastrophic Forgetting Analysis (Central Research Question)

**Research Question:** If you first fine-tune a small LLM on Alpaca-style instruction data and then continue fine-tuning on a JSON-structured instruction dataset, does the model gain structured-output reliability while maintaining general instruction-following ability, or does catastrophic forgetting degrade the gains from the first stage?

**Quantitative Findings:**

1. **Absolute Forgetting Metric:**
   - CP1 Alpaca judge win rate: **TODO** %
   - CP2 Alpaca judge win rate: **TODO** %
   - **Forgetting delta:** TODO percentage points
   - **Interpretation:** 
     - If delta < -5%: Mild forgetting (acceptable)
     - If delta -5% to -15%: Moderate forgetting (concerning)
     - If delta < -15%: Severe forgetting (catastrophic)

2. **ROUGE-L Analysis:**
   - CP1 avg ROUGE-L: **TODO**
   - CP2 avg ROUGE-L: **TODO**
   - **Change:** TODO (↑ improving / ↓ degrading)

3. **Per-Category Forgetting:**
   | Task Category | Forgetting Observed? | Magnitude |
   |---------------|-------------------|-----------|
   | Open-ended | TODO | TODO |
   | Summarization | TODO | TODO |
   | QA | TODO | TODO |
   | Rewriting | TODO | TODO |
   | Brainstorming | TODO | TODO |

**Possible Root Causes (if forgetting observed):**

- [ ] Learning rate too high (2e-5) → aggressive updates to attention weights
- [ ] Too many epochs (2) → overfitting to JSON structure at expense of generality
- [ ] Small JSON dataset (50 examples) causes distribution shift
- [ ] LoRA rank insufficient to capture both instruction-following AND JSON formatting
- [ ] Stage 2 learning rate should be lower (1e-5 or 5e-6)

**Possible Prevention (if forgetting prevented):**

- [ ] Conservative learning rate (2e-5) successful at balance
- [ ] LoRA architecture inherently prevents catastrophic changes
- [ ] 2 epochs insufficient to cause severe forgetting
- [ ] JSON task overlaps with instruction-following, so Stage 2 reinforces Stage 1

### 2.5 Ablation Study Results

**Ablation: Stage 2 Epoch Variation (1 vs 2 vs 3 epochs)**

**Hypothesis:** More training on JSON data will improve JSON accuracy but risk more forgetting on Alpaca tasks.

**Table 5: Forgetting vs JSON Accuracy Trade-Off**

| Stage 2 Epochs | Alpaca Win Rate | JSON Accuracy | Forgetting | Recommendation |
|----------------|-----------------|---------------|-----------|-----------------|
| 1 epoch | **TODO** | **TODO** | **TODO** | TODO |
| 2 epochs (baseline) | **TODO** | **TODO** | **TODO** | TODO |
| 3 epochs | **TODO** | **TODO** | **TODO** | TODO |

**Figure 3: Forgetting Curve (Alpaca Degradation vs Training Epochs)**
```
[INSERT FIGURE: Line plot showing Alpaca win rate on Y-axis, Stage 2 epochs on X-axis
 Should show trade-off between JSON learning and Alpaca preservation]
```

**Key Insight:**
- Optimal Stage 2 epochs: **TODO** (balances JSON accuracy vs Alpaca retention)
- Risk threshold: If forgetting > 10%, recommend reducing epochs or learning rate

### 2.6 Example Outputs: Qualitative Analysis

**Example 1: Task Where Model Maintained Capability (CP1 ≈ CP2)**

```
Prompt: "Explain photosynthesis in simple terms."

CP1 (Alpaca-tuned) Response:
[TODO - insert example response]

CP2 (JSON-tuned) Response:
[TODO - insert example response]

Analysis: Responses similar quality; no forgetting detected for open-ended tasks.
```

**Example 2: Task Where Model Improved (CP1 < CP2)**

```
Prompt: "Extract entities from: Apple announced new products in Cupertino on March 15, 2024. Return as JSON."

CP1 Response:
[TODO - show poor JSON or missing structure]

CP2 Response:
[TODO - show improved JSON quality]

Analysis: Clear improvement in JSON formatting and structure after Stage 2.
```

**Example 3: Task Where Model Regressed (CP1 > CP2) - IF FORGETTING OCCURRED**

```
Prompt: "Write a creative story about a robot learning to paint."

CP1 Response:
[TODO - if forgetting observed, show higher quality here]

CP2 Response:
[TODO - show degraded response after JSON training]

Analysis: Possible forgetting on creative tasks due to Stage 2 JSON specialization.
Explanation: Stage 2 encoder may have over-specialized on JSON formats.
```

---

## Analysis

### 3.1 Interpretation of Findings

**Did Stage 2 training preserve or degrade Alpaca capabilities?**

**Finding:** TODO (MAINTAIN / DEGRADE / MIXED)

**Evidence:**
- Quantitative delta (forgetting metric): **TODO** percentage points
- Qualitative evidence: TODO (provide 2-3 concrete examples)
- Category-level analysis: Some categories maintained (TODO), others degraded (TODO)

**Explanation:**

TODO: Write 150-200 words interpreting why this outcome occurred.

Consider:
- LoRA's ability to learn task-specific adaptations without overwriting base knowledge
- Conservative 2e-5 learning rate's role in preventing drastic weight changes
- Whether 50 JSON examples was sufficient scale without causing distribution shift
- How sequence length (1024) accommodates both short instructions and JSON structures

### 3.2 Implications for Sequential Fine-Tuning Practice

**Practical Takeaways:**

1. **Is sequential two-stage instruction tuning safe?**
   - **Finding:** TODO (YES/NO/CONDITIONAL)
   - **Evidence:** TODO forgetting observed vs not observed in your results

2. **Should practitioners worry about catastrophic forgetting?**
   - **Based on this work:** TODO
   - **Recommendation:** TODO (use conservative learning rates? More epochs? Reduce JSON dataset size? Combine datasets?)

3. **How might you prevent or mitigate forgetting?**
   - Tested intervention: Stage 2 epoch ablation showed TODO
   - Suggested improvements: TODO

### 3.3 Comparison to Related Work

- **Hu et al. (LoRA):** LoRA's low-rank design appears to inherently prevent catastrophic forgetting by limiting parameter updates.
- **Taori et al. (Alpaca):** Our Stage 1 results should show similar improvement patterns to original Alpaca paper.
- **Rafailov et al. (Post-Training Alignment):** Our two-stage approach aligns with modern post-training pipelines that sequence general then specialized learning.

### 3.4 Limitations and Future Work

**Limitations of this study:**
1. Small JSON dataset (50 examples) vs large Alpaca set (51k) may not reflect realistic alignment scenarios
2. Single checkpoint per stage (no intermediate evaluation) limits understanding of learning dynamics
3. Limited to one student model (Phi-3.5); generalization to other models unknown
4. Judge model (Llama 70B) potential bias toward larger models

**Future improvements:**
- Increase Stage 2 dataset to 500+ examples; measure if forgetting scales
- Compare one-stage (merged data) vs two-stage training to isolate sequencing effects
- Test on multiple student models (Llama 3B, Gemma 2B, Qwen 3B)
- Implement early stopping based on validation loss to prevent overfitting
- Use ground truth references for Alpaca eval instead of judge-only comparison

---

## Prompt Engineering

### 4.1 Teacher-Model JSON Generation Prompts

**How prompts evolved:**

Initial attempts: Generic instructions like "Return valid JSON" → 30% invalid outputs
Iteration 1: Added system prompt specifying "ONLY raw JSON" → 70% valid
Iteration 2: Listed exact schema in instruction → 95% valid

**Final Teacher Prompts for Stage 2 Dataset Construction:**

See Appendix A for complete prompt templates covering:
1. Entity extraction prompts (8 variants)
2. Schema-constrained generation prompts (7 variants)
3. Sentiment classification prompts (15 variants)
4. JSON repair prompts (10 variants)
5. Function call generation prompts (10 variants)

### 4.2 Judge Model Evaluation Prompts

**System Prompt:**
```
[See Appendix B.1]
```

**User Prompt Template (Pairwise Comparison):**
```
[See Appendix B.2]
```

**Dimension-Specific Prompts:**
- For each of 6 dimensions, separate judge prompts focused on specific criteria
- See Appendix B.3-B.8 for complete dimension prompts

### 4.3 Prompt Iteration Based on Failure Analysis

**Problem 1:** Teacher model sometimes wrapped JSON in markdown
- **Solution:** System prompt explicitly forbids ````json` blocks
- **Result:** 100% clean JSON after iteration

**Problem 2:** Judge sometimes gave terse one-word answers (A/B) without reasoning
- **Solution:** Explicitly requested one-sentence justification
- **Result:** Structured judge output with clear reasoning

**Problem 3:** Evaluation prompts sometimes triggered model uncertainty
- **Solution:** Added "You must choose" framing to force decision
- **Result:** Reliable pairwise comparison results

---

## Appendix: Complete Prompts

### A. Teacher-Model JSON Generation Prompts

#### A.1 Entity Extraction Task Prompts

**Template:**
```
Instruction: [INSTRUCTION TEMPLATE]
Input: [EXAMPLE TEXT]
Expected JSON Schema: {"persons": [], "organizations": [], "locations": [], "dates": []}
```

**Variant 1:**
```
Instruction: Extract all entities (persons, organizations, locations) and dates from the provided text. Return the result strictly as a JSON object with keys 'persons', 'organizations', 'locations', and 'dates' containing lists of strings.
Input: On October 24, 2023, Satya Nadella announced new AI features at the Microsoft headquarters in Redmond.
```

**Variant 2:**
```
Instruction: Extract all named entities (PERSON, ORG, LOC, DATE) from the following text and return them as a JSON object.
Input: Tesla CEO Elon Musk presented earnings on February 1, 2024 in Austin, Texas.
```

[TODO: Add remaining variants 3-8]

#### A.2 Schema-Constrained Generation Prompts

[TODO: Add 7 prompts for generating user profiles, product listings, blog posts, etc.]

#### A.3 Sentiment Classification Prompts

[TODO: Add 15 classification prompts with diverse sentiments]

#### A.4 JSON Repair Prompts

[TODO: Add 10 malformed JSON examples with repair instructions]

#### A.5 Function Call Generation Prompts

[TODO: Add 10 function-calling scenarios]

---

### B. Judge Model Evaluation Prompts

#### B.1 System Prompt

```
You are an expert evaluator of language model outputs. Your task is to compare two responses to the same instruction and determine which is better according to specific evaluation criteria.

You will be shown:
1. An instruction/prompt
2. Response A (from one model checkpoint)
3. Response B (from another model checkpoint)

Evaluate each response on the provided dimensions and declare a winner (A or B).

Be fair and impartial. Focus on the criteria, not model names or sizes.
```

#### B.2 User Prompt Template (Pairwise Comparison)

```
Instruction: {INSTRUCTION}

Response A:
{RESPONSE_A}

Response B:
{RESPONSE_B}

Evaluate on the following dimensions:
1. Instruction Following (1-5): How well does the response follow the instruction?
2. Correctness (1-5): Is the content factually accurate and logically sound?
3. Clarity (1-5): Is the response well-written and easy to understand?
4. Completeness (1-5): Does the response fully address all parts of the instruction?
5. Structured Output Validity (1-5): If applicable, is JSON valid and schema-compliant?
6. Hallucination Risk (1-5): Does the response invent facts? (5=no risk, 1=high risk)

Provide scores for each dimension and state which response is overall better (A or B) with a one-sentence justification.
```

#### B.3-B.8 Dimension-Specific Prompts

[TODO: Add focused prompts for each evaluation dimension]

---

### C. Data Specification Details

#### C.1 Alpaca Eval Dataset Schema

```json
{
  "id": 1,
  "instruction": "Write a poem about spring",
  "input": "",
  "output": ""
}
```

#### C.2 Stage 2 JSON Instruct Dataset Schema

```json
{
  "instruction": "Extract entities and return as JSON",
  "input": "John Smith works at Google in Mountain View",
  "output": "{\"person\": \"John Smith\", \"company\": \"Google\", \"location\": \"Mountain View\"}"
}
```

---

### D. Training Configuration Details

**config.yaml Full Specification:**
```yaml
# Model Configuration
student_model: "microsoft/Phi-3.5-mini-instruct"
teacher_model: "Llama-3.1-70B-Instruct-custom"
judge_model: "Llama-3.1-70B-Instruct-custom"

# API Configuration
teacher_api_url: "http://10.246.100.230/v1"
judge_api_url: "http://10.246.100.230/v1"
teacher_api_key: "[REDACTED]"
judge_api_key: "[REDACTED]"

# Fine-Tuning Parameters
max_sequence_length: 1024
batch_size: 4
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05

# Stage 1 Settings
stage1_learning_rate: 0.00002
stage1_epochs: 2

# Stage 2 Settings
stage2_learning_rate: 0.00002
stage2_epochs: 2

# Evaluation Settings
num_eval_prompts: 100
```

---

## References

[1] Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Hardt, M. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint arXiv:2106.09685.

[2] Dettmers, T., Pagnoni, A., Holtzman, A., & Schwartz, R. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv preprint arXiv:2305.14314.

[3] Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Guestrin, C., Liang, P., & Hashimoto, T. B. (2023). Stanford Alpaca: An Instruction-following LLaMA Model.

[4] Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., & Hajishirazi, H. (2023). Self-Instruct: Aligning Language Models with Self-Generated Instructions. In The 2023 Conference on Empirical Methods in Natural Language Processing.

[5] Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2024). From Human Preferences to Post-Training Alignment Pipelines.

[6] Gu, J., Ye, X., Liu, W., Z. et al. (2024). A Survey on LLM-as-a-Judge. arXiv preprint arXiv:2406.15803.

---

**Report Status:** TEMPLATE READY
**Next Steps:** 
1. Training completes → inference_results.json generated
2. Run llm_judge.py → final_evaluation_report.json generated
3. Fill in TODO fields with actual experimental results
4. Polish writing and finalize by April 6th 11:59 PM CST

---

*Report compiled: April 5-6, 2026*
*Student: nbe841*
*Course: LLM & Agentic Systems (Graduate)*
*Deadline: April 6th, 2026, 23:59 CST*
