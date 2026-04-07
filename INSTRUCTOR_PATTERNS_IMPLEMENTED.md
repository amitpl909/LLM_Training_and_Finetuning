## Instructor Pattern Implementation

Based on the instructor's lecture transcript on "Fine-Tuning Small LLMs", the following updates have been made to align the codebase with best practices:

### **Key Changes Implemented**

#### 1. **Chat Template Definition & Consistency** ✅
- **Issue**: Different formatting was being used in training vs inference, and the template wasn't formalized
- **Solution**: Created `src/data_utils.py` with a clear, reusable `CHAT_TEMPLATE` that:
  - Defines system message: "Below is an instruction that describes a task..."
  - Structures instruction, input, and response sections consistently
  - Is used identically in both training (with output) and inference (without output)
- **Benefit**: Ensures model learns with the same template structure it will see during inference (following instructor's example with Instruction-Llama template)

#### 2. **Separate Training vs Inference Formatting** ✅
- **Issue**: No clear distinction between training data (with labels) and inference data (without labels)
- **Solution**: Implemented two functions in `data_utils.py`:
  - `format_instruction_for_training(instruction, input, output)` → includes expected response
  - `format_instruction_for_inference(instruction, input)` → leaves response empty for model to complete
- **Follows**: Instructor's pattern shown in the transcript where different formatting functions were used based on use case
- **Result**: Training data includes ground truth labels; evaluation data only shows instructions (matching instructor's approach)

#### 3. **Centralized Data Utilities Module** ✅
- **File**: `src/data_utils.py`
- **Contains**:
  - Formal template definition
  - Stage-specific formatting functions (Alpaca, JSON instruction)
  - Batch tokenization with template application
  - Validation and logging functions
  - Clear documentation matching instructor's recommended style
- **Benefit**: Modular, reusable pattern (instructor emphasized modularity and configuration)

#### 4. **Updated Training Scripts** ✅
- **Files Modified**:
  - `training/stage1_alpaca.py`
  - `training/stage2_json.py`
- **Changes**:
  - Import data_utils functions
  - Call `print_template_info()` at startup (logging/transparency, instructor's pattern)
  - Use `tokenize_for_training()` instead of hardcoded formatting
  - Comments explain instructor's separation of concerns
- **Result**: Both stages now use consistent template with proper training formatting

#### 5. **Updated Inference Script** ✅
- **File**: `evaluation/inference.py`
- **Changes**:
  - Import `format_instruction_for_inference()` from data_utils
  - Modified `generate_responses()` to accept (instruction, input) tuples and apply template
  - Use same template structure as training (critical for consistency)
  - Updated result storage to capture instruction and input separately
- **Benefit**: Inference now uses identical template structure to training (instructor's best practice)

---

### **How This Aligns with Instructor's Patterns**

The instructor showed these best practices in the transcript:

| Instructor Practice | Implementation |
|---|---|
| Define a clear chat template | `CHAT_TEMPLATE` constant in data_utils |
| Separate training/inference formatting | `format_instruction_for_training()` vs `format_instruction_for_inference()` |
| Use dataclasses for config (HF pattern) | Maintained config.yaml with all hyperparameters |
| Modular, reusable code | data_utils module with documented, stageable functions |
| Consistent template across stages | Both stage1 and stage2 use identical template |
| Track templates in code | `print_template_info()` function prints template on startup |

---

### **Usage Notes**

1. **Template is centralized** in `src/data_utils.py` under `CHAT_TEMPLATE`
2. **If you need to change the template**: Update `CHAT_TEMPLATE` in data_utils; all scripts automatically use the new template
3. **Training uses output** - model learns ground truth via the template with `{response}` filled
4. **Inference leaves placeholder** - template has empty `{response}` for model to complete
5. **Documentation**: Each function in data_utils has examples and explains instructor's reasoning

---

### **Next Steps**

Ready to execute training following instructor's approach:
1. **Data preparation**: `python data_prep/1a_prep_alpaca.py` (creates Alpaca train/eval split)
2. **Data generation**: `python data_prep/1b_generate_json_instruct.py` (creates teacher-generated JSON data)
3. **Training Stage 1**: `python training/stage1_alpaca.py` (fine-tune on Alpaca with template-based formatting)
4. **Training Stage 2**: `python training/stage2_json.py` (fine-tune on JSON data, loads Stage 1 adapter)
5. **Inference**: `python evaluation/inference.py` (generate responses using consistent template)

All scripts now follow the instructor's recommended pattern for instruction tuning!
