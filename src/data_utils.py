"""
Data Utilities Module - Instructor's Best Practices Pattern

This module implements the instructor's recommended approach:
1. Define a clear, reusable CHAT TEMPLATE
2. Separate functions for TRAINING (with output) vs INFERENCE (without output)
3. Consistent formatting across all stages

Critical: The SAME TEMPLATE must be used during training and inference!
"""

import json


# ============================================================================
# CHAT TEMPLATE DEFINITION
# ============================================================================
# This template is used for both Alpaca fine-tuning and JSON instruction tuning.
# Format: System message → Instruction (+ Input if available) → Assistant response

CHAT_TEMPLATE = """Below is an instruction that describes a task. If there is an input, it provides further context.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}"""


def format_instruction_for_training(instruction: str, input_text: str, output: str) -> str:
    """
    Format instruction for TRAINING (includes the expected output).
    
    During training, we want the model to learn:
    - Given [instruction + input], predict [output]
    
    Args:
        instruction: The task instruction
        input_text: Additional context/input (can be empty string)
        output: The expected response (used for training labels)
    
    Returns:
        Formatted text ready for tokenization
    
    Examples:
        >>> format_instruction_for_training(
        ...     "What is capital of France?",
        ...     "",  # No input for this task
        ...     "Paris"
        ... )
        "Below is an instruction...### Response:\nParis"
    """
    return CHAT_TEMPLATE.format(
        instruction=instruction,
        input=input_text.strip() if input_text else "",
        response=output
    )


def format_instruction_for_inference(instruction: str, input_text: str) -> str:
    """
    Format instruction for INFERENCE/EVALUATION (NO expected output).
    
    During inference, we want to:
    - Give the model [instruction + input]
    - Ask it to predict [output]
    
    The response placeholder is left for the model to complete.
    
    Args:
        instruction: The task instruction
        input_text: Additional context/input (can be empty string)
    
    Returns:
        Formatted text ready for model.generate()
    
    Examples:
        >>> format_instruction_for_inference(
        ...     "What is capital of France?",
        ...     ""
        ... )
        "Below is an instruction...### Response:\n"  # Model completes this
    """
    prompt = CHAT_TEMPLATE.format(
        instruction=instruction,
        input=input_text.strip() if input_text else "",
        response=""
    )
    # Note: We include up to the last line with "Response:" for the model to complete
    return prompt.rstrip() + "\n"


# ============================================================================
# STAGE-SPECIFIC FORMATTING FUNCTIONS
# ============================================================================

def format_alpaca_example_for_training(example: dict) -> str:
    """Format a single Alpaca example for training."""
    return format_instruction_for_training(
        instruction=example.get("instruction", ""),
        input_text=example.get("input", ""),
        output=example.get("output", "")
    )


def format_json_example_for_training(example: dict) -> str:
    """Format a single JSON instruction example for training."""
    return format_instruction_for_training(
        instruction=example.get("instruction", ""),
        input_text=example.get("input", ""),
        output=example.get("output", "")
    )


# ============================================================================
# BATCH FORMATTING FOR TOKENIZATION
# ============================================================================

def tokenize_for_training(examples: dict, tokenizer, max_length: int, for_inference: bool = False) -> dict:
    """
    Tokenize a batch of examples following the instructor's pattern.
    
    Args:
        examples: Dict with 'instruction', 'input', 'output' keys (batch format)
        tokenizer: Huggingface tokenizer
        max_length: Maximum sequence length
        for_inference: If True, format without output (for evaluation sets)
    
    Returns:
        Tokenized inputs (input_ids, attention_mask)
    
    Pattern from instructor:
    - Training: Include output → model learns to match ground truth
    - Inference: Exclude output → model generates fresh response
    """
    texts = []
    
    instructions = examples.get('instruction', [])
    inputs = examples.get('input', [''] * len(instructions))
    outputs = examples.get('output', [''] * len(instructions))
    
    for inst, inp, out in zip(instructions, inputs, outputs):
        if for_inference:
            # Evaluation: No output shown to model
            text = format_instruction_for_inference(inst, inp)
        else:
            # Training: Include expected output
            text = format_instruction_for_training(inst, inp, out)
        texts.append(text)
    
    # Tokenize with truncation
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    
    return encodings


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_instruction_format(formatted_text: str) -> bool:
    """
    Validate that formatted instruction follows the template structure.
    
    Returns:
        True if format is correct, False otherwise
    """
    required_sections = ["Instruction:", "Input:", "Response:"]
    return all(section in formatted_text for section in required_sections)


def print_template_info():
    """Print information about the active template (for logging/debugging)."""
    print("\n" + "="*70)
    print("CHAT TEMPLATE IN USE (Instructor's Pattern)")
    print("="*70)
    print(CHAT_TEMPLATE)
    print("="*70)
    print("⚠️  CRITICAL: This template must be used consistently in:")
    print("   1. Training tokenization (with output)")
    print("   2. Evaluation inference (without output)")
    print("="*70 + "\n")
