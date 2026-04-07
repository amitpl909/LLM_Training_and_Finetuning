# CRITICAL: Set environment variables BEFORE any heavy imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

import yaml
import torch
import platform
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import PeftModel
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))  # Add src directory to path
from data_utils import tokenize_for_training, print_template_info

# Load central configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Check kernel version for compatibility
def check_kernel_compatibility():
    kernel_version = platform.release()
    try:
        parts = kernel_version.split('.')
        major, minor = int(parts[0]), int(parts[1])
        if major < 5 or (major == 5 and minor < 5):
            print(f"⚠️  WARNING: Kernel version {kernel_version} is below 5.5.0")
            print("   Old kernels may cause hangs during tokenization.")
            print("   Using num_proc=1 for safe single-process tokenization.")
            return False
        return True
    except:
        return True  # Skip check if parsing fails

def main():
    print("Starting Stage 2: JSON Instruct Fine-Tuning (Following Instructor's Pattern)...")
    print_template_info()
    check_kernel_compatibility()
    
    model_id = config["student_model"]
    stage1_adapter_path = "checkpoints/stage1_alpaca_final"
    
    if not os.path.exists(stage1_adapter_path):
        raise FileNotFoundError(f"Stage 1 adapter not found at {stage1_adapter_path}. Check Stage 1 logs.")

    tokenizer = AutoTokenizer.from_pretrained(stage1_adapter_path)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto"  # Use auto for 4-bit - it handles placement correctly
    )
    
    model = PeftModel.from_pretrained(base_model, stage1_adapter_path, is_trainable=True)
    
    # GPU diagnostics - wrapped in try-except for old driver compatibility
    try:
        print(f"GPU available: {torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"GPU device: {torch.cuda.current_device()}", flush=True)
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
    except Exception as e:
        print(f"[WARN] GPU diagnostics failed (old driver?): {e}", flush=True)

    dataset = load_dataset("json", data_files="data_prep/stage2_json_instruct_train.json")["train"]
    
    # Tokenization following instructor's pattern (with output for training)
    def tokenize_function(example):
        """Instructor's pattern: separate functions for training vs inference."""
        return tokenize_for_training(example, tokenizer, config["max_sequence_length"], for_inference=False)

    print("Tokenizing dataset (with instructor's template)...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names, num_proc=1)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Check CUDA availability safely (old driver may fail)
    cuda_available = False
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as e:
        print(f"[WARN] CUDA check failed (old driver?): {e}", flush=True)
    
    training_args = TrainingArguments(
        output_dir="checkpoints/stage2_json",
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=float(config["stage2_learning_rate"]),
        num_train_epochs=config["stage2_epochs"],
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        optim="paged_adamw_32bit",
        bf16=False,  # CRITICAL: Disable for old CUDA driver (v12030 on UTSA HPC)
        fp16=False,  # Disable fp16 - not compatible with 4-bit quantization
        tf32=False,  # Disable TF32 for safety with old driver
        report_to="none",
        logging_first_step=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=data_collator
    )
    
    print("Stage 2 Trainer created. Starting training...", flush=True)
    try:
        trainer.train()
        print("Stage 2 training completed successfully!", flush=True)
    except Exception as e:
        print(f"Stage 2 training error: {e}", flush=True)
        raise
    
    os.makedirs("checkpoints/stage2_json_final", exist_ok=True)
    trainer.model.save_pretrained("checkpoints/stage2_json_final")
    print("Stage 2 Complete. Final model adapter saved to checkpoints/stage2_json_final")

if __name__ == "__main__":
    main()