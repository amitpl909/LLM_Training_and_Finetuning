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
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

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
    print("Starting Stage 1: Alpaca Fine-Tuning (Native Trainer)...")
    check_kernel_compatibility()
    
    model_id = config["student_model"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 4-bit Quantization (FP16 for cluster driver compatibility)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading model with 4-bit quantization...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto"  # Use auto for 4-bit - it handles placement correctly
    )
    print(f"Model loaded. Checking device...", flush=True)
    
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # GPU diagnostics - wrapped in try-except for old driver compatibility
    try:
        print(f"GPU available: {torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"GPU device: {torch.cuda.current_device()}", flush=True)
            print(f"GPU name: {torch.cuda.get_device_name(0)}", flush=True)
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
    except Exception as e:
        print(f"[WARN] GPU diagnostics failed (old driver?): {e}", flush=True)
    
    dataset = load_dataset("json", data_files="data_prep/alpaca_train.json")["train"]
    
    # Native tokenization (Bypassing TRL completely)
    def tokenize_function(example):
        texts = []
        # Use .get() to safely handle missing 'input' columns if they don't exist
        instructions = example.get('instruction', example.get('prompt', []))
        inputs = example.get('input', [''] * len(instructions))
        outputs = example['output']
        
        for inst, inp, out in zip(instructions, inputs, outputs):
            p = f"{inst}\n\nInput: {inp}" if inp and str(inp).strip() != "" else inst
            texts.append(f"User: {p}\nAssistant: {out}{tokenizer.eos_token}")
        return tokenizer(texts, truncation=True, max_length=config["max_sequence_length"])

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names, num_proc=1)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Check CUDA availability safely (old driver may fail)
    cuda_available = False
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as e:
        print(f"[WARN] CUDA check failed (old driver?): {e}", flush=True)
    
    training_args = TrainingArguments(
        output_dir="checkpoints/stage1_alpaca",
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=float(config["stage1_learning_rate"]),
        num_train_epochs=config["stage1_epochs"],
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

    # Core HF Trainer
    print("Creating Trainer object...", flush=True)
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=data_collator
    )
    print("Trainer created successfully. Starting training...", flush=True)

    try:
        trainer.train()
        print("Training completed successfully!", flush=True)
    except Exception as e:
        print(f"Training error: {e}", flush=True)
        raise
    
    os.makedirs("checkpoints/stage1_alpaca_final", exist_ok=True)
    trainer.model.save_pretrained("checkpoints/stage1_alpaca_final")
    tokenizer.save_pretrained("checkpoints/stage1_alpaca_final")
    print("Stage 1 Complete. Adapter saved to checkpoints/stage1_alpaca_final")

if __name__ == "__main__":
    main()