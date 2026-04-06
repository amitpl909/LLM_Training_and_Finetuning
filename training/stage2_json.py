import yaml
import torch
import os
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
    print("Starting Stage 2: JSON Instruct Fine-Tuning (Native Trainer)...")
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

    dataset = load_dataset("json", data_files="data_prep/stage2_json_instruct_train.json")["train"]
    
    # Native tokenization
    def tokenize_function(example):
        texts = []
        # Use .get() to safely handle missing 'input' columns to prevent KeyErrors
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

    training_args = TrainingArguments(
        output_dir="checkpoints/stage2_json",
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=4,
        learning_rate=float(config["stage2_learning_rate"]),
        num_train_epochs=config["stage2_epochs"],
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        optim="paged_adamw_32bit",
        bf16=True,  # Use bfloat16 instead of fp16 with 4-bit quantization
        tf32=True,  # Enable TF32 for better performance
        report_to="none",
        logging_first_step=True,  # Log first step to verify training starts
        remove_unused_columns=False,  # Important for LoRA
        dataloader_pin_memory=False,  # Disable pin_memory for 4-bit quantized models
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