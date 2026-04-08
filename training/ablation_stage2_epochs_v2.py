"""
Ablation Study: Stage 2 Training with Varying Epochs
Follows the proven pattern from stage2_json.py

Usage:
    python training/ablation_stage2_epochs.py --epochs 1
    python training/ablation_stage2_epochs.py --epochs 2
    python training/ablation_stage2_epochs.py --epochs 3
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

import yaml
import torch
import platform
import json
import argparse
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_utils import tokenize_for_training, print_template_info

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def main(epochs=2):
    """
    Train Stage 2 with configurable epochs, following proven pipeline.
    
    Args:
        epochs: Number of training epochs (1, 2, or 3 for ablation)
    """
    
    print(f"\n{'='*60}")
    print(f"ABLATION STUDY: Stage 2 Training (epochs={epochs})")
    print(f"{'='*60}\n")
    
    print_template_info()
    
    # Check kernel
    kernel_version = platform.release()
    print(f"Kernel version: {kernel_version}\n")
    
    model_id = config["student_model"]
    stage1_adapter_path = "checkpoints/stage1_alpaca_final"
    
    if not os.path.exists(stage1_adapter_path):
        raise FileNotFoundError(f"Stage 1 adapter not found at {stage1_adapter_path}")
    
    print(f"Loading Stage 1 checkpoint from: {stage1_adapter_path}")
    
    # Load tokenizer from Stage 1 checkpoint (proven approach)
    tokenizer = AutoTokenizer.from_pretrained(stage1_adapter_path)
    
    # Configure quantization (same as working pipeline)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Load Stage 1 adapter
    print("Loading Stage 1 adapter...")
    model = PeftModel.from_pretrained(base_model, stage1_adapter_path, is_trainable=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files="data_prep/stage2_json_instruct_train.json")["train"]
    
    # Tokenize using proven function
    def tokenize_function(example):
        """Use same tokenization as proven training."""
        return tokenize_for_training(example, tokenizer, config["max_sequence_length"], for_inference=False)
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=1  # Single process on old kernel
    )
    
    # Use standard data collator (proven approach)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Setup training arguments
    output_dir = f"checkpoints/ablation_epochs{epochs}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=float(config["stage2_learning_rate"]),
        num_train_epochs=epochs,  # ← ABLATION PARAMETER
        logging_steps=10,
        save_strategy="no",  # Don't save intermediate checkpoints for ablation
        optim="paged_adamw_32bit",
        bf16=False,
        fp16=False,
        tf32=False,
        report_to="none",
        logging_first_step=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Batch size (per device): {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Output dir: {output_dir}\n")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=data_collator
    )
    
    # Train
    print(f"Starting training for {epochs} epoch(s)...")
    train_result = trainer.train()
    
    print(f"\n✅ Training completed!")
    print(f"  Final training loss: {train_result.training_loss:.4f}")
    
    # Save final model
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    
    # Save metadata
    metadata = {
        "epochs": epochs,
        "learning_rate": float(config["stage2_learning_rate"]),
        "batch_size": config["batch_size"],
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 8),
        "training_loss": float(train_result.training_loss),
        "checkpoint_path": output_dir,
        "output_dir": output_dir,
    }
    
    metadata_file = f"results/ablation_epochs{epochs}_metadata.json"
    os.makedirs("results", exist_ok=True)
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Metadata saved to: {metadata_file}")
    print(f"  Model saved to: {output_dir}")
    
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study: Stage 2 with varying epochs")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    args = parser.parse_args()
    
    if args.epochs not in [1, 2, 3]:
        print("⚠️  Warning: Expected epochs in [1, 2, 3] for ablation study")
    
    result = main(epochs=args.epochs)
    print("\n" + "="*60)
    print("ABLATION VARIANT COMPLETE")
    print("="*60)
