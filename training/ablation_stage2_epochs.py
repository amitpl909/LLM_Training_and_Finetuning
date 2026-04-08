"""
Ablation Study: Stage 2 Training with Varying Epochs
Tests how Stage 2 epochs (1, 2, 3) affects catastrophic forgetting

Usage:
    # For specific epochs variant
    python training/ablation_stage2_epochs.py --epochs 1

    # Or use SLURM for all variants
    sbatch hpc_scripts/run_ablation.slurm
"""

import json
import yaml
import torch
import argparse
from pathlib import Path
from typing import Dict, Any
import sys

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load training configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_ablation_dataset(
    data_path: str,
    tokenizer,
    max_length: int = 512,
    dataset_fraction: float = 1.0
) -> Dataset:
    """Load and prepare JSON training dataset for ablation."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Apply dataset fraction if needed
    if dataset_fraction < 1.0:
        cutoff = int(len(data) * dataset_fraction)
        data = data[:cutoff]
    
    def format_prompt(example):
        # Format: "User: <input>\n\nAssistant: <output>"
        # Note: JSON data has 'output' field, not 'response'
        prompt = f"User: {example['instruction']}\n\nAssistant: {example['output']}"
        tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    
    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_prompt, batched=False)
    return dataset


def train_ablation_variant(
    epochs: int,
    output_suffix: str = "",
    learning_rate: float = 2e-4,
    dataset_fraction: float = 1.0,
) -> Dict[str, Any]:
    """
    Train Stage 2 with specified epochs variant
    
    Args:
        epochs: Number of training epochs for Stage 2
        output_suffix: Suffix for output checkpoint directory
        learning_rate: Training learning rate
        dataset_fraction: Fraction of training data to use
    
    Returns:
        Dictionary with training metrics and paths
    """
    config = load_config()
    
    # Model setup
    model_id = config["student_model"]
    
    # Load Stage 1 checkpoint as base
    stage1_checkpoint = Path("checkpoints/stage1_alpaca_final")
    if not stage1_checkpoint.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_checkpoint}")
    
    print(f"\n{'='*60}")
    print(f"Ablation: Stage 2 Training (epochs={epochs})")
    print(f"{'='*60}")
    
    # Load base model with quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load Stage 1 adapter
    model = PeftModel.from_pretrained(base_model, str(stage1_checkpoint))
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset
    train_data_path = "data_prep/stage2_json_instruct_train.json"
    train_dataset = prepare_ablation_dataset(
        train_data_path,
        tokenizer,
        dataset_fraction=dataset_fraction,
    )
    
    # Add evaluation dataset
    eval_data_path = "data_prep/stage2_json_instruct_eval.json"
    eval_dataset = prepare_ablation_dataset(eval_data_path, tokenizer)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print(f"Learning rate: {learning_rate}")
    
    # Setup training arguments for ablation variant
    output_dir = f"checkpoints/ablation_epochs{epochs}_{output_suffix}".rstrip("_")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("batch_size", 8),
        per_device_eval_batch_size=config.get("batch_size", 8),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        num_train_epochs=epochs,  # ← ABLATION PARAMETER
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        save_strategy="steps",
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="paged_adamw_32bit",
        seed=42,
        report_to="none",
        bf16=False,
        fp16=True,
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda batch: {
            'input_ids': torch.stack([torch.tensor(b['input_ids']) if isinstance(b['input_ids'], list) else b['input_ids'] for b in batch]),
            'attention_mask': torch.stack([torch.tensor(b['attention_mask']) if isinstance(b['attention_mask'], list) else b['attention_mask'] for b in batch]),
            'labels': torch.stack([torch.tensor(b['labels']) if isinstance(b['labels'], list) else b['labels'] for b in batch]),
        },
    )
    
    # Train
    print("\n🔄 Starting training...")
    train_result = trainer.train()
    
    # Evaluate
    eval_result = trainer.evaluate()
    
    # Save final model
    final_checkpoint = Path(output_dir) / "final_model"
    model.save_pretrained(final_checkpoint)
    
    print(f"\n✅ Training complete!")
    print(f"   Output: {output_dir}")
    print(f"   Training loss: {train_result.training_loss:.4f}")
    print(f"   Eval loss: {eval_result.get('eval_loss', 'N/A')}")
    
    # Return metrics
    return {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "dataset_fraction": dataset_fraction,
        "output_dir": output_dir,
        "training_loss": train_result.training_loss,
        "eval_loss": eval_result.get("eval_loss"),
        "checkpoint_path": str(final_checkpoint),
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation study: Stage 2 epochs variation")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for Stage 2")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--dataset_fraction", type=float, default=1.0, help="Fraction of dataset to use")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output directory")
    
    args = parser.parse_args()
    
    # Run ablation variant
    result = train_ablation_variant(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        dataset_fraction=args.dataset_fraction,
        output_suffix=args.output_suffix,
    )
    
    # Save result metadata
    result_file = Path("results") / f"ablation_epochs{args.epochs}_metadata.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n📊 Result metadata saved to {result_file}")


if __name__ == "__main__":
    main()
