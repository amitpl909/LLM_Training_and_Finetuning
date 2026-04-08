"""
Minimal Inference - Using transformers pipeline API directly
No custom generation logic, just direct pipeline usage
"""

import json
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_model_and_pipes(checkpoint_path=None):
    """Load model with optional adapter."""
    model_id = config["student_model"]
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    if checkpoint_path:
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
    else:
        model = base_model
    
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path if checkpoint_path else model_id,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_tokens=100):
    """Use pipeline for generation."""
    try:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.0
        )
        
        result = pipe(prompt)
        if result and len(result) > 0:
            generated = result[0]["generated_text"]
            # Remove the prompt from the output
            response = generated.replace(prompt, "").strip()
            return response if response else "[no output]"
        return "[no output]"
    except Exception as e:
        return f"[error: {str(e)[:40]}]"


def main():
    os.makedirs("results", exist_ok=True)
    
    # Load eval datasets
    with open("data_prep/alpaca_eval.json") as f:
        alpaca_eval = json.load(f)
    
    with open("data_prep/stage2_json_instruct_eval.json") as f:
        json_eval = json.load(f)
    
    results = {
        "alpaca": [],
        "json": []
    }
    
    # Load base model
    print("Loading base model...")
    base_model, tokenizer = load_model_and_pipes()
    
    checkpoints = {
        "baseline": None,
        "phase1_alpaca": "checkpoints/stage1_alpaca_final",
        "phase2_json": "checkpoints/stage2_json_final"
    }
    
    for ckpt_name, ckpt_path in checkpoints.items():
        print(f"\n=== {ckpt_name} ===")
        
        if ckpt_path:
            print(f"Loading checkpoint: {ckpt_path}")
            model, tokenizer = load_model_and_pipes(ckpt_path)
        else:
            model, tokenizer = load_model_and_pipes()
        
        model.eval()
        
        # Process Alpaca
        print(f"Processing {len(alpaca_eval)} Alpaca prompts...")
        for i, item in enumerate(alpaca_eval[:10]):  # Process first 10 for testing
            instruction = item["instruction"]
            prompt = f"User: {instruction}\n\nAssistant:"
            response = generate_text(model, tokenizer, prompt)
            
            result_item = {
                "instruction": instruction,
                "input": item.get("input", ""),
                "expected_output": item.get("output", ""),
                f"{ckpt_name}_response": response
            }
            
            if ckpt_name == "baseline":
                results["alpaca"].append(result_item)
            else:
                results["alpaca"][i][f"{ckpt_name}_response"] = response
    
    # Save minimal results
    with open("results/inference_minimal.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Inference complete!")
    print(f"Saved to results/inference_minimal.json")


if __name__ == "__main__":
    main()
