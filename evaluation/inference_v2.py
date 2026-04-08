"""
Fixed Inference Module - Simplified Generation for Phi-3.5

Simplified to avoid DynamicCache issues with minimal dependencies on model-specific behavior.
Uses text-generation-webui compatible approach with careful tokenization.
"""

import json
import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from tqdm import tqdm
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_utils import format_instruction_for_inference, print_template_info

# Load central configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def generate_response_simple(model, tokenizer, instruction_text, input_text, max_new_tokens=100):
    """
    Generate a single response using ultra-simplified generation.
    Avoids any caching issues by using greedy decoding with no special options.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        instruction_text: The instruction
        input_text: The input context
        max_new_tokens: Max tokens to generate
    
    Returns:
        Generated response text, or error message if failed
    """
    try:
        # Format prompt
        formatted_prompt = format_instruction_for_inference(instruction_text, input_text)
        
        # Encode with explicit parameters
        inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
        if inputs.shape[1] > 512:
            inputs = inputs[:, -512:]  # Truncate to max length
        
        inputs = inputs.to(model.device)
        
        # Ultra-simple generation: no options, just greedy
        with torch.no_grad():
            # Use eos_token_id or None to generate until max_new_tokens
            attention_mask = torch.ones_like(inputs)
            
            # Iterative generation to avoid caching issues
            max_tokens_to_predict = min(max_new_tokens, 100)
            generated_ids = inputs.clone()
            
            for _ in range(max_tokens_to_predict):
                # Get logits for next token
                outputs = model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask[:, :generated_ids.shape[1]],
                    return_dict=True
                )
                
                # Get last token logits
                next_token_logits = outputs.logits[:, -1, :]
                
                # Greedy selection
                next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
                
                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
                
                # Extend attention mask
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
                ], dim=1)
                
                # Stop if EOS token
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
        
        # Decode output
        full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        response = full_text.replace(formatted_prompt, "").strip()
        
        # Clean up response
        if not response or len(response) < 5:
            return "[no output]"
        
        return response
        
    except Exception as e:
        error_msg = str(e)[:40]
        return f"[error: {error_msg}]"


def main():
    """Generate responses at all three checkpoints."""
    os.makedirs("results", exist_ok=True)
    
    print_template_info()

    print("Loading evaluation datasets...")
    
    # Load Alpaca eval prompts
    with open("data_prep/alpaca_eval.json", "r") as f:
        alpaca_data = json.load(f)
        alpaca_prompts = [(ex["instruction"], ex.get("input", "")) for ex in alpaca_data]

    # Load JSON eval prompts
    with open("data_prep/stage2_json_instruct_eval.json", "r") as f:
        json_eval_data = json.load(f)
        json_prompts = [(ex["instruction"], ex.get("input", "")) for ex in json_eval_data]

    all_prompts = {
        "alpaca": alpaca_prompts,
        "json": json_prompts
    }
    
    eval_data = {
        "alpaca": alpaca_data,
        "json": json_eval_data
    }
    
    # Initialize results
    results = {
        "alpaca": [{} for _ in alpaca_prompts],
        "json": [{} for _ in json_prompts]
    }

    print(f"\nLoading Phi-3.5-mini-instruct base model...")
    model_id = config["student_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load base model with error handling
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="eager"  # Avoid DynamicCache
        )
    except Exception as e:
        print(f"⚠️  Attention implementation not supported, retrying without eager: {e}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True
        )
    
    base_model.eval()
    
    # Three checkpoints to test
    models_to_test = {
        "baseline": None,
        "phase1_alpaca": "checkpoints/stage1_alpaca_final",
        "phase2_json": "checkpoints/stage2_json_final"
    }

    for model_name, adapter_path in models_to_test.items():
        print(f"\n{'='*60}")
        print(f"Generating responses: {model_name}")
        print(f"{'='*60}")
        
        if adapter_path:
            if not os.path.exists(adapter_path):
                print(f"⚠️  Skipping {model_name}: Adapter not found at {adapter_path}")
                continue
            
            # Reload base for clean state
            print(f"Reloading base model for {model_name}...")
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    attn_implementation="eager"
                )
            except:
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    quantization_config=bnb_config,
                    trust_remote_code=True
                )
            
            # Load adapter
            model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            model = base_model
        
        model.eval()

        # Generate on both datasets
        for dataset_name, prompt_pairs in all_prompts.items():
            print(f"\nProcessing {len(prompt_pairs)} {dataset_name} prompts...")
            
            for i, (instruction, input_text) in enumerate(tqdm(prompt_pairs, desc=f"{dataset_name} generation")):
                # Generate response
                response = generate_response_simple(model, tokenizer, instruction, input_text)
                
                # Store result
                expected_output = eval_data[dataset_name][i].get("output", "")
                
                if model_name == "baseline":
                    results[dataset_name][i] = {
                        "instruction": instruction,
                        "input": input_text,
                        "expected_output": expected_output,
                        "baseline_response": response
                    }
                else:
                    results[dataset_name][i][f"{model_name}_response"] = response
    
    # Save results
    with open("results/inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Inference complete! Results saved to results/inference_results.json")
    
    # Quick validation
    print(f"\nValidation:")
    print(f"  Alpaca responses: {len(results['alpaca'])}")
    print(f"  JSON responses: {len(results['json'])}")
    
    # Sample check
    if results['alpaca']:
        sample = results['alpaca'][0]
        print(f"\n  Sample Alpaca response (baseline): {sample.get('baseline_response', 'N/A')[:80]}...")
    
    if results['json']:
        sample = results['json'][0]
        print(f"  Sample JSON response (baseline): {sample.get('baseline_response', 'N/A')[:80]}...")


if __name__ == "__main__":
    main()
