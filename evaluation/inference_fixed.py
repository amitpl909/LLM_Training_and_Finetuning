#!/usr/bin/env python3
"""
Simplified inference script - generates responses from 3 checkpoints
Uses direct model.generate() with proper configuration
"""

import os
import sys
import json
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

def load_checkpoints():
    """Load all 3 checkpoints"""
    checkpoints = {
        'baseline': 'microsoft/Phi-3.5-mini-instruct',
        'stage1_alpaca': 'checkpoints/stage1_alpaca_final/',
        'stage2_json': 'checkpoints/stage2_json_final/',
    }
    return checkpoints

def generate_responses(model_name, checkpoint_path, prompts, max_length=200):
    """Generate responses from a checkpoint"""
    print(f"\n{'='*60}")
    print(f"Loading: {model_name}")
    print(f"{'='*60}")
    
    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        'microsoft/Phi-3.5-mini-instruct',
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
    )
    
    # Load adapter if not baseline
    if model_name != 'baseline':
        print(f"Loading adapter from {checkpoint_path}...")
        model = PeftModel.from_pretrained(model, checkpoint_path)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3.5-mini-instruct', trust_remote_code=True)
    
    # Generate responses
    responses = []
    model.eval()
    
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            try:
                # Format prompt
                formatted_prompt = f"Instruction: {prompt['instruction']}\nInput: {prompt.get('input', '')}\n\nOutput:"
                
                # Tokenize
                inputs = tokenizer(formatted_prompt, return_tensors="pt", max_length=512, truncation=True).to(model.device)
                
                # Generate
                outputs = model.generate(
                    **inputs,
                    max_length=min(512 + max_length, 1024),
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                
                # Decode
                response_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                responses.append({
                    'prompt_idx': i,
                    'instruction': prompt['instruction'],
                    'response': response_text[:200]  # Limit to 200 chars
                })
                
                if (i + 1) % 10 == 0:
                    print(f"  Generated {i+1}/{len(prompts)} responses...")
                    
            except Exception as e:
                print(f"  Error on prompt {i}: {str(e)[:100]}")
                responses.append({
                    'prompt_idx': i,
                    'instruction': prompt['instruction'],
                    'response': f"[error: {str(e)[:50]}]"
                })
    
    print(f"✅ Generated {len(responses)} responses from {model_name}")
    return responses

def main():
    print("SIMPLIFIED INFERENCE PIPELINE")
    print("=" * 60)
    
    # Load eval prompts
    print("\nLoading evaluation prompts...")
    alpaca_prompts = json.load(open('data_prep/alpaca_eval.json'))
    json_prompts = json.load(open('data_prep/stage2_json_instruct_eval.json'))
    
    all_prompts = alpaca_prompts[:30] + json_prompts[:30]  # Use 30 each for speed
    print(f"Loaded {len(all_prompts)} prompts for inference")
    
    # Load checkpoints
    checkpoints = load_checkpoints()
    
    # Generate responses from each checkpoint
    all_results = {}
    for model_name, checkpoint_path in checkpoints.items():
        try:
            responses = generate_responses(model_name, checkpoint_path, all_prompts)
            all_results[model_name] = responses
        except Exception as e:
            print(f"❌ Failed to generate from {model_name}: {e}")
            all_results[model_name] = []
    
    # Save results
    output_file = 'results/inference_results_simple.json'
    os.makedirs('results', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Saved inference results to {output_file}")
    print(f"   Total responses: {sum(len(r) for r in all_results.values())}")

if __name__ == '__main__':
    main()
