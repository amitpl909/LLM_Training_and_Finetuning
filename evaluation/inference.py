import json
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from tqdm import tqdm
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))  # Add src directory to path
from data_utils import format_instruction_for_inference, print_template_info
from transformers import BitsAndBytesConfig

# Load central configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def generate_responses(model, tokenizer, prompts_with_context, max_new_tokens=150):
    """
    Generate responses using the instructor's template pattern.
    Simplified to avoid DynamicCache issues with Phi3.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        prompts_with_context: List of tuples (instruction, input_text)
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        List of generated responses
    """
    responses = []
    for instruction, input_text in tqdm(prompts_with_context, desc="Generating"):
        # Use the instructor's template for consistent formatting
        formatted_prompt = format_instruction_for_inference(instruction, input_text)
        
        try:
            # Minimal tokenization - just encode the prompt
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                # Simplest generation: no caching, greedy only
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    max_new_tokens=min(max_new_tokens, 100),
                    do_sample=False  # Greedy decoding only
                )
            
            # Decode and extract only the new generated text
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_only = full_output.replace(formatted_prompt, "").strip()
            responses.append(response_only if response_only else "[no output]")
        except Exception as e:
            # Continue on error - inference is resilient
            responses.append(f"[error: {str(e)[:30]}]")
            
    return responses

def main():
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    print_template_info()

    print("Loading evaluation datasets...")
    # 1. Load the Alpaca Eval Prompts (held-out evaluation set)
    with open("data_prep/alpaca_eval.json", "r") as f:
        alpaca_data = json.load(f)
        # Format as (instruction, input) tuples for template
        alpaca_prompts = [(ex["instruction"], ex.get("input", "")) for ex in alpaca_data]

    # 2. Load the JSON Instruct Eval Set (held-out, not training data!)
    with open("data_prep/stage2_json_instruct_eval.json", "r") as f:
        json_eval_data = json.load(f)
        json_prompts = []
        for ex in json_eval_data:
            # Format as (instruction, input) tuples for template
            json_prompts.append((ex["instruction"], ex.get("input", "")))

    all_prompts = {
        "alpaca": alpaca_prompts,
        "json": json_prompts
    }
    
    eval_data = {
        "alpaca": alpaca_data,
        "json": json_eval_data
    }
    
    # Initialize results as list of dicts, one per prompt
    results = {
        "alpaca": [{} for _ in alpaca_prompts],
        "json": [{} for _ in json_prompts]
    }

    print("Loading Base Model (Phi-3.5-mini)...")
    model_id = config["student_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure 4-bit quantization for inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load base model in 4-bit to save memory during inference
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    # These are the three stages of the model we need to compare
    models_to_test = {
        "baseline": None,                             # Raw model
        "phase1_alpaca": "checkpoints/stage1_alpaca_final", # After general tuning
        "phase2_json": "checkpoints/stage2_json_final"      # After JSON specific tuning
    }

    for model_name, adapter_path in models_to_test.items():
        print(f"\n--- Starting Inference for: {model_name} ---")
        
        if adapter_path:
            if not os.path.exists(adapter_path):
                print(f"Skipping {model_name}: Adapter not found at {adapter_path}. Training might still be running.")
                continue
            # Reload base model for clean state before loading new adapter
            print(f"Reloading base model for {model_name}...")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                device_map="auto", 
                quantization_config=bnb_config,
                trust_remote_code=True
            )
            # Attach the specific fine-tuned adapter to the clean base model
            model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            model = base_model

        model.eval()

        for dataset_name, prompt_pairs in all_prompts.items():
            print(f"Processing {len(prompt_pairs)} {dataset_name} samples...")
            responses = generate_responses(model, tokenizer, prompt_pairs)
            
            # Organize results for the LLM Judge to read later
            for i, (instruction, input_text) in enumerate(prompt_pairs):
                expected_output = eval_data[dataset_name][i].get("output", "")
                if model_name == "baseline":
                    # Initialize new entry on first model
                    results[dataset_name][i] = {
                        "instruction": instruction,
                        "input": input_text,
                        "expected_output": expected_output,
                        "baseline_response": responses[i]
                    }
                else:
                    # Add response to existing entry
                    results[dataset_name][i][f"{model_name}_response"] = responses[i]

    # Save everything to a structured JSON for Phase 4b
    with open("results/inference_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nInference complete! Results saved to results/inference_results.json")

if __name__ == "__main__":
    main()