import json
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import os

# Load central configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def generate_responses(model, tokenizer, prompts, max_new_tokens=150):
    responses = []
    for prompt in tqdm(prompts, desc="Generating"):
        # Format for Phi-3.5 (standard User/Assistant template)
        formatted_prompt = f"User: {prompt}\nAssistant:"
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and extract only the new assistant text
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_only = full_output.replace(formatted_prompt, "").strip()
        responses.append(response_only)
    return responses

def main():
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    print("Loading evaluation datasets...")
    # 1. Load the Alpaca Eval Prompts (held-out evaluation set)
    with open("data_prep/alpaca_eval.json", "r") as f:
        alpaca_data = json.load(f)
        alpaca_prompts = [ex["instruction"] for ex in alpaca_data]

    # 2. Load the JSON Instruct Eval Set (held-out, not training data!)
    with open("data_prep/stage2_json_instruct_eval.json", "r") as f:
        json_eval_data = json.load(f)
        json_prompts = []
        for ex in json_eval_data:
            prompt = f"{ex['instruction']}\nInput: {ex['input']}" if ex['input'] else ex['instruction']
            json_prompts.append({
                "prompt": prompt,
                "expected_output": ex["output"]
            })

    all_prompts = {
        "alpaca": [(p, None) for p in alpaca_prompts],  # (prompt, expected_output)
        "json": [(p["prompt"], p["expected_output"]) for p in json_prompts]
    }
    # Initialize results as list of dicts, one per prompt
    results = {
        "alpaca": [{} for _ in alpaca_prompts],
        "json": [{} for _ in json_prompts]
    }

    print("Loading Base Model (Phi-3.5-mini)...")
    model_id = config["student_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Load base model in 4-bit to save memory during inference
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.float16, 
        load_in_4bit=True,
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
            # Attach the specific fine-tuned adapter to the base model
            model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            model = base_model

        model.eval()

        for dataset_name, prompt_pairs in all_prompts.items():
            prompts = [p[0] for p in prompt_pairs]  # Extract just the prompt text
            print(f"Processing {len(prompts)} {dataset_name} samples...")
            responses = generate_responses(model, tokenizer, prompts)
            
            # Organize results for the LLM Judge to read later
            for i, (prompt, expected_output) in enumerate(prompt_pairs):
                if model_name == "baseline":
                    # Initialize new entry on first model
                    results[dataset_name][i] = {
                        "prompt": prompt,
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