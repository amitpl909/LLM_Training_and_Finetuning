import json
import yaml
import os
import re
from openai import OpenAI
from rouge_score import rouge_scorer  # Required: pip install rouge-score

# Load central configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize the Judge Client (Llama 3.1 70B as per Section 2.7 & 8)
client = OpenAI(
    base_url=config["judge_api_url"],
    api_key=config["judge_api_key"]
)

def calculate_json_metrics(response_text, expected_output):
    """Calculates Section 4.3 required JSON metrics"""
    metrics = {"valid": 0, "schema": 0, "exact": 0}
    
    # Clean markdown backticks if present
    clean_text = re.sub(r'^```json\s*|```$', '', response_text.strip(), flags=re.MULTILINE).strip()
    
    try:
        parsed_resp = json.loads(clean_text)
        metrics["valid"] = 1
        
        # Load expected for comparison
        expected_json = json.loads(expected_output)
        
        # Exact Match
        if parsed_resp == expected_json:
            metrics["exact"] = 1
            metrics["schema"] = 1
        else:
            # Schema Compliance: Check if all keys exist
            if isinstance(parsed_resp, dict) and isinstance(expected_json, dict):
                if set(expected_json.keys()).issubset(set(parsed_resp.keys())):
                    metrics["schema"] = 1
    except:
        pass
    return metrics

def get_rouge_l(prediction, reference):
    """Calculates ROUGE-L for Forgetting Analysis"""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores['rougeL'].fmeasure

def get_judge_decision(prompt, resp_a, resp_b):
    """Pairwise Comparison Judge (Section 4.2 methodology)"""
    system_prompt = "You are an impartial judge. Evaluate which response follows instructions better."
    user_content = f"Instruction: {prompt}\n\nResponse A: {resp_a}\n\nResponse B: {resp_b}\n\n" \
                   "Answer ONLY with '[[A]]' or '[[B]]' followed by a one-sentence reason."
    
    try:
        response = client.chat.completions.create(
            model=config["judge_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0 # Strict determinism
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def main():
    input_file = "results/inference_results.json"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run inference first.")
        return

    with open(input_file, "r") as f:
        data = json.load(f)

    # Initialize Table 4.1 Comparison Data
    results_table = {
        "alpaca": {"P1_win_vs_Base": 0, "P2_win_vs_P1": 0, "avg_rougeL_P1": 0.0, "avg_rougeL_P2": 0.0},
        "json": {"validity": 0.0, "schema": 0.0, "exact": 0.0}
    }

    # 1. ALPACA EVALUATION & FORGETTING ANALYSIS
    print("\n--- Running Alpaca Forgetting Analysis ---")
    total_alpaca = len(data["alpaca"])
    rouge_p1, rouge_p2 = [], []

    for item in data["alpaca"]:
        # Stage 1 vs Base
        res1 = get_judge_decision(item["prompt"], item["baseline_response"], item["phase1_alpaca_response"])
        if "[[B]]" in res1: results_table["alpaca"]["P1_win_vs_Base"] += 1
        
        # Stage 2 vs Stage 1 (The Forgetting Check)
        res2 = get_judge_decision(item["prompt"], item["phase1_alpaca_response"], item["phase2_json_response"])
        if "[[B]]" in res2: results_table["alpaca"]["P2_win_vs_P1"] += 1
        
        # Calculate ROUGE-L relative to baseline (or ground truth if available)
        rouge_p1.append(get_rouge_l(item["phase1_alpaca_response"], item["baseline_response"]))
        rouge_p2.append(get_rouge_l(item["phase2_json_response"], item["baseline_response"]))

    results_table["alpaca"]["avg_rougeL_P1"] = sum(rouge_p1)/total_alpaca
    results_table["alpaca"]["avg_rougeL_P2"] = sum(rouge_p2)/total_alpaca

    # 2. JSON STRUCTURED EVALUATION
    print("--- Running JSON Validity Metrics ---")
    total_json = len(data["json"])
    # Use the evaluation set (held-out, not training data)
    v_sum, s_sum, e_sum = 0, 0, 0
    for i, item in enumerate(data["json"]):
        # expected_output is stored in the inference results already
        expected = item.get("expected_output", "{}")
        
        # Evaluate using phase2_json response (final tuned model)
        if "phase2_json_response" in item:
            m = calculate_json_metrics(item["phase2_json_response"], expected)
        else:
            # If phase2 not ready, use phase1
            m = calculate_json_metrics(item["phase1_alpaca_response"], expected)
        
        v_sum += m["valid"] 
        s_sum += m["schema"]
        e_sum += m["exact"]

    results_table["json"]["validity"] = (v_sum / total_json) * 100
    results_table["json"]["schema"] = (s_sum / total_json) * 100
    results_table["json"]["exact"] = (e_sum / total_json) * 100

    # Save final statistics for the Three-Checkpoint Comparison Table
    with open("results/final_evaluation_report.json", "w") as f:
        json.dump(results_table, f, indent=4)
    
    print("\nFINAL REPORT READY FOR BLOG POST")
    print(json.dumps(results_table, indent=4))

if __name__ == "__main__":
    main()