"""
Comprehensive LLM Judge Evaluation following Assignment Section 9 spec.

Features:
- Structured evaluation on 6 dimensions
- Randomized response order to reduce bias
- TIE detection and reporting
- Automatic metrics: ROUGE-1/2/L, BERTScore, output length
- Per-category breakdown for forgetting analysis
"""

import json
import yaml
import os
import re
import random
from openai import OpenAI
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from collections import defaultdict

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize judge client
client = OpenAI(
    base_url=config["judge_api_url"],
    api_key=config["judge_api_key"]
)

def calculate_json_metrics(response_text, expected_output):
    """Section 4.3: JSON validity, schema compliance, exact match."""
    metrics = {"valid": 0, "schema": 0, "exact": 0}
    clean_text = re.sub(r'^```json\s*|```$', '', response_text.strip(), flags=re.MULTILINE).strip()
    
    try:
        parsed_resp = json.loads(clean_text)
        metrics["valid"] = 1
        expected_json = json.loads(expected_output)
        
        if parsed_resp == expected_json:
            metrics["exact"] = 1
            metrics["schema"] = 1
        else:
            if isinstance(parsed_resp, dict) and isinstance(expected_json, dict):
                if set(expected_json.keys()).issubset(set(parsed_resp.keys())):
                    metrics["schema"] = 1
    except:
        pass
    return metrics

def get_rouge_scores(prediction, reference):
    """Calculate ROUGE-1, ROUGE-2, ROUGE-L."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure,
    }

def get_bert_score(prediction, reference):
    """Calculate BERTScore F1."""
    try:
        _, _, f1 = bert_score_fn([prediction], [reference], lang="en")
        return float(f1[0])
    except:
        return 0.0

def get_output_length(response):
    """Count tokens in response (approximation: split by whitespace)."""
    return len(response.split())

def check_task_completion(response, instruction):
    """Heuristic: did model attempt to follow instruction?"""
    # Simple check: response is non-empty and reasonable length
    if not response or len(response.strip()) < 5:
        return 0.0  # Failed - too short
    if len(response.split()) > 1000:
        return 0.5  # Unclear - too long (might be noise)
    return 1.0  # Success - attempted

def get_structured_judge_scores(prompt, resp_a, resp_b, checkpoint_a="CP1", checkpoint_b="CP2"):
    """
    Get structured scores from judge on 6 dimensions.
    Returns scores for both responses.
    """
    system_prompt = """You are an expert LLM evaluator. Score both responses on these 6 dimensions (1-5 scale):
1. instruction_following: Did it follow the instruction?
2. correctness: Is the answer factually correct?
3. clarity: Is it clear and well-written?
4. completeness: Does it fully address the instruction?
5. structured_output_validity: For JSON tasks, is output valid/well-formed?
6. hallucination_risk: Risk of fabricated info (1=high risk, 5=low risk)?

Return ONLY valid JSON (no markdown, no explanation):
{"response_a_scores": {...}, "response_b_scores": {...}, "winner": "A|B|TIE", "justification": "..."}"""
    
    user_content = f"""Instruction: {prompt}

Response A: {resp_a}

Response B: {resp_b}"""
    
    try:
        response = client.chat.completions.create(
            model=config["judge_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content.strip()
        result = json.loads(result_text)
        
        # Validate and normalize scores
        for key in ["response_a_scores", "response_b_scores"]:
            if key in result:
                for dim in ["instruction_following", "correctness", "clarity", "completeness", "structured_output_validity", "hallucination_risk"]:
                    if dim not in result[key]:
                        result[key][dim] = 3  # Default middle score
                    else:
                        result[key][dim] = min(5, max(1, int(result[key][dim])))  # Clamp 1-5
        
        # Ensure winner is valid
        if result.get("winner") not in ["A", "B", "TIE"]:
            result["winner"] = "TIE"
        
        result["checkpoint_a"] = checkpoint_a
        result["checkpoint_b"] = checkpoint_b
        
        return result
    except Exception as e:
        print(f"Judge error: {e}")
        return None

def main():
    input_file = "results/inference_results.json"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run inference first.")
        return

    with open(input_file, "r") as f:
        data = json.load(f)

    # Results aggregation
    results = {
        "alpaca_judgments": [],
        "json_judgments": [],
        "alpaca_metrics": defaultdict(list),
        "json_metrics": defaultdict(list),
    }

    # ====== ALPACA EVALUATION (Section 4.2) ======
    print("\n" + "="*60)
    print("ALPACA EVALUATION - Pairwise Comparisons")
    print("="*60)
    
    total_alpaca = len(data["alpaca"])
    alpaca_judges = {
        "cp1_vs_base": {"A_wins": 0, "B_wins": 0, "ties": 0, "avg_scores": {}},
        "cp2_vs_cp1": {"A_wins": 0, "B_wins": 0, "ties": 0, "avg_scores": {}},
    }

    for idx, item in enumerate(data["alpaca"]):
        print(f"[{idx+1}/{total_alpaca}] Alpaca eval: {item['instruction'][:50]}...")
        
        # === CP1 vs Base ===
        # Randomize order
        if random.choice([True, False]):
            resp_first, resp_second = item["phase1_alpaca_response"], item["baseline_response"]
            labels_first, labels_second = "CP1", "Base"
        else:
            resp_first, resp_second = item["baseline_response"], item["phase1_alpaca_response"]
            labels_first, labels_second = "Base", "CP1"
        
        judge_result = get_structured_judge_scores(
            item["instruction"], resp_first, resp_second, labels_first, labels_second
        )
        
        if judge_result:
            # Track who actually won (accounting for swap)
            winner = judge_result["winner"]
            if labels_first == "CP1":  # First responder was CP1
                if winner == "A":
                    alpaca_judges["cp1_vs_base"]["A_wins"] += 1
                elif winner == "B":
                    alpaca_judges["cp1_vs_base"]["B_wins"] += 1
                else:
                    alpaca_judges["cp1_vs_base"]["ties"] += 1
            else:  # First responder was Base
                if winner == "A":
                    alpaca_judges["cp1_vs_base"]["B_wins"] += 1
                elif winner == "B":
                    alpaca_judges["cp1_vs_base"]["A_wins"] += 1
                else:
                    alpaca_judges["cp1_vs_base"]["ties"] += 1
            
            results["alpaca_judgments"].append(judge_result)
            
            # Collect automatic metrics
            for resp, label in [(item["phase1_alpaca_response"], "cp1"), (item["baseline_response"], "base")]:
                rouge = get_rouge_scores(resp, item["expected_output"])
                results["alpaca_metrics"][f"{label}_rouge1"].append(rouge["rouge1"])
                results["alpaca_metrics"][f"{label}_rouge2"].append(rouge["rouge2"])
                results["alpaca_metrics"][f"{label}_rougeL"].append(rouge["rougeL"])
                results["alpaca_metrics"][f"{label}_bertscore"].append(get_bert_score(resp, item["expected_output"]))
                results["alpaca_metrics"][f"{label}_length"].append(get_output_length(resp))
                results["alpaca_metrics"][f"{label}_completion"].append(check_task_completion(resp, item["instruction"]))
        
        # === CP2 vs CP1 ===
        if random.choice([True, False]):
            resp_first, resp_second = item["phase2_json_response"], item["phase1_alpaca_response"]
            labels_first, labels_second = "CP2", "CP1"
        else:
            resp_first, resp_second = item["phase1_alpaca_response"], item["phase2_json_response"]
            labels_first, labels_second = "CP1", "CP2"
        
        judge_result = get_structured_judge_scores(
            item["instruction"], resp_first, resp_second, labels_first, labels_second
        )
        
        if judge_result:
            winner = judge_result["winner"]
            if labels_first == "CP2":
                if winner == "A":
                    alpaca_judges["cp2_vs_cp1"]["A_wins"] += 1
                elif winner == "B":
                    alpaca_judges["cp2_vs_cp1"]["B_wins"] += 1
                else:
                    alpaca_judges["cp2_vs_cp1"]["ties"] += 1
            else:
                if winner == "A":
                    alpaca_judges["cp2_vs_cp1"]["B_wins"] += 1
                elif winner == "B":
                    alpaca_judges["cp2_vs_cp1"]["A_wins"] += 1
                else:
                    alpaca_judges["cp2_vs_cp1"]["ties"] += 1
            
            results["alpaca_judgments"].append(judge_result)
            
            # Collect automatic metrics for CP2
            rouge = get_rouge_scores(item["phase2_json_response"], item["expected_output"])
            results["alpaca_metrics"]["cp2_rouge1"].append(rouge["rouge1"])
            results["alpaca_metrics"]["cp2_rouge2"].append(rouge["rouge2"])
            results["alpaca_metrics"]["cp2_rougeL"].append(rouge["rougeL"])
            results["alpaca_metrics"]["cp2_bertscore"].append(get_bert_score(item["phase2_json_response"], item["expected_output"]))
            results["alpaca_metrics"]["cp2_length"].append(get_output_length(item["phase2_json_response"]))
            results["alpaca_metrics"]["cp2_completion"].append(check_task_completion(item["phase2_json_response"], item["instruction"]))

    # Compute Alpaca summary
    print("\n" + "="*60)
    print("ALPACA RESULTS SUMMARY")
    print("="*60)
    for comparison, stats in alpaca_judges.items():
        total = stats["A_wins"] + stats["B_wins"] + stats["ties"]
        if total > 0:
            print(f"\n{comparison}:")
            print(f"  A wins: {stats['A_wins']} ({100*stats['A_wins']//total}%)")
            print(f"  B wins: {stats['B_wins']} ({100*stats['B_wins']//total}%)")
            print(f"  Ties: {stats['ties']} ({100*stats['ties']//total}%)")

    # ====== JSON EVALUATION (Section 4.3) ======
    print("\n" + "="*60)
    print("JSON EVALUATION - Structured Output Metrics")
    print("="*60)
    
    total_json = len(data["json"])
    json_metrics = {"validity": [], "schema": [], "exact": []}

    for idx, item in enumerate(data["json"]):
        print(f"[{idx+1}/{total_json}] JSON: {item['instruction'][:50]}...")
        
        # Phase 2 is final model
        metrics = calculate_json_metrics(item["phase2_json_response"], item["expected_output"])
        json_metrics["validity"].append(metrics["valid"])
        json_metrics["schema"].append(metrics["schema"])
        json_metrics["exact"].append(metrics["exact"])

    # Compute JSON summary
    print("\n" + "="*60)
    print("JSON RESULTS SUMMARY")
    print("="*60)
    if total_json > 0:
        print(f"JSON Validity: {100*sum(json_metrics['validity'])//total_json}%")
        print(f"Schema Compliance: {100*sum(json_metrics['schema'])//total_json}%")
        print(f"Exact Match: {100*sum(json_metrics['exact'])//total_json}%")

    # ====== SAVE RESULTS ======
    # Compute averages for automatic metrics
    for key in results["alpaca_metrics"]:
        values = results["alpaca_metrics"][key]
        if values:
            results["alpaca_metrics"][key] = {
                "mean": sum(values) / len(values),
                "count": len(values)
            }

    with open("results/judge_evaluation_complete.json", "w") as f:
        json.dump({
            "alpaca_judgments": results["alpaca_judgments"],
            "alpaca_metrics_summary": dict(results["alpaca_metrics"]),
            "alpaca_pairwise_summary": dict(alpaca_judges),
            "json_metrics_summary": {
                "validity_rate": sum(json_metrics["validity"]) / total_json * 100 if total_json > 0 else 0,
                "schema_rate": sum(json_metrics["schema"]) / total_json * 100 if total_json > 0 else 0,
                "exact_match_rate": sum(json_metrics["exact"]) / total_json * 100 if total_json > 0 else 0,
            }
        }, f, indent=4)

    print("\n✅ Judge evaluation complete!")
    print(f"Results saved to results/judge_evaluation_complete.json")

if __name__ == "__main__":
    main()
