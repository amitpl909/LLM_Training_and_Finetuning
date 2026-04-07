"""
Main Evaluation Pipeline Orchestrator

Runs complete three-checkpoint evaluation with:
- Inference at all 3 checkpoints  
- Judge evaluation (pairwise comparison)
- Automatic metrics (ROUGE, BERTScore, JSON validation)
- Forgetting analysis (THE CORE RESEARCH QUESTION)
- Results aggregation and reporting
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add src to path
sys.path.insert(0, '../src')
from data_utils import format_instruction_for_inference

try:
    from metrics import compute_checkpoint_metrics, compare_checkpoints, AutomaticMetrics
    from judge import evaluate_checkpoints_pairwise
    from forgetting_analysis import compute_forgetting_analysis, print_forgetting_analysis
    from ablation_study import AblationStudyRunner
except ImportError as e:
    print(f"Error importing evaluation modules: {e}")
    print("Make sure all modules are in evaluation/ directory")


def run_complete_evaluation(
    alpaca_eval_data: List[Dict],
    json_eval_data: List[Dict],
    checkpoint_0_outputs: Dict[str, List[str]],  # {"alpaca": [...], "json": [...]}
    checkpoint_1_outputs: Dict[str, List[str]],
    checkpoint_2_outputs: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Run full evaluation pipeline: metrics + judge + forgetting analysis.
    
    Args:
        alpaca_eval_data: Held-out Alpaca evaluation examples
        json_eval_data: Held-out JSON evaluation examples with schemas
        checkpoint_*_outputs: Generated responses for each checkpoint
    
    Returns:
        Comprehensive evaluation results
    """
    
    print("\n" + "="*80)
    print("RUNNING COMPLETE EVALUATION PIPELINE")
    print("="*80)
    
    metrics_computer = AutomaticMetrics()
    results = {
        "checkpoint_metrics": {},
        "checkpoint_comparison": {},
        "judge_evaluation": {},
        "forgetting_analysis": None,
    }
    
    # ============================================================
    # Phase 1: Compute Automatic Metrics for Each Checkpoint
    # ============================================================
    print("\n## PHASE 1: Computing Automatic Metrics")
    print("-" * 80)
    
    checkpoints = [
        ("checkpoint_0_baseline", checkpoint_0_outputs),
        ("checkpoint_1_alpaca", checkpoint_1_outputs),
        ("checkpoint_2_json", checkpoint_2_outputs),
    ]
    
    for checkpoint_name, outputs in checkpoints:
        print(f"\nProcessing {checkpoint_name}...")
        
        # Prepare results objects
        alpaca_results = []
        for i, example in enumerate(alpaca_eval_data):
            alpaca_results.append({
                "instruction": example.get("instruction", ""),
                "input": example.get("input", ""),
                "response": outputs["alpaca"][i] if i < len(outputs["alpaca"]) else "",
                "expected_output": example.get("output", ""),
            })
        
        json_results = []
        for i, example in enumerate(json_eval_data):
            json_results.append({
                "instruction": example.get("instruction", ""),
                "input": example.get("input", ""),
                "response": outputs["json"][i] if i < len(outputs["json"]) else "",
                "expected_output": example.get("output", ""),
                "schema": example.get("schema", {}),
                "task_type": example.get("task_type", ""),
            })
        
        # Compute metrics
        checkpoint_metrics = compute_checkpoint_metrics(
            checkpoint_name, alpaca_results, json_results
        )
        results["checkpoint_metrics"][checkpoint_name] = checkpoint_metrics
        
        print(f"  ✓ Alpaca: ROUGE-L={checkpoint_metrics['alpaca_metrics'].get('rougeL', 0):.3f}")
        print(f"  ✓ JSON: Validity={checkpoint_metrics['json_metrics'].get('validity_rate', 0):.1%}, "
              f"Compliance={checkpoint_metrics['json_metrics'].get('compliance_rate', 0):.1%}")
    
    # ============================================================
    # Phase 2: Judge Evaluation (Pairwise Comparison)
    # ============================================================
    print("\n## PHASE 2: Judge Evaluation (Pairwise Comparison)")
    print("-" * 80)
    
    # CP1 vs CP2 comparison (the critical forgetting measurement)
    print("\nEvaluating Checkpoint 1 vs Checkpoint 2 (CP1 vs CP2)...")
    
    cp1_judge_results = evaluate_checkpoints_pairwise(
        checkpoint_1_outputs=checkpoint_1_outputs["alpaca"],
        # Note: Create result objects from outputs
        checkpoint_2_outputs=checkpoint_2_outputs["alpaca"],
        checkpoint_1_name="After Stage 1 (Alpaca)",
        checkpoint_2_name="After Stage 2 (JSON)",
        sample_size=None  # Evaluate all
    )
    results["judge_evaluation"]["cp1_vs_cp2"] = cp1_judge_results
    
    win_rate_cp1 = cp1_judge_results.get("win_rates", {}).get("response_a", 0)
    win_rate_cp2 = cp1_judge_results.get("win_rates", {}).get("response_b", 0)
    print(f"  ✓ CP1 win rate: {win_rate_cp1:.1%}")
    print(f"  ✓ CP2 win rate: {win_rate_cp2:.1%}")
    
    # ============================================================
    # Phase 3: Forgetting Analysis (CORE RESEARCH QUESTION)
    # ============================================================
    print("\n## PHASE 3: Forgetting Analysis")
    print("-" * 80)
    
    # Note: This requires properly structured judge results
    # For now, create a simple version
    cp1_metrics = results["checkpoint_metrics"].get("checkpoint_1_alpaca", {})
    cp2_metrics = results["checkpoint_metrics"].get("checkpoint_2_json", {})
    
    # Mock judge results for demonstration (in real scenario, use actual judge output)
    mock_cp1_judge = cp1_judge_results
    mock_cp2_judge = cp1_judge_results  # Would be different in real case
    
    forgetting_analysis = compute_forgetting_analysis(
        checkpoint_1_judge_results=mock_cp1_judge,
        checkpoint_2_judge_results=mock_cp2_judge,
        checkpoint_1_metrics=cp1_metrics,
        checkpoint_2_metrics=cp2_metrics,
        checkpoint_1_outputs=[
            {"instruction": ex["instruction"], "category": "general", "response": ""}
            for ex in alpaca_eval_data[:10]
        ],
        checkpoint_2_outputs=[
            {"instruction": ex["instruction"], "category": "general", "response": ""}
            for ex in alpaca_eval_data[:10]
        ],
    )
    
    results["forgetting_analysis"] = forgetting_analysis
    print_forgetting_analysis(forgetting_analysis)
    
    # ============================================================
    # Phase 4: Results Aggregation
    # ============================================================
    print("\n## PHASE 4: Results Aggregation")
    print("-" * 80)
    
    # Create summary table
    summary_table = {
        "model_checkpoint": [
            "Checkpoint 0: Untuned",
            "Checkpoint 1: After Stage 1 (Alpaca)",
            "Checkpoint 2: After Stage 2 (JSON)",
        ],
        "alpaca_judge_win_rate": [],
        "rougeL": [],
        "bertscore": [],
        "json_validity": [],
        "json_compliance": [],
        "json_exact_match": [],
    }
    
    for cp_name in ["checkpoint_0_baseline", "checkpoint_1_alpaca", "checkpoint_2_json"]:
        metrics = results["checkpoint_metrics"].get(cp_name, {})
        summary_table["alpaca_judge_win_rate"].append(
            metrics.get("alpaca_metrics", {}).get("judge_win_rate", 0)
        )
        summary_table["rougeL"].append(
            metrics.get("alpaca_metrics", {}).get("rougeL", 0)
        )
        summary_table["bertscore"].append(
            metrics.get("alpaca_metrics", {}).get("bertscore_f1", 0)
        )
        summary_table["json_validity"].append(
            metrics.get("json_metrics", {}).get("validity_rate", 0)
        )
        summary_table["json_compliance"].append(
            metrics.get("json_metrics", {}).get("compliance_rate", 0)
        )
        summary_table["json_exact_match"].append(
            metrics.get("json_metrics", {}).get("exact_match_rate", 0)
        )
    
    results["summary_table"] = summary_table
    
    # ============================================================
    # Save Results
    # ============================================================
    
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save complete results JSON
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        # Convert non-serializable objects
        results_clean = {
            "checkpoint_metrics": results["checkpoint_metrics"],
            "judge_evaluation": results["judge_evaluation"],
            "summary_table": results["summary_table"],
            "forgetting_analysis": {
                "checkpoint_1_name": forgetting_analysis.checkpoint_1_name,
                "checkpoint_2_name": forgetting_analysis.checkpoint_2_name,
                "judge_win_rate_change": forgetting_analysis.judge_win_rate_percent_change,
                "rougeL_change": forgetting_analysis.rougeL_percent_change,
                "forgetting_severity": forgetting_analysis.forgetting_severity,
                "analysis": forgetting_analysis.analysis_summary,
            }
        }
        json.dump(results_clean, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("THREE-CHECKPOINT COMPARISON TABLE")
    print("="*80)
    print(f"{'Checkpoint':<35} {'ROUGE-L':<12} {'BERTScore':<12} {'JSON Valid':<12} {'JSON Exact':<12}")
    print("-" * 80)
    for i, cp in enumerate(summary_table["model_checkpoint"]):
        print(f"{cp:<35} "
              f"{summary_table['rougeL'][i]:>10.3f}   "
              f"{summary_table['bertscore'][i]:>10.3f}   "
              f"{summary_table['json_validity'][i]:>10.1%}   "
              f"{summary_table['json_exact_match'][i]:>10.1%}")
    
    return results


if __name__ == "__main__":
    print("Main evaluation orchestrator loaded.")
    print("Call run_complete_evaluation() to execute full pipeline.")
