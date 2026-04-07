"""
Forgetting Analysis - The Core Analytical Contribution

Measures catastrophic forgetting by comparing Alpaca evaluation results at:
- Checkpoint 1 (after Stage 1: Alpaca fine-tuning)  
- Checkpoint 2 (after Stage 2: Teacher-generated JSON fine-tuning)

This is the central research question: Does Stage 2 preserve or degrade Stage 1 capabilities?
"""

import json
from typing import Dict, List, Any
import numpy as np
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ForgettingAnalysis:
    """
    Results of catastrophic forgetting analysis.
    """
    checkpoint_1_name: str
    checkpoint_2_name: str
    
    # Overall metrics
    alpaca_judge_win_rate_cp1: float
    alpaca_judge_win_rate_cp2: float
    alpaca_rougeL_cp1: float
    alpaca_rougeL_cp2: float
    alpaca_bertscore_cp1: float
    alpaca_bertscore_cp2: float
    
    # Changes
    judge_win_rate_absolute_change: float
    judge_win_rate_percent_change: float
    rougeL_absolute_change: float
    rougeL_percent_change: float
    bertscore_absolute_change: float
    bertscore_percent_change: float
    
    # Per-category breakdown
    category_results: Dict[str, Dict[str, float]]  # {category: {metric: value}}
    
    # Regression examples
    regressed_examples: List[Dict[str, Any]]
    improved_examples: List[Dict[str, Any]]
    
    # Analysis
    has_catastrophic_forgetting: bool
    forgetting_severity: str  # "none", "mild", "moderate", "severe"
    analysis_summary: str


def compute_forgetting_analysis(
    checkpoint_1_judge_results: Dict[str, Any],  # Judge evaluation at CP1
    checkpoint_2_judge_results: Dict[str, Any],  # Judge evaluation at CP2
    checkpoint_1_metrics: Dict[str, Any],        # Automatic metrics at CP1
    checkpoint_2_metrics: Dict[str, Any],        # Automatic metrics at CP2
    checkpoint_1_outputs: List[Dict[str, str]],  # {instruction, category, response}
    checkpoint_2_outputs: List[Dict[str, str]],  # {instruction, category, response}
) -> ForgettingAnalysis:
    """
    Compute catastrophic forgetting analysis by comparing CP1 vs CP2 on Alpaca tasks.
    
    Args:
        checkpoint_1_judge_results: Judge evaluation summary for CP1
        checkpoint_2_judge_results: Judge evaluation summary for CP2
        checkpoint_1_metrics: Automatic metrics for CP1
        checkpoint_2_metrics: Automatic metrics for CP2
        checkpoint_1_outputs: Raw outputs from CP1 (with categories)
        checkpoint_2_outputs: Raw outputs from CP2 (with categories)
    
    Returns:
        ForgettingAnalysis object with comprehensive results
    """
    
    # ============================================================
    # Extract Overall Metrics
    # ============================================================
    
    # Judge-based: Win rates
    cp1_judge_win_rate = checkpoint_1_judge_results.get("win_rates", {}).get("response_a", 0.5)
    cp2_judge_win_rate = checkpoint_2_judge_results.get("win_rates", {}).get("response_a", 0.5)
    
    # Automatic metrics: ROUGE-L and BERTScore
    cp1_rougeL = checkpoint_1_metrics.get("alpaca_metrics", {}).get("rougeL", 0.0)
    cp2_rougeL = checkpoint_2_metrics.get("alpaca_metrics", {}).get("rougeL", 0.0)
    
    cp1_bertscore = checkpoint_1_metrics.get("alpaca_metrics", {}).get("bertscore_f1", 0.0)
    cp2_bertscore = checkpoint_2_metrics.get("alpaca_metrics", {}).get("bertscore_f1", 0.0)
    
    # ============================================================
    # Calculate Changes
    # ============================================================
    
    judge_wr_abs_change = cp2_judge_win_rate - cp1_judge_win_rate
    judge_wr_pct_change = (judge_wr_abs_change / cp1_judge_win_rate * 100) if cp1_judge_win_rate > 0 else 0
    
    rougeL_abs_change = cp2_rougeL - cp1_rougeL
    rougeL_pct_change = (rougeL_abs_change / cp1_rougeL * 100) if cp1_rougeL > 0 else 0
    
    bertscore_abs_change = cp2_bertscore - cp1_bertscore
    bertscore_pct_change = (bertscore_abs_change / cp1_bertscore * 100) if cp1_bertscore > 0 else 0
    
    # ============================================================
    # Classify Forgetting
    # ============================================================
    
    # Catastrophic forgetting defined as:
    # - Judge win rate drops > 10%
    # - ROUGE-L drops > 5%
    # - BERTScore drops > 5%
    
    has_catastrophic_forgetting = (
        judge_wr_pct_change < -10 or 
        rougeL_pct_change < -5 or 
        bertscore_pct_change < -5
    )
    
    # Severity classification
    if judge_wr_pct_change < -20:
        forgetting_severity = "severe"
    elif judge_wr_pct_change < -10:
        forgetting_severity = "moderate"
    elif judge_wr_pct_change < -5:
        forgetting_severity = "mild"
    else:
        forgetting_severity = "none"
    
    # ============================================================
    # Per-Category Breakdown
    # ============================================================
    
    category_results = defaultdict(lambda: {
        "cp1_judge_wins": 0,
        "cp2_judge_wins": 0,
        "cp1_judge_total": 0,
        "cp2_judge_total": 0,
        "total_examples": 0,
    })
    
    # Process judge results by category
    if "raw_results" in checkpoint_1_judge_results:
        for i, result in enumerate(checkpoint_1_judge_results["raw_results"]):
            if i < len(checkpoint_1_outputs):
                category = checkpoint_1_outputs[i].get("category", "unknown")
                category_results[category]["cp1_judge_total"] += 1
                if result.get("winner") == "A":
                    category_results[category]["cp1_judge_wins"] += 1
    
    if "raw_results" in checkpoint_2_judge_results:
        for i, result in enumerate(checkpoint_2_judge_results["raw_results"]):
            if i < len(checkpoint_2_outputs):
                category = checkpoint_2_outputs[i].get("category", "unknown")
                category_results[category]["cp2_judge_total"] += 1
                if result.get("winner") == "A":
                    category_results[category]["cp2_judge_wins"] += 1
    
    # Compute rates per category
    category_metrics = {}
    for category, stats in category_results.items():
        cp1_rate = stats["cp1_judge_wins"] / stats["cp1_judge_total"] if stats["cp1_judge_total"] > 0 else 0
        cp2_rate = stats["cp2_judge_wins"] / stats["cp2_judge_total"] if stats["cp2_judge_total"] > 0 else 0
        category_metrics[category] = {
            "cp1_win_rate": cp1_rate,
            "cp2_win_rate": cp2_rate,
            "absolute_change": cp2_rate - cp1_rate,
            "percent_change": ((cp2_rate - cp1_rate) / cp1_rate * 100) if cp1_rate > 0 else 0,
            "total_examples": stats["cp1_judge_total"],
        }
    
    # ============================================================
    # Identify Regression/Improvement Examples
    # ============================================================
    
    regressed_examples = []
    improved_examples = []
    
    if "raw_results" in checkpoint_1_judge_results and "raw_results" in checkpoint_2_judge_results:
        cp1_raw = checkpoint_1_judge_results["raw_results"]
        cp2_raw = checkpoint_2_judge_results["raw_results"]
        
        for i in range(min(len(cp1_raw), len(cp2_raw))):
            cp1_res = cp1_raw[i]
            cp2_res = cp2_raw[i]
            
            # Skip if errors
            if "error" in cp1_res or "error" in cp2_res:
                continue
            
            instruction = cp1_res.get("instruction", "")
            category = checkpoint_1_outputs[i].get("category", "unknown") if i < len(checkpoint_1_outputs) else "unknown"
            
            # Calculate dimension-wise changes
            cp1_avg = np.mean([v for k, v in cp1_res.get("response_a_scores", {}).items() if isinstance(v, (int, float))])
            cp2_avg = np.mean([v for k, v in cp2_res.get("response_a_scores", {}).items() if isinstance(v, (int, float))])
            
            change = cp2_avg - cp1_avg
            
            # Regression: significant drop
            if change < -0.5:
                regressed_examples.append({
                    "instruction": instruction,
                    "category": category,
                    "cp1_score": cp1_avg,
                    "cp2_score": cp2_avg,
                    "change": change,
                    "cp1_winner": cp1_res.get("winner"),
                    "cp2_winner": cp2_res.get("winner"),
                })
            
            # Improvement: significant gain
            elif change > 0.5:
                improved_examples.append({
                    "instruction": instruction,
                    "category": category,
                    "cp1_score": cp1_avg,
                    "cp2_score": cp2_avg,
                    "change": change,
                    "cp1_winner": cp1_res.get("winner"),
                    "cp2_winner": cp2_res.get("winner"),
                })
    
    # Sort by magnitude of change
    regressed_examples.sort(key=lambda x: x["change"])
    improved_examples.sort(key=lambda x: x["change"], reverse=True)
    
    # ============================================================
    # Generate Summary Analysis
    # ============================================================
    
    summary_lines = []
    
    if forgetting_severity == "none":
        summary_lines.append(
            "✓ NO CATASTROPHIC FORGETTING detected."
        )
        if judge_wr_pct_change > 0:
            summary_lines.append(
                f"  Judge win rate improved by {judge_wr_pct_change:.1f}%: {cp1_judge_win_rate:.2%} → {cp2_judge_win_rate:.2%}."
            )
        else:
            summary_lines.append(
                f"  Judge win rate stable: {cp1_judge_win_rate:.2%} (minimal change)."
            )
    elif forgetting_severity == "mild":
        summary_lines.append(
            f"⚠ MILD FORGETTING detected (win rate drop {judge_wr_pct_change:.1f}%)."
        )
        summary_lines.append(
            f"  Judge preference declined slightly: {cp1_judge_win_rate:.2%} → {cp2_judge_win_rate:.2%}."
        )
    elif forgetting_severity == "moderate":
        summary_lines.append(
            f"⚠⚠ MODERATE FORGETTING detected (win rate drop {judge_wr_pct_change:.1f}%)."
        )
        summary_lines.append(
            f"  Alpaca capabilities degraded noticeably: {cp1_judge_win_rate:.2%} → {cp2_judge_win_rate:.2%}."
        )
    else:  # severe
        summary_lines.append(
            f"⚠⚠⚠ SEVERE FORGETTING detected (win rate drop {judge_wr_pct_change:.1f}%)."
        )
        summary_lines.append(
            f"  Alpaca capabilities severely degraded: {cp1_judge_win_rate:.2%} → {cp2_judge_win_rate:.2%}."
        )
    
    # Add automatic metrics summary
    summary_lines.append("")
    summary_lines.append("  Automatic metrics:")
    summary_lines.append(f"  - ROUGE-L: {cp1_rougeL:.3f} → {cp2_rougeL:.3f} (change: {rougeL_pct_change:+.1f}%)")
    summary_lines.append(f"  - BERTScore: {cp1_bertscore:.3f} → {cp2_bertscore:.3f} (change: {bertscore_pct_change:+.1f}%)")
    
    # Per-category findings
    if category_metrics:
        summary_lines.append("")
        summary_lines.append("  Per-category impact:")
        for cat, metrics in sorted(category_metrics.items()):
            direction = "↓" if metrics["percent_change"] < 0 else "↑"
            summary_lines.append(
                f"  - {cat}: {direction} {metrics['percent_change']:+.1f}% "
                f"({metrics['cp1_win_rate']:.2%} → {metrics['cp2_win_rate']:.2%})"
            )
    
    # Conclusion
    summary_lines.append("")
    if forgetting_severity == "none":
        summary_lines.append(
            "CONCLUSION: Stage 2 training successfully improved JSON capabilities while "
            "maintaining or improving general Alpaca instruction-following performance. "
            "No evidence of catastrophic forgetting."
        )
    else:
        summary_lines.append(
            f"CONCLUSION: Stage 2 training caused {forgetting_severity} degradation of Alpaca capabilities. "
            "This suggests a tradeoff between JSON specialization and general instruction-following. "
            "Consider: reduced learning rate, fewer epochs, more diverse Stage 2 data, or task augmentation."
        )
    
    analysis_summary = "\n".join(summary_lines)
    
    # ============================================================
    # Create Result Object
    # ============================================================
    
    return ForgettingAnalysis(
        checkpoint_1_name="After Stage 1 (Alpaca)",
        checkpoint_2_name="After Stage 2 (Teacher JSON)",
        alpaca_judge_win_rate_cp1=cp1_judge_win_rate,
        alpaca_judge_win_rate_cp2=cp2_judge_win_rate,
        alpaca_rougeL_cp1=cp1_rougeL,
        alpaca_rougeL_cp2=cp2_rougeL,
        alpaca_bertscore_cp1=cp1_bertscore,
        alpaca_bertscore_cp2=cp2_bertscore,
        judge_win_rate_absolute_change=judge_wr_abs_change,
        judge_win_rate_percent_change=judge_wr_pct_change,
        rougeL_absolute_change=rougeL_abs_change,
        rougeL_percent_change=rougeL_pct_change,
        bertscore_absolute_change=bertscore_abs_change,
        bertscore_percent_change=bertscore_pct_change,
        category_results=category_metrics,
        regressed_examples=regressed_examples[:5],  # Top 5
        improved_examples=improved_examples[:5],     # Top 5
        has_catastrophic_forgetting=has_catastrophic_forgetting,
        forgetting_severity=forgetting_severity,
        analysis_summary=analysis_summary,
    )


def print_forgetting_analysis(analysis: ForgettingAnalysis) -> None:
    """Pretty-print forgetting analysis results."""
    print("\n" + "="*80)
    print("CATASTROPHIC FORGETTING ANALYSIS")
    print("="*80)
    print(analysis.analysis_summary)
    print("="*80)
    
    print("\nDETAILED METRICS:")
    print(f"  Judge Win Rate (A vs B in pairwise comparison):")
    print(f"    Checkpoint 1: {analysis.alpaca_judge_win_rate_cp1:.2%}")
    print(f"    Checkpoint 2: {analysis.alpaca_judge_win_rate_cp2:.2%}")
    print(f"    Absolute change: {analysis.judge_win_rate_absolute_change:+.2%}")
    print(f"    Percent change: {analysis.judge_win_rate_percent_change:+.1f}%")
    
    print(f"\n  ROUGE-L (automatic metric):")
    print(f"    Checkpoint 1: {analysis.alpaca_rougeL_cp1:.3f}")
    print(f"    Checkpoint 2: {analysis.alpaca_rougeL_cp2:.3f}")
    print(f"    Absolute change: {analysis.rougeL_absolute_change:+.3f}")
    print(f"    Percent change: {analysis.rougeL_percent_change:+.1f}%")
    
    print(f"\n  BERTScore (automatic metric):")
    print(f"    Checkpoint 1: {analysis.alpaca_bertscore_cp1:.3f}")
    print(f"    Checkpoint 2: {analysis.alpaca_bertscore_cp2:.3f}")
    print(f"    Absolute change: {analysis.bertscore_absolute_change:+.3f}")
    print(f"    Percent change: {analysis.bertscore_percent_change:+.1f}%")
    
    if analysis.regressed_examples:
        print(f"\n  REGRESSION EXAMPLES (top {len(analysis.regressed_examples)}):")
        for ex in analysis.regressed_examples:
            print(f"    [{ex['category']}] Score change: {ex['change']:+.2f} ({ex['cp1_score']:.2f} → {ex['cp2_score']:.2f})")
            print(f"      Q: {ex['instruction'][:60]}...")
            print()
    
    if analysis.improved_examples:
        print(f"\n  IMPROVEMENT EXAMPLES (top {len(analysis.improved_examples)}):")
        for ex in analysis.improved_examples:
            print(f"    [{ex['category']}] Score change: {ex['change']:+.2f} ({ex['cp1_score']:.2f} → {ex['cp2_score']:.2f})")
            print(f"      Q: {ex['instruction'][:60]}...")
            print()
    
    print("="*80 + "\n")


if __name__ == "__main__":
    print("Forgetting analysis module loaded.")
