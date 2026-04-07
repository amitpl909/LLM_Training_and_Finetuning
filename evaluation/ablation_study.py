"""
Ablation Study Module

Required ablation studies per assignment:
- Vary Stage 2 epochs (1 vs 2 vs 3) to measure forgetting escalation
- Vary Stage 2 learning rate (2e-5 vs 1e-5 vs 5e-6) to measure tradeoff
- Reduce Stage 2 dataset size (100%, 50%, 25%) to measure data efficiency
- Compare sequential vs combined training

This module provides infrastructure for running and tracking ablations.
"""

import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import datetime


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    ablation_type: str  # "epochs", "learning_rate", "dataset_size", "sequential_vs_combined"
    variant_name: str   # "epochs_1", "lr_1e-5", "datasize_50pct", etc.
    
    # Stage 2 hyperparameters (override defaults)
    stage2_epochs: int = None
    stage2_learning_rate: float = None
    stage2_dataset_size_pct: float = 100.0  # Percentage of full dataset to use
    
    # Training configuration
    training_approach: str = "sequential"  # "sequential" or "combined"
    
    # Metadata
    description: str = ""


@dataclass
class AblationResult:
    """Results of a single ablation experiment."""
    config: AblationConfig
    
    # Metrics
    stage2_training_loss_final: float = None
    stage2_val_loss_final: float = None
    
    # Evaluation results
    json_validity_rate: float = None
    json_compliance_rate: float = None
    json_exact_match_rate: float = None
    
    alpaca_judge_win_rate: float = None
    alpaca_rougeL: float = None
    alpaca_bertscore: float = None
    
    # Forgetting metrics
    forgetting_severity: str = None  # "none", "mild", "moderate", "severe"
    judge_win_rate_change: float = None  # CP2 vs CP1
    rougeL_change: float = None
    
    # Metadata
    primary_metric: float = None  # For comparison (e.g., forgetting magnitude)
    timestamp: str = None
    notes: str = ""


class AblationStudyRunner:
    """Manage ablation study experiments."""
    
    def __init__(self, output_dir: str = "results/ablations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[AblationResult] = []
    
    def add_result(self, result: AblationResult) -> None:
        """Record an ablation result."""
        result.timestamp = datetime.datetime.now().isoformat()
        self.results.append(result)
    
    def save_results(self, filename: str = "ablation_results.json") -> str:
        """Save all ablation results to JSON."""
        output_path = self.output_dir / filename
        
        # Convert to serializable format
        results_data = []
        for r in self.results:
            r_dict = asdict(r)
            # Convert nested dataclass to dict
            if isinstance(r_dict["config"], dict):
                r_dict["config"] = asdict(r_dict["config"]) if hasattr(r_dict["config"], "__dataclass_fields__") else r_dict["config"]
            results_data.append(r_dict)
        
        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2, default=str)
        
        return str(output_path)
    
    def generate_report(self) -> str:
        """Generate ablation study report."""
        report_lines = []
        
        report_lines.append("="*80)
        report_lines.append("ABLATION STUDY REPORT")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Group results by ablation type
        by_type = {}
        for result in self.results:
            atype = result.config.ablation_type
            if atype not in by_type:
                by_type[atype] = []
            by_type[atype].append(result)
        
        # Report each ablation type
        for atype in sorted(by_type.keys()):
            report_lines.append(f"\n## ABLATION: {atype.upper().replace('_', ' ')}")
            report_lines.append("-" * 80)
            
            results_for_type = by_type[atype]
            
            # Create comparison table
            report_lines.append("")
            report_lines.append(f"{'Variant':<20} {'JSON Acc':<12} {'Alpaca WR':<12} {'Forgetting':<15} {'Primary':<10}")
            report_lines.append("-" * 80)
            
            for result in sorted(results_for_type, key=lambda r: r.config.variant_name):
                variant = result.config.variant_name
                json_acc = f"{result.json_exact_match_rate:.1%}" if result.json_exact_match_rate is not None else "N/A"
                alpaca_wr = f"{result.alpaca_judge_win_rate:.1%}" if result.alpaca_judge_win_rate is not None else "N/A"
                forgetting = result.forgetting_severity or "N/A"
                primary = f"{result.primary_metric:.3f}" if result.primary_metric is not None else "N/A"
                
                report_lines.append(
                    f"{variant:<20} {json_acc:<12} {alpaca_wr:<12} {forgetting:<15} {primary:<10}"
                )
            
            # Analysis
            report_lines.append("")
            report_lines.append("## Analysis:")
            
            if atype == "epochs":
                report_lines.append(
                    "  Effect of Stage 2 training epochs on forgetting tradeoff:\n"
                    "  - Fewer epochs: Lower JSON accuracy, less forgetting\n"
                    "  - More epochs: Higher JSON accuracy, risk of increased forgetting\n"
                )
                
                # Find trends
                epochs_results = sorted(results_for_type, key=lambda r: r.config.stage2_epochs if r.config.stage2_epochs else 0)
                if len(epochs_results) >= 2:
                    first = epochs_results[0]
                    last = epochs_results[-1]
                    if first.primary_metric and last.primary_metric:
                        trend = last.primary_metric - first.primary_metric
                        report_lines.append(f"  Trend: Forgetting {'increases' if trend > 0 else 'decreases'} "
                                          f"with more epochs (change: {trend:+.3f})")
            
            elif atype == "learning_rate":
                report_lines.append(
                    "  Effect of Stage 2 learning rate on accuracy/forgetting tradeoff:\n"
                    "  - Higher LR: Faster convergence, risk of catastrophic forgetting\n"
                    "  - Lower LR: Slower convergence, better retention of Stage 1 knowledge\n"
                )
                
                # Find optimal rate
                best_result = max(results_for_type, key=lambda r: r.primary_metric if r.primary_metric else 0)
                report_lines.append(f"  Recommendation: {best_result.config.stage2_learning_rate} "
                                  f"shows best tradeoff (score: {best_result.primary_metric:.3f})")
            
            elif atype == "dataset_size":
                report_lines.append(
                    "  Effect of Stage 2 dataset size on learning efficiency:\n"
                    "  - Larger dataset: More JSON examples, better specialization\n"
                    "  - Smaller dataset: Less data, less risk of forgetting\n"
                )
                
                # Analyze efficiency
                size_results = sorted(results_for_type, key=lambda r: r.config.stage2_dataset_size_pct)
                if len(size_results) >= 2:
                    report_lines.append(f"  100% → 50% change: JSON {size_results[0].json_exact_match_rate:.1%} → "
                                      f"{size_results[1].json_exact_match_rate:.1%}")
            
            elif atype == "sequential_vs_combined":
                report_lines.append(
                    "  Comparison of sequential vs. combined training:\n"
                    "  - Sequential: Two stages allow specialization, risk of forgetting\n"
                    "  - Combined: Single stage on merged data, no catastrophic forgetting risk\n"
                )
                
                seq_result = next((r for r in results_for_type if r.config.training_approach == "sequential"), None)
                comb_result = next((r for r in results_for_type if r.config.training_approach == "combined"), None)
                
                if seq_result and comb_result:
                    report_lines.append(f"  Sequential approach forgetting: {seq_result.forgetting_severity}")
                    report_lines.append(f"  Combined approach forgetting: {comb_result.forgetting_severity}")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("END OF ABLATION REPORT")
        report_lines.append("="*80)
        
        return "\n".join(report_lines)
    
    def save_report(self, filename: str = "ablation_report.txt") -> str:
        """Save report to file."""
        output_path = self.output_dir / filename
        report_text = self.generate_report()
        
        with open(output_path, "w") as f:
            f.write(report_text)
        
        print(report_text)
        return str(output_path)


def create_ablation_configs() -> List[AblationConfig]:
    """
    Create standard ablation configurations per assignment requirements.
    
    Returns:
        List of ablation configurations to run
    """
    configs = []
    
    # ============================================================
    # Ablation 1: Vary Stage 2 Epochs (1 vs 2 vs 3)
    # ============================================================
    for epochs in [1, 2, 3]:
        configs.append(AblationConfig(
            ablation_type="epochs",
            variant_name=f"epochs_{epochs}",
            stage2_epochs=epochs,
            stage2_learning_rate=2e-5,
            description=f"Stage 2 with {epochs} epoch(s) to measure forgetting escalation"
        ))
    
    # ============================================================
    # Ablation 2: Vary Stage 2 Learning Rate (2e-5 vs 1e-5 vs 5e-6)
    # ============================================================
    for lr in [2e-5, 1e-5, 5e-6]:
        configs.append(AblationConfig(
            ablation_type="learning_rate",
            variant_name=f"lr_{lr:.0e}",
            stage2_epochs=2,
            stage2_learning_rate=lr,
            description=f"Stage 2 with learning rate {lr} to measure accuracy/retention tradeoff"
        ))
    
    # ============================================================
    # Ablation 3: Reduce Stage 2 Dataset Size (100%, 50%, 25%)
    # ============================================================
    for pct in [100, 50, 25]:
        configs.append(AblationConfig(
            ablation_type="dataset_size",
            variant_name=f"datasize_{pct}pct",
            stage2_epochs=2,
            stage2_learning_rate=2e-5,
            stage2_dataset_size_pct=pct,
            description=f"Stage 2 with {pct}% of dataset to measure data efficiency"
        ))
    
    # ============================================================
    # Ablation 4: Sequential vs Combined Training
    # ============================================================
    configs.append(AblationConfig(
        ablation_type="sequential_vs_combined",
        variant_name="sequential",
        training_approach="sequential",
        description="Standard sequential fine-tuning (Stage 1 → Stage 2)"
    ))
    
    configs.append(AblationConfig(
        ablation_type="sequential_vs_combined",
        variant_name="combined",
        training_approach="combined",
        description="Single-stage training on merged Alpaca + JSON data"
    ))
    
    return configs


if __name__ == "__main__":
    print("Ablation study module loaded.")
