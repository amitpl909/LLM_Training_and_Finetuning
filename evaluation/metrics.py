"""
Automatic Metrics Computation for Evaluation

Computes metrics across all 3 checkpoints:
- JSON validity and schema compliance
- ROUGE-L and BERTScore for Alpaca tasks
- Exact match for structured tasks
- Per-category breakdown
"""

import json
import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    from bert_score import score as bert_score
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False


class AutomaticMetrics:
    """Compute JSON validity, schema compliance, and text similarity metrics."""
    
    def __init__(self):
        if HAS_ROUGE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        if HAS_BERTSCORE:
            pass  # BERTScore initialized per batch for efficiency
    
    def validate_json(self, text: str) -> Tuple[bool, str]:
        """
        Validate if text is valid JSON.
        Returns: (is_valid: bool, error_msg: str)
        """
        try:
            json.loads(text)
            return True, ""
        except json.JSONDecodeError as e:
            return False, str(e)
    
    def check_schema_compliance(self, json_obj: dict, schema: dict) -> Tuple[bool, List[str]]:
        """
        Check if JSON object matches required schema.
        Schema: {"required_fields": ["field1", "field2"], "types": {"field1": "string", ...}}
        Returns: (is_compliant: bool, errors: List[str])
        """
        errors = []
        
        if not isinstance(json_obj, dict):
            errors.append("Output is not a JSON object")
            return False, errors
        
        # Check required fields
        if "required_fields" in schema:
            for field in schema["required_fields"]:
                if field not in json_obj:
                    errors.append(f"Missing required field: {field}")
        
        # Check type compliance
        if "types" in schema:
            for field, expected_type in schema["types"].items():
                if field in json_obj:
                    value = json_obj[field]
                    if expected_type == "string" and not isinstance(value, str):
                        errors.append(f"Field '{field}' should be string, got {type(value).__name__}")
                    elif expected_type == "number" and not isinstance(value, (int, float)):
                        errors.append(f"Field '{field}' should be number, got {type(value).__name__}")
                    elif expected_type == "boolean" and not isinstance(value, bool):
                        errors.append(f"Field '{field}' should be boolean, got {type(value).__name__}")
                    elif expected_type == "array" and not isinstance(value, list):
                        errors.append(f"Field '{field}' should be array, got {type(value).__name__}")
        
        is_compliant = len(errors) == 0
        return is_compliant, errors
    
    def exact_match(self, predicted: str, reference: str) -> bool:
        """Exact string match (optionally with JSON normalization)."""
        try:
            # Normalize JSON for comparison
            pred_json = json.loads(predicted)
            ref_json = json.loads(reference)
            return json.dumps(pred_json, sort_keys=True) == json.dumps(ref_json, sort_keys=True)
        except:
            # Fall back to string comparison
            return predicted.strip() == reference.strip()
    
    def rouge_score(self, predicted: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE-1, ROUGE-2, ROUGE-L."""
        if not HAS_ROUGE:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        try:
            scores = self.rouge_scorer.score(reference, predicted)
            return {
                "rouge1": scores['rouge1'].fmeasure,
                "rouge2": scores['rouge2'].fmeasure,
                "rougeL": scores['rougeL'].fmeasure,
            }
        except:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    def bert_score(self, predicted_list: List[str], reference_list: List[str]) -> Dict[str, float]:
        """Compute BERTScore F1 across a list of predictions."""
        if not HAS_BERTSCORE or len(predicted_list) == 0:
            return {"bertscore_f1": 0.0}
        
        try:
            P, R, F1 = bert_score(predicted_list, reference_list, lang="en", verbose=False)
            return {"bertscore_f1": float(F1.mean())}
        except:
            return {"bertscore_f1": 0.0}
    
    def compute_json_error_taxonomy(self, outputs: List[str]) -> Dict[str, int]:
        """Categorize common JSON formatting errors."""
        errors = defaultdict(int)
        
        for output in outputs:
            output = output.strip()
            
            is_valid, error_msg = self.validate_json(output)
            if is_valid:
                continue
            
            # Categorize error
            if "Expecting value" in error_msg:
                errors["invalid_value"] += 1
            elif "Expecting property name" in error_msg or "Expecting ':'" in error_msg:
                errors["missing_colon_or_key"] += 1
            elif "Expecting ',' delimiter" in error_msg:
                errors["missing_comma"] += 1
            elif error_msg.startswith("Extra data"):
                errors["extra_data"] += 1
            elif "Unterminated string" in error_msg:
                errors["unterminated_string"] += 1
            else:
                # Analyze output structure
                if not output.startswith('{') and not output.startswith('['):
                    errors["not_json_format"] += 1
                elif output.count('{') != output.count('}'):
                    errors["unmatched_braces"] += 1
                elif output.count('[') != output.count(']'):
                    errors["unmatched_brackets"] += 1
                else:
                    errors["other_error"] += 1
        
        return dict(errors)


def compute_checkpoint_metrics(
    checkpoint_name: str,
    alpaca_results: List[Dict[str, str]],  # List of {instruction, response, expected_output}
    json_results: List[Dict[str, Any]],     # List of {instruction, response, expected_output, schema}
) -> Dict[str, Any]:
    """
    Compute all automatic metrics for a checkpoint.
    
    Args:
        checkpoint_name: "baseline", "phase1_alpaca", "phase2_json"
        alpaca_results: Alpaca evaluation results
        json_results: JSON evaluation results with schema info
    
    Returns:
        Comprehensive metrics dict
    """
    metrics = AutomaticMetrics()
    results = {
        "checkpoint": checkpoint_name,
        "alpaca_metrics": {},
        "json_metrics": {},
    }
    
    # ========== ALPACA METRICS ==========
    alpaca_predictions = [r["response"] for r in alpaca_results]
    alpaca_references = [r["expected_output"] for r in alpaca_results]
    
    results["alpaca_metrics"]["count"] = len(alpaca_results)
    results["alpaca_metrics"]["avg_response_length"] = np.mean([len(p.split()) for p in alpaca_predictions])
    
    # ROUGE scores
    rouge_scores = []
    for pred, ref in zip(alpaca_predictions, alpaca_references):
        rouge_scores.append(metrics.rouge_score(pred, ref))
    
    if rouge_scores:
        results["alpaca_metrics"]["rouge1"] = np.mean([s["rouge1"] for s in rouge_scores])
        results["alpaca_metrics"]["rouge2"] = np.mean([s["rouge2"] for s in rouge_scores])
        results["alpaca_metrics"]["rougeL"] = np.mean([s["rougeL"] for s in rouge_scores])
    
    # BERTScore
    bertscore_result = metrics.bert_score(alpaca_predictions, alpaca_references)
    results["alpaca_metrics"].update(bertscore_result)
    
    # ========== JSON METRICS ==========
    json_valid_count = 0
    json_compliant_count = 0
    json_exact_match_count = 0
    json_error_taxonomy = defaultdict(int)
    
    results["json_metrics"]["count"] = len(json_results)
    
    for result in json_results:
        response = result["response"]
        expected = result["expected_output"]
        schema = result.get("schema", {})
        
        # Validity
        is_valid, error_msg = metrics.validate_json(response)
        if is_valid:
            json_valid_count += 1
            
            try:
                json_obj = json.loads(response)
                
                # Compliance
                is_compliant, errors = metrics.check_schema_compliance(json_obj, schema)
                if is_compliant:
                    json_compliant_count += 1
                
                # Exact match
                if metrics.exact_match(response, expected):
                    json_exact_match_count += 1
            except:
                pass
        else:
            # Track error type
            if "Expecting value" in error_msg:
                json_error_taxonomy["invalid_value"] += 1
            elif "Expecting property name" in error_msg:
                json_error_taxonomy["missing_key"] += 1
            elif "Expecting ':'" in error_msg:
                json_error_taxonomy["missing_colon"] += 1
            elif "Expecting ',' delimiter" in error_msg:
                json_error_taxonomy["missing_comma"] += 1
            else:
                json_error_taxonomy["other_error"] += 1
    
    results["json_metrics"]["validity_rate"] = json_valid_count / len(json_results) if json_results else 0
    results["json_metrics"]["compliance_rate"] = json_compliant_count / len(json_results) if json_results else 0
    results["json_metrics"]["exact_match_rate"] = json_exact_match_count / len(json_results) if json_results else 0
    results["json_metrics"]["error_taxonomy"] = dict(json_error_taxonomy)
    
    return results


def compare_checkpoints(
    checkpoint1_metrics: Dict[str, Any],
    checkpoint2_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare metrics between two checkpoints to measure forgetting.
    
    Returns: {metric_name: (value_cp1, value_cp2, absolute_change, percent_change)}
    """
    comparison = {
        "checkpoint_1": checkpoint1_metrics["checkpoint"],
        "checkpoint_2": checkpoint2_metrics["checkpoint"],
        "alpaca_metrics": {},
        "json_metrics": {},
    }
    
    # Compare Alpaca metrics
    for metric_name in ["rouge1", "rouge2", "rougeL", "bertscore_f1"]:
        v1 = checkpoint1_metrics["alpaca_metrics"].get(metric_name, 0)
        v2 = checkpoint2_metrics["alpaca_metrics"].get(metric_name, 0)
        abs_change = v2 - v1
        pct_change = (abs_change / v1 * 100) if v1 > 0 else 0
        comparison["alpaca_metrics"][metric_name] = {
            "checkpoint_1": v1,
            "checkpoint_2": v2,
            "absolute_change": abs_change,
            "percent_change": pct_change,
        }
    
    # Compare JSON metrics
    for metric_name in ["validity_rate", "compliance_rate", "exact_match_rate"]:
        v1 = checkpoint1_metrics["json_metrics"].get(metric_name, 0)
        v2 = checkpoint2_metrics["json_metrics"].get(metric_name, 0)
        abs_change = v2 - v1
        pct_change = (abs_change / v1 * 100) if v1 > 0 else 0
        comparison["json_metrics"][metric_name] = {
            "checkpoint_1": v1,
            "checkpoint_2": v2,
            "absolute_change": abs_change,
            "percent_change": pct_change,
        }
    
    return comparison


if __name__ == "__main__":
    print("Metrics module loaded. Use compute_checkpoint_metrics() and compare_checkpoints().")
