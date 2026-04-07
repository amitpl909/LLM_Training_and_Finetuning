"""
Judge Evaluation Module - LLM-as-a-Judge Pairwise Comparison

Implements the Alpaca Self-Instruct evaluation protocol:
- Pairwise comparison between checkpoints
- Scoring on 6 dimensions (instruction following, correctness, clarity, completeness, structured output validity, hallucination risk)
- Following the Taori et al. (2023) methodology
"""

import json
import random
from typing import Dict, List, Tuple, Any
from openai import OpenAI
import yaml

# Load config to get judge model settings
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


# Judge prompt template following Taori et al. (2023) Self-Instruct protocol
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of LLM responses. Your task is to compare two responses to the same instruction and score them on multiple dimensions. You must be fair, objective, and consistent.

Scoring guidelines:
- Instruction Following (1-5): Does the response address the instruction? Does it follow constraints?
- Correctness (1-5): Is the information accurate? Are there factual errors?  
- Clarity (1-5): Is the response well-organized and easy to understand?
- Completeness (1-5): Does the response fully address the question or task?
- Structured Output Validity (1-5): For JSON/formatted tasks, is the structure valid and well-formed?
- Hallucination Risk (1-5): Are there fabricated facts or unsupported claims? Higher is better (lower risk).

If you cannot meaningfully score a dimension for the task type, use 3 as neutral."""

JUDGE_USER_PROMPT_TEMPLATE = """Please compare the following two responses to this instruction:

### INSTRUCTION:
{instruction}

### RESPONSE A:
{response_a}

### RESPONSE B:
{response_b}

Please evaluate both responses and provide:
1. Detailed scores for Response A on each dimension (1-5 scale)
2. Detailed scores for Response B on each dimension (1-5 scale)
3. Which response is better overall (A, B, or TIE)
4. Brief justification for your decision

Respond in the following JSON format:
{{
  "response_a_scores": {{
    "instruction_following": <1-5>,
    "correctness": <1-5>,
    "clarity": <1-5>,
    "completeness": <1-5>,
    "structured_output_validity": <1-5>,
    "hallucination_risk": <1-5>
  }},
  "response_b_scores": {{
    "instruction_following": <1-5>,
    "correctness": <1-5>,
    "clarity": <1-5>,
    "completeness": <1-5>,
    "structured_output_validity": <1-5>,
    "hallucination_risk": <1-5>
  }},
  "winner": "A" or "B" or "TIE",
  "justification": "<brief explanation>"
}}"""


class JudgeEvaluator:
    """LLM-as-a-Judge evaluator for pairwise comparison."""
    
    def __init__(self):
        """Initialize judge model client."""
        self.client = OpenAI(
            api_key=config["judge_api_key"],
            base_url=config["judge_api_url"]
        )
        self.model = config["judge_model"]
        self.evaluation_results = []
    
    def evaluate_pair(
        self,
        instruction: str,
        response_a: str,
        response_b: str,
        randomize_order: bool = True
    ) -> Dict[str, Any]:
        """
        Compare two responses using the judge model.
        
        Args:
            instruction: The task instruction
            response_a: First response (typically Checkpoint 1)
            response_b: Second response (typically Checkpoint 2)
            randomize_order: Randomly swap A/B to reduce ordering bias
        
        Returns:
            Judge evaluation with scores and winner
        """
        # Randomize order to reduce bias
        swap = randomize_order and random.random() < 0.5
        if swap:
            response_a, response_b = response_b, response_a
            label_a, label_b = "B", "A"
        else:
            label_a, label_b = "A", "B"
        
        user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
            instruction=instruction,
            response_a=response_a,
            response_b=response_b
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # Deterministic evaluation
                max_tokens=1024
            )
            
            result_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                result_json = json.loads(result_text)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                start = result_text.find('{')
                end = result_text.rfind('}') + 1
                if start >= 0 and end > start:
                    result_json = json.loads(result_text[start:end])
                else:
                    return {"error": "Could not parse judge response", "raw": result_text}
            
            # Map back to original labels if order was swapped
            if swap:
                result_json["response_a_scores"], result_json["response_b_scores"] = \
                    result_json["response_b_scores"], result_json["response_a_scores"]
                if result_json.get("winner") == "A":
                    result_json["winner"] = "B"
                elif result_json.get("winner") == "B":
                    result_json["winner"] = "A"
            
            return result_json
            
        except Exception as e:
            return {"error": str(e)}
    
    def evaluate_dataset(
        self,
        eval_pairs: List[Tuple[str, str, str]],  # List of (instruction, response_a, response_b)
        randomize_order: bool = True,
        sampling_rate: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a full dataset of instruction pairs.
        
        Args:
            eval_pairs: List of (instruction, response_a, response_b)
            randomize_order: Randomly swap A/B for each pair
            sampling_rate: Evaluate only this fraction of pairs (for cost control)
        
        Returns:
            List of evaluation results
        """
        results = []
        
        # Sample if requested
        if sampling_rate < 1.0:
            sample_size = max(1, int(len(eval_pairs) * sampling_rate))
            eval_pairs = random.sample(eval_pairs, sample_size)
        
        for i, (instruction, response_a, response_b) in enumerate(eval_pairs):
            print(f"  Evaluating pair {i+1}/{len(eval_pairs)}...", end="\r")
            result = self.evaluate_pair(instruction, response_a, response_b, randomize_order)
            result["pair_id"] = i
            result["instruction"] = instruction
            results.append(result)
        
        print(f"  Evaluated {len(results)} pairs                    ")
        self.evaluation_results.extend(results)
        return results
    
    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate judge scores across all pairs.
        
        Returns: Summary statistics
        """
        if not results:
            return {}
        
        valid_results = [r for r in results if "error" not in r]
        if not valid_results:
            return {"error": "All evaluations had errors"}
        
        wins = {"A": 0, "B": 0, "TIE": 0}
        dimension_scores_a = {
            "instruction_following": [],
            "correctness": [],
            "clarity": [],
            "completeness": [],
            "structured_output_validity": [],
            "hallucination_risk": []
        }
        dimension_scores_b = {
            "instruction_following": [],
            "correctness": [],
            "clarity": [],
            "completeness": [],
            "structured_output_validity": [],
            "hallucination_risk": []
        }
        
        for result in valid_results:
            winner = result.get("winner", "TIE")
            wins[winner] += 1
            
            for dim in dimension_scores_a.keys():
                score_a = result.get("response_a_scores", {}).get(dim)
                score_b = result.get("response_b_scores", {}).get(dim)
                
                if isinstance(score_a, (int, float)):
                    dimension_scores_a[dim].append(score_a)
                if isinstance(score_b, (int, float)):
                    dimension_scores_b[dim].append(score_b)
        
        # Compute statistics
        total = sum(wins.values())
        summary = {
            "total_pairs": len(valid_results),
            "error_count": len(results) - len(valid_results),
            "win_rates": {
                "response_a": wins["A"] / total if total > 0 else 0,
                "response_b": wins["B"] / total if total > 0 else 0,
                "tie": wins["TIE"] / total if total > 0 else 0,
            },
            "dimension_averages": {},
        }
        
        for dim in dimension_scores_a.keys():
            scores_a = dimension_scores_a[dim]
            scores_b = dimension_scores_b[dim]
            
            summary["dimension_averages"][dim] = {
                "response_a_mean": sum(scores_a) / len(scores_a) if scores_a else 0,
                "response_b_mean": sum(scores_b) / len(scores_b) if scores_b else 0,
            }
        
        return summary


def evaluate_checkpoints_pairwise(
    checkpoint_1_results: List[Dict[str, str]],  # {instruction, response}
    checkpoint_2_results: List[Dict[str, str]],  # {instruction, response}
    checkpoint_1_name: str = "Checkpoint 1",
    checkpoint_2_name: str = "Checkpoint 2",
    sample_size: int = None
) -> Dict[str, Any]:
    """
    Evaluate two checkpoints in pairwise comparison using judge model.
    
    Args:
        checkpoint_1_results: Results from first checkpoint
        checkpoint_2_results: Results from second checkpoint
        checkpoint_1_name: Label for first checkpoint
        checkpoint_2_name: Label for second checkpoint
        sample_size: If set, evaluate only this many pairs (for cost control)
    
    Returns:
        Comprehensive judge evaluation results
    """
    print(f"\nInitializing judge evaluator...")
    judge = JudgeEvaluator()
    
    # Prepare pairs
    eval_pairs = []
    for r1, r2 in zip(checkpoint_1_results, checkpoint_2_results):
        eval_pairs.append((
            r1["instruction"],
            r1["response"],
            r2["response"]
        ))
    
    # Optionally sample
    if sample_size and len(eval_pairs) > sample_size:
        eval_pairs = random.sample(eval_pairs, sample_size)
    
    print(f"Evaluating {len(eval_pairs)} pairs with judge model...")
    results = judge.evaluate_dataset(eval_pairs, randomize_order=True)
    
    print(f"Aggregating results...")
    summary = judge.aggregate_results(results)
    summary["checkpoint_1"] = checkpoint_1_name
    summary["checkpoint_2"] = checkpoint_2_name
    summary["raw_results"] = results
    
    return summary


if __name__ == "__main__":
    print("Judge evaluation module loaded.")
