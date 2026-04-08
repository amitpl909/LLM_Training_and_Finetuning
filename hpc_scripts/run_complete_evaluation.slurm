#!/bin/bash
#SBATCH --job-name=complete_eval
#SBATCH --partition=gpu1v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/complete_eval_%j.log
#SBATCH --error=logs/complete_eval_%j.err

# Load modules
module load anaconda3
source activate llm_env

cd /work/nbe841/LLM_Training_and_Finetuning

echo "=========================================="
echo "COMPLETE EVALUATION PIPELINE"
echo "=========================================="
echo "Start time: $(date)"
echo

# Step 1: Run inference on all 3 checkpoints
echo "Step 1: Generating responses from all 3 checkpoints..."
python evaluation/inference_minimal.py 2>&1 | tail -100

if [ $? -ne 0 ]; then
    echo "ERROR: Inference failed. Attempting fallback inference_v2.py..."
    python evaluation/inference_v2.py 2>&1 | tail -100
fi

echo -e "\n✅ Inference complete"
echo

# Step 2: Run judge evaluation
echo "Step 2: Running judge evaluation on responses..."
python evaluation/llm_judge_v2.py 2>&1 | tail -50

if [ -f "results/judge_evaluation_complete.json" ]; then
    echo "✅ Judge evaluation complete"
    wc -l results/judge_evaluation_complete.json
else
    echo "⚠️ Judge results file not found"
fi
echo

# Step 3: Extract metrics and calculate forgetting
echo "Step 3: Extracting metrics and forgetting analysis..."
python evaluation/forgetting_analysis.py 2>&1 | tail -50

if [ -f "results/forgetting_metrics.json" ]; then
    echo "✅ Metrics extraction complete"
    cat results/forgetting_metrics.json | python -m json.tool | head -50
else
    echo "⚠️ Metrics file not created"
fi

echo
echo "=========================================="
echo "EVALUATION PIPELINE COMPLETE"
echo "End time: $(date)"
echo "=========================================="
