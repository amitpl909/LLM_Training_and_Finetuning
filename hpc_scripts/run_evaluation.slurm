#!/bin/bash
#SBATCH --job-name=eval_pipeline
#SBATCH --output=logs/eval_%j.log
#SBATCH --error=logs/eval_%j.err
#SBATCH --partition=gpu1v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00

# Load modules
module load anaconda3
module load gcc

# Activate environment
conda activate llm_env

# Navigate to project
cd /work/nbe841/LLM_Training_and_Finetuning

# Run the full evaluation pipeline
echo "====== Starting Evaluation Pipeline ======"
date

echo ""
echo "--- Step 1: Generating Inference Outputs ---"
python evaluation/inference.py
if [ $? -ne 0 ]; then
    echo "ERROR: Inference failed!"
    exit 1
fi

echo ""
echo "--- Step 2: Running Comprehensive Judge Evaluation ---"
python evaluation/llm_judge_v2.py
if [ $? -ne 0 ]; then
    echo "ERROR: Judge evaluation failed!"
    exit 1
fi

echo ""
echo "--- Step 3: Running Forgetting Analysis ---"
python evaluation/forgetting_analysis.py
if [ $? -ne 0 ]; then
    echo "ERROR: Forgetting analysis failed!"
    exit 1
fi

echo ""
echo "====== Evaluation Pipeline Complete ======"
date
echo "Results saved to results/"
ls -lh results/
