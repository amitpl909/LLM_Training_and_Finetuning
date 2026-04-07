#!/bin/bash
#SBATCH --job-name=llm_inference
#SBATCH --output=logs/inference_%j.log
#SBATCH --error=logs/inference_%j.err
#SBATCH --partition=gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00

# Exit immediately if a command exits with a non-zero status
set -e

echo "Loading environment..."
module load anaconda3
source activate llm_env

echo "Starting Inference Generation..."
python evaluation/inference.py

echo "Inference Complete. Check results/inference_results.json"