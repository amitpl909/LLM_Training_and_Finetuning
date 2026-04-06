#!/bin/bash
#SBATCH --job-name=env_setup
#SBATCH --output=logs/install_%j.log
#SBATCH --error=logs/install_%j.err
#SBATCH --partition=gpu1v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

# Load Anaconda
module load anaconda3

# Create and setup the environment
# Using --prefix or a specific name to ensure it's in your work directory
conda create -n llm_env python=3.10 -y
source activate llm_env

# Install the stack
pip install torch transformers datasets peft trl accelerate bitsandbytes pyyaml tqdm openai

echo "Environment llm_env is ready."