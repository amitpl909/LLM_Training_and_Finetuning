#!/bin/bash
#SBATCH --job-name=llm_finetuning
#SBATCH --output=logs/training_%j.log
#SBATCH --error=logs/training_%j.err
#SBATCH --partition=gpu1v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu004,gpu013,gpu014
#SBATCH --time=12:00:00

# Note: Using explicit exclude list instead of nodelist for flexibility.
#       If scheduling fails, try: --nodelist=gpu001,gpu002,gpu003,gpu005,gpu006,gpu007,gpu008,gpu009,gpu010,gpu011,gpu012,gpu014,gpu015,gpu016,gpu017,gpu018,gpu019,gpu020,gpu021,gpu022

module load anaconda3
source activate llm_env

echo "=== HPC Training Job Starting ==="
echo "Kernel version check:"
uname -r
echo ""

echo "Starting Phase 1: Alpaca Tuning"
python training/stage1_alpaca.py
STAGE1_EXIT=$?

if [ $STAGE1_EXIT -ne 0 ]; then
    echo "❌ Stage 1 failed with exit code $STAGE1_EXIT"
    exit $STAGE1_EXIT
fi

echo ""
echo "Starting Phase 2: JSON Instruct Tuning"
python training/stage2_json.py
STAGE2_EXIT=$?

if [ $STAGE2_EXIT -ne 0 ]; then
    echo "❌ Stage 2 failed with exit code $STAGE2_EXIT"
    exit $STAGE2_EXIT
fi

echo ""
echo "✅ All training complete."
