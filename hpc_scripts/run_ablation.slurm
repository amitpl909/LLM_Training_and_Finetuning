#!/bin/bash
#SBATCH --job-name=ablation_epochs
#SBATCH --partition=gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --output=logs/ablation_%j.log
#SBATCH --error=logs/ablation_%j.err

echo "=========================================="
echo "ABLATION STUDY: Stage 2 Epochs Variation"
echo "=========================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

cd /work/nbe841/LLM_Training_and_Finetuning

# Load environment
module load anaconda3
source activate llm_env

echo "Python version: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Run ablation variants sequentially
VARIANTS=(1 2 3)

for EPOCHS in "${VARIANTS[@]}"; do
    echo ""
    echo "============================================"
    echo "Running: Stage 2 with epochs=$EPOCHS"
    echo "============================================"
    echo "Start time: $(date)"
    
    python training/ablation_stage2_epochs.py \
        --epochs "$EPOCHS" \
        --learning_rate 2e-4 \
        --dataset_fraction 1.0 \
        --output_suffix ""
    
    if [ $? -eq 0 ]; then
        echo "✅ Completed: epochs=$EPOCHS"
    else
        echo "❌ Failed: epochs=$EPOCHS"
        exit 1
    fi
    
    echo "End time: $(date)"
    echo ""
done

echo "============================================"
echo "ABLATION STUDY COMPLETE"
echo "============================================"
echo "End time: $(date)"
echo ""

# Extract summary
echo "Generating ablation summary..."
python3 << 'PYSCRIPT'
import json
from pathlib import Path

result_files = Path("results").glob("ablation_epochs*_metadata.json")
results = []

for f in sorted(result_files):
    with open(f) as fp:
        result = json.load(fp)
        results.append(result)
        print(f"\n📊 Epochs={result['epochs']}")
        print(f"   Training loss: {result['training_loss']:.4f}")
        print(f"   Eval loss: {result['eval_loss']:.4f}")
        print(f"   Output: {result['output_dir']}")

if results:
    print(f"\n✅ Ablation complete with {len(results)} variants")
PYSCRIPT

echo ""
echo "Ablation job completed successfully"
