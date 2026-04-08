#!/bin/bash
#
# monitor_ablation.sh
# Monitors ablation study progress and auto-triggers inference re-run when complete
#

ABLATION_JOB_ID=$1
WORK_DIR="/work/nbe841/LLM_Training_and_Finetuning"
CHECK_INTERVAL=30  # seconds

if [ -z "$ABLATION_JOB_ID" ]; then
    echo "Usage: $0 <JOB_ID>"
    echo "Example: $0 724497"
    exit 1
fi

echo "=========================================="
echo "ABLATION STUDY MONITOR"
echo "=========================================="
echo "Monitoring Job: $ABLATION_JOB_ID"
echo "Work directory: $WORK_DIR"
echo "Check interval: $CHECK_INTERVAL seconds"
echo "=========================================="
echo ""

cd "$WORK_DIR"

check_ablation_status() {
    # Check if job still running
    if squeue -j "$ABLATION_JOB_ID" | grep -q "$ABLATION_JOB_ID"; then
        # Still running
        local elapsed=$(squeue -j "$ABLATION_JOB_ID" -o "%M" | tail -1)
        echo "[$(date '+%H:%M:%S')] Job still running (elapsed: $elapsed)"
        
        # Show recent log
        if [ -f "logs/ablation_${ABLATION_JOB_ID}.log" ]; then
            echo "  Last line: $(tail -1 logs/ablation_${ABLATION_JOB_ID}.log)"
        fi
        return 1
    else
        # Job completed or failed
        echo "[$(date '+%H:%M:%S')] Job finished"
        
        # Check exit code
        if [ -f "logs/ablation_${ABLATION_JOB_ID}.err" ]; then
            if grep -q "✅ Ablation complete" "logs/ablation_${ABLATION_JOB_ID}.log" 2>/dev/null; then
                echo "✅ ABLATION SUCCESSFUL"
                return 0
            else
                echo "❌ ABLATION FAILED"
                echo "  Error log:"
                tail -20 "logs/ablation_${ABLATION_JOB_ID}.err" | sed 's/^/    /'
                return 2
            fi
        fi
        return 1
    fi
}

# Monitor loop
echo "Starting monitoring..."
attempts=0
max_attempts=720  # 6 hours at 30-second intervals

while [ $attempts -lt $max_attempts ]; do
    check_ablation_status
    status=$?
    
    if [ $status -eq 0 ]; then
        # Ablation successful
        echo ""
        echo "=========================================="
        echo "ABLATION COMPLETE - TRIGGERING INFERENCE"
        echo "=========================================="
        
        # Check for ablation results
        if ls results/ablation_epochs*_metadata.json 1>/dev/null 2>&1; then
            echo "Found ablation metadata:"
            ls -lh results/ablation_epochs*_metadata.json
            echo ""
            echo "Ablation metrics:"
            for f in results/ablation_epochs*_metadata.json; do
                echo "  $f:"
                python3 -c "import json; d=json.load(open('$f')); print(f\"    Epochs: {d['epochs']}, Loss: {d['training_loss']:.4f}\")"
            done
        fi
        
        echo ""
        echo "Next steps:"
        echo "1. Review ablation results"
        echo "2. Run inference with fixed pipeline:"
        echo "   python evaluation/inference_v2.py"
        echo "3. Re-run judge evaluation"
        echo "4. Complete REPORT.md with findings"
        break
        
    elif [ $status -eq 2 ]; then
        # Ablation failed
        echo ""
        echo "=========================================="
        echo "ABLATION FAILED - INVESTIGATING"
        echo "=========================================="
        echo "Check logs/ablation_${ABLATION_JOB_ID}.{log,err} for details"
        break
    fi
    
    # Increment counter and sleep
    attempts=$((attempts + 1))
    sleep $CHECK_INTERVAL
done

if [ $attempts -ge $max_attempts ]; then
    echo ""
    echo "=========================================="
    echo "TIMEOUT: Job still running after 6 hours"
    echo "=========================================="
    echo "Check manually with: squeue -j $ABLATION_JOB_ID"
fi
