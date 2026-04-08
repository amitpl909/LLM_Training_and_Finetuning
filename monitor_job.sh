#!/bin/bash
# Comprehensive Job Monitoring Script
# Monitors training job 724036 and runs evaluation when complete

set -e

JOBID=724036
LOG_DIR="/work/nbe841/LLM_Training_and_Finetuning/logs"
MONITOR_LOG="${LOG_DIR}/monitor_724036.log"
TRAINING_LOG="${LOG_DIR}/training_724036.log"
PROJECT_DIR="/work/nbe841/LLM_Training_and_Finetuning"

# Initialize monitor log
echo "🔍 Monitoring Job $JOBID - Started at $(date)" | tee "$MONITOR_LOG"
echo "================================================" >> "$MONITOR_LOG"

# Function to check job status
check_job_status() {
    squeue -j "$JOBID" 2>&1 | tail -1
}

# Function to get last log lines
get_last_log_lines() {
    tail -5 "$TRAINING_LOG" 2>&1
}

# Function to log monitoring event
log_event() {
    local msg="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" | tee -a "$MONITOR_LOG"
}

# Main monitoring loop
log_event "Starting continuous monitoring..."
log_event "Job ID: $JOBID"
log_event "Training Log: $TRAINING_LOG"
log_event ""

JOB_COMPLETED=0
while [ $JOB_COMPLETED -eq 0 ]; do
    JOB_INFO=$(check_job_status)
    
    if [ -z "$JOB_INFO" ]; then
        # Job not found in queue - might be completed or failed
        JOB_COMPLETED=1
        log_event "⚠️  Job $JOBID no longer in queue (completed or cancelled)"
    else
        # Job still running
        STATUS=$(echo "$JOB_INFO" | awk '{print $5}')
        RUNTIME=$(echo "$JOB_INFO" | awk '{print $6}')
        
        if [ "$STATUS" = "R" ]; then
            log_event "✅ Job still running | Runtime: $RUNTIME"
        elif [ "$STATUS" = "CG" ]; then
            log_event "⏳ Job completing | Runtime: $RUNTIME"
        else
            log_event "⚠️  Job status: $STATUS | Runtime: $RUNTIME"
        fi
        
        # Show last training metrics every 5 minutes
        LAST_LINES=$(get_last_log_lines)
        if echo "$LAST_LINES" | grep -q "loss"; then
            LOSS=$(echo "$LAST_LINES" | grep -o "'loss': '[^']*'" | tail -1 | cut -d"'" -f4)
            EPOCH=$(echo "$LAST_LINES" | grep -o "'epoch': '[^']*'" | tail -1 | cut -d"'" -f4)
            if [ -n "$LOSS" ] && [ -n "$EPOCH" ]; then
                log_event "   Latest: loss=$LOSS, epoch=$EPOCH"
            fi
        fi
    fi
    
    # Check every 60 seconds
    sleep 60
done

# Job has completed - now check results
log_event ""
log_event "═══════════════════════════════════════════════════════"
log_event "🎉 TRAINING JOB COMPLETED!"
log_event "═══════════════════════════════════════════════════════"
log_event ""

# Check if training was successful
if [ -f "$TRAINING_LOG" ]; then
    if tail -20 "$TRAINING_LOG" | grep -q "Stage 2 training completed\|Training completed"; then
        log_event "✅ Training completed successfully"
        
        # Check if checkpoints were created
        if ls "$PROJECT_DIR"/checkpoints/stage*/checkpoint-* &> /dev/null; then
            log_event "✅ Checkpoints found and ready for evaluation"
            CHECKPOINT_COUNT=$(find "$PROJECT_DIR"/checkpoints -name "checkpoint-*" -type d | wc -l)
            log_event "   Total checkpoints: $CHECKPOINT_COUNT"
        else
            log_event "⚠️  No checkpoints found - check logs for errors"
        fi
    else
        log_event "⚠️  Training may have incomplete - check logs"
    fi
    
    # Show last 20 lines of training log
    log_event ""
    log_event "Last 20 lines of training log:"
    log_event "────────────────────────────"
    tail -20 "$TRAINING_LOG" | tee -a "$MONITOR_LOG"
else
    log_event "❌ Training log file not found!"
fi

log_event ""
log_event "═══════════════════════════════════════════════════════"
log_event "📋 NEXT STEPS:"
log_event "═══════════════════════════════════════════════════════"
log_event "1. Review final training metrics in the log above"
log_event "2. Run evaluation: cd $PROJECT_DIR && python evaluation/inference.py"
log_event "3. Run judge scoring: python evaluation/llm_judge.py"
log_event "4. Check results: cat results/metrics_table.json"
log_event "5. Submit to course portal before deadline"
log_event ""
log_event "Monitor log saved to: $MONITOR_LOG"
log_event "Training log: $TRAINING_LOG"
log_event ""
log_event "Monitoring completed at $(date)"
