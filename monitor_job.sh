#!/bin/bash
WORKSPACE="/work/nbe841/LLM_Training_and_Finetuning"
JOB_ID="722618"
LOGFILE="$WORKSPACE/monitor_722618.log"

# Function to log with timestamp
log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

log_msg "=== Monitoring started for Job $JOB_ID ==="

# Check every 2 minutes for 13 hours  
for i in {1..390}; do
    STATUS=$(squeue -j $JOB_ID -h 2>/dev/null)
    
    if [ -z "$STATUS" ]; then
        # Job not found - either completed or failed
        log_msg "Job $JOB_ID no longer running - checking logs..."
        
        if [ -f "$WORKSPACE/logs/training_${JOB_ID}.log" ]; then
            TAIL_OUTPUT=$(tail -20 "$WORKSPACE/logs/training_${JOB_ID}.log" 2>/dev/null)
            log_msg "Last 20 lines of training_${JOB_ID}.log:"
            log_msg "$TAIL_OUTPUT"
        fi
        
        if [ -f "$WORKSPACE/logs/training_${JOB_ID}.err" ]; then
            ERR_OUTPUT=$(tail -20 "$WORKSPACE/logs/training_${JOB_ID}.err" 2>/dev/null)
            log_msg "Last 20 lines of training_${JOB_ID}.err:"
            log_msg "$ERR_OUTPUT"
        fi
        
        log_msg "=== JOB COMPLETED or FAILED ==="
        break
    else
        # Job still running
        ELAPSED=$(echo "$STATUS" | awk '{print $4}')
        TIME_LIMIT=$(echo "$STATUS" | awk '{print $5}')
        STATE=$(echo "$STATUS" | awk '{print $3}')
        NODE=$(echo "$STATUS" | awk '{print $8}')
        
        log_msg "Job $JOB_ID status: STATE=$STATE ELAPSED=$ELAPSED TIME_LIMIT=$TIME_LIMIT NODE=$NODE"
        
        # Check every 2 minutes
        sleep 120
    fi
done

log_msg "=== Monitoring finished ==="
