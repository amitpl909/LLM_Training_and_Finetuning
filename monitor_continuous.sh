#!/bin/bash
# Continuous monitoring script for job 723856

JOB_ID=723856
LOG_FILE="/work/nbe841/LLM_Training_and_Finetuning/logs/training_723856.log"
ERR_FILE="/work/nbe841/LLM_Training_and_Finetuning/logs/training_723856.err"
MONITOR_LOG="/work/nbe841/LLM_Training_and_Finetuning/logs/monitor_723856.log"
CHECK_INTERVAL=120  # Check every 2 minutes

echo "=== Job Monitoring Started at $(date '+%Y-%m-%d %H:%M:%S') ===" > "$MONITOR_LOG"
echo "Monitoring Job $JOB_ID for 2 hours..." >> "$MONITOR_LOG"

CHECK_COUNT=0
START_TIME=$(date +%s)
MAX_DURATION=$((2 * 3600))  # 2 hours in seconds

while true; do
  CURRENT_TIME=$(date +%s)
  ELAPSED=$((CURRENT_TIME - START_TIME))
  
  if [ $ELAPSED -gt $MAX_DURATION ]; then
    echo "=== Monitoring period (2 hours) complete ===" >> "$MONITOR_LOG"
    break
  fi
  
  CHECK_COUNT=$((CHECK_COUNT + 1))
  TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
  
  {
    echo ""
    echo "=== CHECK #$CHECK_COUNT at $TIMESTAMP (Elapsed: $((ELAPSED/60))m) ==="
    
    # Get job status
    STATUS=$(squeue -j $JOB_ID 2>/dev/null | tail -1)
    
    if [ -z "$STATUS" ]; then
      echo "❌ JOB COMPLETED OR CANCELLED"
      echo "Final status:"
      sacct -j $JOB_ID --format=JobID,State,Elapsed,MaxRSS,ExitCode | grep $JOB_ID
      break
    else
      echo "✅ Job Still Running"
      echo "$STATUS"
      
      # Get resource stats
      echo ""
      echo "Resource Summary:"
      sstat -j $JOB_ID --format=AveRSS,MaxRSS,AveCPU,MaxDiskRead,MaxDiskWrite 2>/dev/null | tail -2
      
      # Get latest log progress
      if [ -f "$LOG_FILE" ]; then
        LAST_LINE=$(tail -1 "$LOG_FILE")
        LOG_LINES=$(wc -l < "$LOG_FILE")
        echo ""
        echo "Latest log entry ($LOG_LINES total lines):"
        echo "  $LAST_LINE"
      fi
      
      # Check for errors
      if [ -f "$ERR_FILE" ]; then
        ERR_LINES=$(wc -l < "$ERR_FILE" 2>/dev/null || echo "0")
        if [ "$ERR_LINES" -gt 0 ]; then
          echo ""
          echo "⚠️  Error log has $ERR_LINES lines. Last entry:"
          tail -1 "$ERR_FILE"
        fi
      fi
    fi
  } >> "$MONITOR_LOG"
  
  sleep $CHECK_INTERVAL
done

echo "" >> "$MONITOR_LOG"
echo "=== Monitoring session ended at $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$MONITOR_LOG"
echo "Check monitor log at: $MONITOR_LOG"
