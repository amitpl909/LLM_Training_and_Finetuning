#!/bin/bash

JOBID=723323
LOG_FILE="/work/nbe841/LLM_Training_and_Finetuning/logs/training_723323.log"
ERR_FILE="/work/nbe841/LLM_Training_and_Finetuning/logs/training_723323.err"
REPORT_FILE="/work/nbe841/LLM_Training_and_Finetuning/logs/job_report.md"
CHECK_INTERVAL=600  # Check every 10 minutes

# Initialize report
cat > "$REPORT_FILE" << 'REPORT_INIT'
# Job 723323 Status Report
**Generated:** $(date)
**Status:** Monitoring in Progress

---

## Timeline of Checks
REPORT_INIT

echo "Starting continuous reporting for Job 723323..." >&2

CHECK_COUNT=0
while true; do
  CHECK_COUNT=$((CHECK_COUNT + 1))
  TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
  
  # Get job status
  STATUS=$(squeue -j $JOBID 2>/dev/null | tail -1)
  
  # Create timestamped report section
  {
    echo ""
    echo "## Check #$CHECK_COUNT - $TIMESTAMP"
    echo ""
    
    if [ -z "$STATUS" ]; then
      # Job completed
      echo "### ❌ JOB COMPLETED"
      echo ""
      echo "**Job Status:** FINISHED"
      echo ""
      
      # Get final stats
      TOTAL_LINES=$(wc -l < "$LOG_FILE" 2>/dev/null || echo "0")
      echo "**Total log lines:** $TOTAL_LINES"
      echo ""
      
      # Show last training output
      echo "### Final Training Output"
      echo '```'
      tail -30 "$LOG_FILE" 2>/dev/null || echo "Log file not found"
      echo '```'
      echo ""
      
      # Check for errors
      if [ -f "$ERR_FILE" ]; then
        ERROR_COUNT=$(wc -l < "$ERR_FILE")
        echo "**Error log lines:** $ERROR_COUNT"
        if [ "$ERROR_COUNT" -gt 0 ]; then
          echo "### Error Summary"
          echo '```'
          tail -20 "$ERR_FILE"
          echo '```'
        fi
      fi
      
      echo ""
      echo "---"
      echo "**Report Status:** Job finished. Check complete."
      
      break
    else
      # Job still running
      RUNTIME=$(echo "$STATUS" | awk '{print $6}')
      NODE=$(echo "$STATUS" | awk '{print $7}')
      
      echo "### ✅ Job Still Running"
      echo ""
      echo "| Metric | Value |"
      echo "|--------|-------|"
      echo "| **Runtime** | $RUNTIME |"
      echo "| **Node** | $NODE |"
      echo "| **Job ID** | $JOBID |"
      echo ""
      
      # Get latest log progress
      if [ -f "$LOG_FILE" ]; then
        LAST_LINE=$(tail -1 "$LOG_FILE")
        LOG_LINES=$(wc -l < "$LOG_FILE")
        
        echo "**Latest log entry:**"
        echo '```'
        echo "$LAST_LINE"
        echo '```'
        echo ""
        echo "**Total log lines:** $LOG_LINES"
        echo ""
      fi
      
      # Check for errors
      if [ -f "$ERR_FILE" ]; then
        ERROR_COUNT=$(wc -l < "$ERR_FILE" 2>/dev/null || echo "0")
        if [ "$ERROR_COUNT" -gt 0 ]; then
          echo "⚠️ **WARNINGS/ERRORS DETECTED:**"
          echo '```'
          grep -i "error\|warning\|exception" "$ERR_FILE" | tail -5 || echo "Error log contains lines but no errors/warnings found"
          echo '```'
          echo ""
        fi
      fi
    fi
  } >> "$REPORT_FILE"
  
  # Display progress to console
  echo "[$TIMESTAMP] Check #$CHECK_COUNT complete. Report updated."
  
  # Check if job completed
  if [ -z "$STATUS" ]; then
    echo "Job completed. Stopping monitoring."
    {
      echo ""
      echo "---"
      echo "**Monitoring ended at:** $(date '+%Y-%m-%d %H:%M:%S')"
      echo "**Total checks:** $CHECK_COUNT"
    } >> "$REPORT_FILE"
    break
  fi
  
  # Sleep before next check
  sleep $CHECK_INTERVAL
done

echo "Reporting complete."
