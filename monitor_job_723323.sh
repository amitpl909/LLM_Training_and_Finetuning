#!/bin/bash

JOBID=723323
LOG_FILE="/work/nbe841/LLM_Training_and_Finetuning/logs/training_723323.log"
ERR_FILE="/work/nbe841/LLM_Training_and_Finetuning/logs/training_723323.err"
MONITOR_LOG="/work/nbe841/LLM_Training_and_Finetuning/logs/monitor_723323.log"

echo "=== Starting Job Monitor for Job $JOBID ===" | tee -a $MONITOR_LOG

while true; do
  TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
  echo -e "\n[$TIMESTAMP] Checking job..." | tee -a $MONITOR_LOG
  
  # Check job status
  STATUS=$(squeue -j $JOBID 2>/dev/null | tail -1)
  
  if [ -z "$STATUS" ]; then
    echo "[$TIMESTAMP] ❌ Job 723323 COMPLETED" | tee -a $MONITOR_LOG
    echo "[$TIMESTAMP] Last 50 lines of log:" | tee -a $MONITOR_LOG
    tail -50 $LOG_FILE | tee -a $MONITOR_LOG
    echo "[$TIMESTAMP] Exit status: $(cat $ERR_FILE 2>/dev/null | tail -5)" | tee -a $MONITOR_LOG
    break
  else
    RUNTIME=$(echo "$STATUS" | awk '{print $6}')
    NODE=$(echo "$STATUS" | awk '{print $7}')
    echo "[$TIMESTAMP] ✅ Job Running - Runtime: $RUNTIME on $NODE" | tee -a $MONITOR_LOG
    echo "[$TIMESTAMP] Last training line:" | tee -a $MONITOR_LOG
    tail -1 $LOG_FILE | tee -a $MONITOR_LOG
  fi
  
  # Check for errors
  if grep -q "Error\|error\|Exception" $ERR_FILE 2>/dev/null; then
    echo "[$TIMESTAMP] ⚠️  ERRORS found in error log:" | tee -a $MONITOR_LOG
    tail -10 $ERR_FILE | tee -a $MONITOR_LOG
  fi
  
  # Wait 5 minutes before next check
  echo "[$TIMESTAMP] Sleeping 5 minutes..." | tee -a $MONITOR_LOG
  sleep 300
done

echo "[$TIMESTAMP] Monitor finished." | tee -a $MONITOR_LOG
