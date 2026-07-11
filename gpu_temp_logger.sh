#!/bin/bash
# Log GPU temp/util/power/VRAM every 60s to gpu_temp_log.csv.
# Exits on its own when MULTISEED_DONE appears in multiseed_master.log.
cd /home/johnb/Desktop/SinogramKernels
LOG=gpu_temp_log.csv
[ -f "$LOG" ] || echo "timestamp,temp_C,util_pct,power_W,mem_used_MiB" > "$LOG"
while true; do
  echo "$(date +%Y-%m-%dT%H:%M:%S),$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw,memory.used --format=csv,noheader,nounits | tr -d ' ')" >> "$LOG"
  grep -q MULTISEED_DONE multiseed_master.log && exit 0
  sleep 60
done
