#!/bin/bash
# No-root thermal governor with PRIORITY throttling. Enforces GPU < 80 C by
# duty-cycling jobs (SIGSTOP/SIGCONT), preferring to throttle the LOWER-priority
# work first:
#   1st choice to pause: supplementary Tier-2 cells (flip/rc/sn)
#   2nd choice:          the multiseed seed 4/5 runs (only if no Tier-2 cell live)
# This keeps the user-priority seed runs going while holding temp down during the
# concurrent phase. Never pauses the primary seed_*_s1..3 / abl_* runs.
# Exits when the whole pipeline (through the seed extension) is done.
set -u
cd /home/johnb/Desktop/SinogramKernels
HOT=80
COOL=75
SECONDARY="out (flip_noaug|rc_|sn_)"
SEEDS="out seed_(base|gfb)_s[45]"
paused=""
log(){ echo "$(date +%H:%M:%S) [gov] $*" >> thermal_governor.log; }
log "start priority-mode (HOT=$HOT COOL=$COOL; throttle Tier-2 first, then seeds4/5)"
while true; do
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d ' ')
  if [ "$t" -ge "$HOT" ] && [ -z "$paused" ]; then
    pids=$(pgrep -f "$SECONDARY"); who="tier2"
    [ -z "$pids" ] && { pids=$(pgrep -f "$SEEDS"); who="seed4/5"; }
    if [ -n "$pids" ]; then kill -STOP $pids 2>/dev/null && paused="$pids" && log "PAUSE $who at ${t}C"; fi
  elif [ "$t" -le "$COOL" ] && [ -n "$paused" ]; then
    kill -CONT $paused 2>/dev/null && log "RESUME at ${t}C"; paused=""
  fi
  # stop when the seed extension has completed (final stage of the pipeline)
  done_ext=$(grep -c EXTSEED_DONE extseed_master.log 2>/dev/null); done_ext=${done_ext:-0}
  if [ "$done_ext" -gt 0 ]; then
    [ -n "$paused" ] && kill -CONT $paused 2>/dev/null
    log "pipeline done (EXTSEED_DONE) -> governor exit at ${t}C"; exit 0
  fi
  sleep 10
done
