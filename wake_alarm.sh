#!/bin/bash
# 8 AM wake alarm. Waits until 08:00 local, assembles the angular-flip verdict,
# then physically wakes the user: speaks the verdict (spd-say), plays a repeating
# tone (paplay), desktop notification (notify-send) and a persistent popup (zenity).
# Writes VERDICT_8AM.txt. Env (DISPLAY, XDG_RUNTIME_DIR, PULSE) inherited at launch.
set -u
cd /home/johnb/Desktop/SinogramKernels
PY=/home/johnb/PyCharmMiscProject/.venv/bin/python
WAKE_HHMM=0800
TONE=/usr/share/sounds/freedesktop/stereo/complete.oga
STOP=/home/johnb/Desktop/SinogramKernels/STOP_ALARM
export DISPLAY="${DISPLAY:-:1}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/1000}"
log(){ echo "$(date +%H:%M:%S) [alarm] $*" >> wake_alarm.log; }

log "armed for $WAKE_HHMM"
# 1) wait until wake time
while [ "$(date +%H%M)" -lt "$WAKE_HHMM" ]; do sleep 30; done
log "wake time reached"

# 2) assemble verdict
$PY make_verdict.py > VERDICT_8AM.txt 2>&1 || echo "verdict assembly failed" > VERDICT_8AM.txt
SPOKEN=$(grep '^SPOKEN::' VERDICT_8AM.txt | sed 's/^SPOKEN:: //')
[ -z "$SPOKEN" ] && SPOKEN="Good morning. The sinogram results are on screen."
BODY=$(grep -v '^SPOKEN::' VERDICT_8AM.txt)
log "verdict: $SPOKEN"

# 3) persistent on-screen popup (non-blocking)
( zenity --info --title="SinogramKernels — 8 AM verdict" \
    --width=520 --text="$SPOKEN

$BODY" 2>/dev/null ) &
notify-send -u critical "SinogramKernels — verdict ready" "$SPOKEN" 2>/dev/null

# 4) audible wake loop (~8 min max, or until STOP_ALARM file appears)
for i in $(seq 1 24); do
  [ -f "$STOP" ] && { log "stopped by user (STOP_ALARM)"; break; }
  paplay "$TONE" 2>/dev/null || aplay -q /usr/share/sounds/alsa/Front_Center.wav 2>/dev/null
  if [ $((i % 2)) -eq 1 ]; then
    spd-say -w "Good morning John. $SPOKEN" 2>/dev/null
  fi
  sleep 15
done
log "alarm sequence done"
