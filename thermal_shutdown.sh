#!/bin/bash
# Thermal safety shutdown: power OFF the machine when the GPU reaches 80 C.
# Debounced (2 consecutive reads ~10-20s) so a 1-sample sensor blip doesn't
# trigger it. Gracefully SIGTERMs training first and syncs disks; all training
# is --resume-safe (checkpoints each epoch), so at most the in-progress epoch is
# lost. Poweroff is authorized non-interactively (logind CanPowerOff=yes).
set -u
cd /home/johnb/Desktop/SinogramKernels
HOT=80
NEED=2
hits=0
log(){ echo "$(date +%H:%M:%S) [shutdown] $*" >> thermal_shutdown.log; }
log "armed: poweroff when GPU >= ${HOT}C for ${NEED} consecutive reads"
while true; do
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d ' ')
  if [ "${t:-0}" -ge "$HOT" ]; then hits=$((hits+1)); else hits=0; fi
  [ "$hits" -ge 1 ] && log "temp=${t}C hits=${hits}/${NEED}"
  if [ "$hits" -ge "$NEED" ]; then
    log "SUSTAINED ${t}C >= ${HOT}C -> graceful stop + poweroff"
    pkill -TERM -f train_twin.py 2>/dev/null
    pkill -TERM -f make_recon_wedge_bank.py 2>/dev/null
    sync; sleep 3
    systemctl poweroff 2>>thermal_shutdown.log \
      || busctl call org.freedesktop.login1 /org/freedesktop/login1 \
           org.freedesktop.login1.Manager PowerOff b false 2>>thermal_shutdown.log
    exit 0
  fi
  sleep 10
done
