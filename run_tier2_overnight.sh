#!/bin/bash
# Master orchestrator for the angular-flip Tier-2 factorial. Fully session-
# independent: waits for the GPU to free (multiseed + Tier-1) and the recon-wedge
# bank to finish, then trains the factorial cells SEQUENTIALLY (one job at a time,
# so the thermal governor rarely needs to intervene), evaluates retention for each,
# and re-collates RESULTS_OVERNIGHT.md after every cell. Deadline-guarded so the
# final collation always runs before 08:00. Resume-safe (each cell --resume).
set -u
cd /home/johnb/Desktop/SinogramKernels
PY=/home/johnb/PyCharmMiscProject/.venv/bin/python
export SINO_CACHE=/mnt/mlcache/subset200k
LOG=tier2_master.log
DEADLINE=$(date -d "today 07:40" +%s)
COMMON="--arch aniso_cfg --epochs 20 --bs 128 --lr 3e-4 --wd 1e-4 --seed 0 --resume"
say(){ echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
past_deadline(){ [ "$(date +%s)" -ge "$DEADLINE" ]; }

# cells: outdir | train-domain | extra-train-flags | robust-eval-domain
# ORDER: decisive whiten=off recon cells FIRST (they complete the 2x2 that
# answers the flip), then the whiten=on cells.
CELLS=(
  "rc_none_off|recon_wedge||recon_wedge"
  "rc_wedge_off|recon_wedge|--aug-wedge-recon|recon_wedge"
  "sn_none_on|sinogram|--whiten|sinogram"
  "sn_wedge_on|sinogram|--whiten --aug-angular|sinogram"
  "rc_none_on|recon_wedge|--whiten|recon_wedge"
  "rc_wedge_on|recon_wedge|--whiten --aug-wedge-recon|recon_wedge"
)

say "=== TIER2 orchestrator start (deadline $(date -d @$DEADLINE +%H:%M)) ==="

# 1) wait for prerequisites (GPU free + bank ready), but never past deadline
while true; do
  ms=$(grep -c MULTISEED_DONE multiseed_master.log 2>/dev/null); ms=${ms:-0}
  t1=$(grep -c TIER1_DONE tier1_flip_master.log 2>/dev/null); t1=${t1:-0}
  bk=$(grep -c "\[bank\] DONE" recon_wedge_bank.log 2>/dev/null); bk=${bk:-0}
  if [ "$ms" -gt 0 ] && [ "$t1" -gt 0 ] && [ "$bk" -gt 0 ]; then
    say "prereqs met (multiseed+tier1 done, bank built) -> start factorial"; break
  fi
  if past_deadline; then say "DEADLINE hit while waiting (ms=$ms t1=$t1 bank=$bk); collate only"; break; fi
  sleep 60
done

# 2) eval the two already-trained whiten=off sinogram cells (idempotent)
[ -f abl_base/sinogram/best.pt ]  && ROBUST_EXP=abl_base  ROBUST_DOMAIN=sinogram $PY angular_robustness_flip.py >>"$LOG" 2>&1
[ -f flip_noaug/sinogram/best.pt ] && ROBUST_EXP=flip_noaug ROBUST_DOMAIN=sinogram $PY angular_robustness_flip.py >>"$LOG" 2>&1
$PY collate_flip.py >/dev/null 2>&1 || true

# 3) train + eval each factorial cell, collating after each
for spec in "${CELLS[@]}"; do
  IFS='|' read -r out dom extra rdom <<< "$spec"
  if past_deadline; then say "DEADLINE reached; skipping remaining cells (stopping at $out)"; break; fi
  mkdir -p "$out"
  say "--- cell $out (domain=$dom flags='$extra') train ---"
  $PY train_twin.py $COMMON --domains "$dom" $extra --out "$out" > "$out/stdout.log" 2>&1
  say "    trained $out : $(grep -o 'best_val_macroAUC=[0-9.]*' "$out/stdout.log" | tail -1)"
  say "    eval retention ($rdom) ..."
  ROBUST_EXP="$out" ROBUST_DOMAIN="$rdom" $PY angular_robustness_flip.py >> "$out/eval.log" 2>&1
  say "    $(grep -o 'retention@0.3 = [0-9.]*' "$out/eval.log" | tail -1)"
  $PY collate_flip.py >/dev/null 2>&1 || true
done

# 4) final collation + multiseed pooled analysis
say "=== final collation ==="
$PY collate_flip.py > collate_stdout.txt 2>&1 || true
{ echo; echo "## Multiseed noise-floor (pooled)"; echo '```'; $PY multiseed_pooled.py 2>&1; echo '```'; } >> RESULTS_OVERNIGHT.md
say "TIER2_DONE"
