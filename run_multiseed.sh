#!/bin/bash
# Multi-seed noise-floor test: baseline aniso vs go-for-broke (aniso2 arch), 200k,
# 20 epochs, seeds 1-3 (seed 0 already exists as abl_base / abl_all). Pool all 4
# seeds per config to see if the ~0.004 gap separates from seed variance.
# Resume-safe: re-run to continue (finished runs skip).
set -u
cd /home/johnb/Desktop/SinogramKernels
PY=/home/johnb/PyCharmMiscProject/.venv/bin/python
export SINO_CACHE=/mnt/mlcache/subset200k
C="--domains sinogram --epochs 20 --bs 128 --lr 3e-4 --wd 1e-4 --aug-angular --resume"
run(){ out="$1"; shift; mkdir -p "$out"
  echo "=== $(date +%H:%M:%S) START $out ==="
  $PY train_twin.py "$@" $C --out "$out" > "$out/stdout.log" 2>&1
  echo "=== $(date +%H:%M:%S) DONE $out : $(grep -o 'best_val_macroAUC=[0-9.]*' "$out/stdout.log" | tail -1) ==="; }
for S in 1 2 3; do
  run seed_base_s$S --arch aniso_cfg --seed $S
  run seed_gfb_s$S  --arch aniso_cfg --stem tall --fulres --resk 5 --resdil --seed $S
done
echo "MULTISEED_DONE $(date +%H:%M:%S)"
