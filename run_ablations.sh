#!/bin/bash
# Overnight ablation: isolate which of aniso2's modifications helped.
# Add ONE change at a time from the baseline (aniso_h: 15x60 stem, stride2, 3x3
# body, no dilation), all at 200k / 20 epochs so the only variable is the change.
# Resume-safe: re-run this script to continue (completed runs skip).
set -u
cd /home/johnb/Desktop/SinogramKernels
PY=/home/johnb/PyCharmMiscProject/.venv/bin/python
export SINO_CACHE=/mnt/mlcache/subset200k
C="--domains sinogram --epochs 20 --bs 128 --lr 3e-4 --wd 1e-4 --aug-angular --resume"

run(){ out="$1"; shift; mkdir -p "$out"
       echo "=== $(date +%H:%M:%S) START $out ($*) ==="
       $PY train_twin.py "$@" $C --out "$out" > "$out/stdout.log" 2>&1
       echo "=== $(date +%H:%M:%S) DONE  $out : $(grep -o 'best_val_macroAUC=[0-9.]*' "$out/stdout.log" | tail -1) ==="; }

run abl_base   --arch aniso_cfg
run abl_tall   --arch aniso_cfg --stem tall
run abl_fulres --arch aniso_cfg --fulres
run abl_dil    --arch aniso_cfg --resdil
run abl_5x5    --arch aniso_cfg --resk 5
run abl_all    --arch aniso_cfg --stem tall --fulres --resk 5 --resdil
run abl_iso    --arch iso
echo "ALL_ABLATIONS_DONE $(date +%H:%M:%S)"
