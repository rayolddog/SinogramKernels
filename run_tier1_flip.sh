#!/bin/bash
# Tier-1 angular-flip test: train the aug-OFF sinogram cell, then evaluate both
# cells' wedge retention. aug-ON cell already exists as abl_base (0.8772).
# flip_noaug = identical to abl_base (aniso_cfg default, sinogram, whiten off,
# seed 0, 200k, 20ep) EXCEPT no --aug-angular. Resume-safe.
set -u
cd /home/johnb/Desktop/SinogramKernels
PY=/home/johnb/PyCharmMiscProject/.venv/bin/python
export SINO_CACHE=/mnt/mlcache/subset200k

echo "=== $(date +%H:%M:%S) TIER1 train flip_noaug ==="
$PY train_twin.py --arch aniso_cfg --domains sinogram --epochs 20 --bs 128 \
    --lr 3e-4 --wd 1e-4 --seed 0 --resume --out flip_noaug \
    > flip_noaug/stdout.log 2>&1
echo "=== $(date +%H:%M:%S) DONE train : $(grep -o 'best_val_macroAUC=[0-9.]*' flip_noaug/stdout.log | tail -1) ==="

echo "=== $(date +%H:%M:%S) TIER1 eval retention ==="
ROBUST_EXP=abl_base    $PY angular_robustness_flip.py
ROBUST_EXP=flip_noaug  $PY angular_robustness_flip.py
echo "TIER1_DONE $(date +%H:%M:%S)"
