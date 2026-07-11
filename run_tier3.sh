#!/bin/bash
# Tier-3 capacity check: does the H_aug angular-flip result reproduce at HIGH
# capacity (SE-ResNeXt50, ~27M params) or was it an artifact of the small
# aniso_cfg? The aug-ON cell already exists = se_200k/sinogram (seresnext50,
# --aug-angular, whiten off, 25ep). Train the matched aug-OFF twin and compare
# wedge retention on the same 2000-slice test set.
set -u
cd /home/johnb/Desktop/SinogramKernels
PY=/home/johnb/PyCharmMiscProject/.venv/bin/python
export SINO_CACHE=/mnt/mlcache/subset200k
say(){ echo "$(date +%H:%M:%S) $*" | tee -a tier3_master.log; }
mkdir -p se_noaug

say "=== TIER3 train se_noaug (seresnext50 sino, NO aug, whiten off, 25ep) ==="
$PY train_twin.py --arch seresnext50 --domains sinogram --epochs 25 --bs 128 \
    --lr 3e-4 --wd 1e-4 --seed 0 --resume --out se_noaug > se_noaug/stdout.log 2>&1
say "trained se_noaug : $(grep -o 'best_val_macroAUC=[0-9.]*' se_noaug/stdout.log | tail -1)"

say "=== eval retention (sinogram, same test set) ==="
ROBUST_EXP=se_noaug ROBUST_DOMAIN=sinogram $PY angular_robustness_flip.py >> tier3_master.log 2>&1
ROBUST_EXP=se_200k  ROBUST_DOMAIN=sinogram $PY angular_robustness_flip.py >> tier3_master.log 2>&1

$PY - <<'PYEOF' | tee -a tier3_master.log RESULTS_TIER3.txt
import json
def r(e):
    d = json.load(open(f"{e}_sinogram_flip.json")); return d["retention"][3], d["macro"][0]
off_r, off_m = r("se_noaug"); on_r, on_m = r("se_200k")
drop = on_r - off_r
print("SE-ResNeXt50 (high capacity) sinogram wedge retention@0.3:")
print(f"  aug-OFF (se_noaug) = {off_r:.3f}  (macro@0 {off_m:.4f})")
print(f"  aug-ON  (se_200k)  = {on_r:.3f}  (macro@0 {on_m:.4f})")
print(f"  delta (aug lift)   = {drop:+.3f}")
print("Reference (aniso_cfg small model): aug-OFF 0.541 -> aug-ON 0.983 (delta +0.442)")
if off_r < 0.75 and drop > 0.15:
    print("VERDICT: H_aug REPRODUCES at high capacity -> the flip is NOT a capacity "
          "artifact; sinogram wedge robustness is trained regardless of model size.")
elif off_r >= 0.9:
    print("VERDICT: high-capacity model IS robust without aug -> capacity matters; "
          "the small-model result may not generalize. Revisit.")
else:
    print("VERDICT: intermediate; see numbers above.")
PYEOF
say "TIER3_DONE"
