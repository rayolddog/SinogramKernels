#!/bin/bash
# Tier-3b: is the high-capacity aug-OFF wedge cushion sinogram-SPECIFIC or just
# "big models generalize to the corruption"? Train the recon-domain twin of
# se_noaug at seresnext50 (recon_wedge, NO aug, whiten off, 25ep) and compare its
# wedge retention to se_noaug (sinogram aug-OFF = 0.814@0.3). Similar retention
# => cushion is domain-general (not sino-special); sino >> recon => sino-specific.
set -u
cd /home/johnb/Desktop/SinogramKernels
PY=/home/johnb/PyCharmMiscProject/.venv/bin/python
export SINO_CACHE=/mnt/mlcache/subset200k
say(){ echo "$(date +%H:%M:%S) $*" | tee -a tier3b_master.log; }
mkdir -p se_recon_noaug

say "=== TIER3b train se_recon_noaug (seresnext50 recon_wedge, NO aug, whiten off, 25ep) ==="
$PY train_twin.py --arch seresnext50 --domains recon_wedge --epochs 25 --bs 128 \
    --lr 3e-4 --wd 1e-4 --seed 0 --resume --out se_recon_noaug > se_recon_noaug/stdout.log 2>&1
say "trained se_recon_noaug : $(grep -o 'best_val_macroAUC=[0-9.]*' se_recon_noaug/stdout.log | tail -1)"

say "=== eval retention (recon_wedge, same test set) ==="
ROBUST_EXP=se_recon_noaug ROBUST_DOMAIN=recon_wedge $PY angular_robustness_flip.py >> tier3b_master.log 2>&1

$PY - <<'PYEOF' | tee -a tier3b_master.log RESULTS_TIER3B.txt
import json
rec = json.load(open("se_recon_noaug_recon_wedge_flip.json"))
sino = json.load(open("se_noaug_sinogram_flip.json"))
rr, sr = rec["retention"][3], sino["retention"][3]
print("High-capacity (seresnext50) aug-OFF wedge retention@0.3:")
print(f"  sinogram (se_noaug)        = {sr:.3f}")
print(f"  recon    (se_recon_noaug)  = {rr:.3f}")
print(f"  sino - recon               = {sr-rr:+.3f}")
print("Small-model (aniso_cfg) aug-OFF ref: sino 0.541, recon 0.674")
print(f"  recon full curve: {[round(x,3) for x in rec['retention']]}")
if abs(sr-rr) < 0.06:
    print("VERDICT: recon ~= sino at high capacity -> the aug-OFF cushion is "
          "DOMAIN-GENERAL (big models tolerate the shift), NOT sinogram-specific. "
          "Further undercuts H2 (no projection-domain-specific robustness).")
elif sr - rr > 0.06:
    print("VERDICT: sino > recon at high capacity -> a sinogram-SPECIFIC cushion "
          "exists at moderate wedges; partial support for a projection-domain edge.")
else:
    print("VERDICT: recon > sino at high capacity -> image domain is MORE robust; "
          "opposite of H2.")
PYEOF
say "TIER3B_DONE"
