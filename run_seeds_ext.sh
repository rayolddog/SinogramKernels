#!/bin/bash
# Extend the multiseed noise-floor test with the 5th and 6th seeds (seed values
# 4 and 5) for both configs, to resolve the borderline base-vs-gfb gap
# (was t=3.05 vs t_crit=3.18 at n=4). Waits for the GPU to free (TIER2_DONE),
# runs the 4 jobs SEQUENTIALLY (single-job load stays ~68C, well under the 80C
# limit), then re-runs the pooled analysis over all 6 seeds. Resume-safe.
set -u
cd /home/johnb/Desktop/SinogramKernels
PY=/home/johnb/PyCharmMiscProject/.venv/bin/python
export SINO_CACHE=/mnt/mlcache/subset200k
C="--domains sinogram --epochs 20 --bs 128 --lr 3e-4 --wd 1e-4 --aug-angular --resume"
say(){ echo "$(date +%H:%M:%S) $*" | tee -a extseed_master.log; }

run(){ out="$1"; shift; mkdir -p "$out"
  say "=== START $out ($*) ==="
  $PY train_twin.py "$@" $C --out "$out" > "$out/stdout.log" 2>&1
  say "=== DONE $out : $(grep -o 'best_val_macroAUC=[0-9.]*' "$out/stdout.log" | tail -1) ==="; }

if [ "${EXTSEED_NOWAIT:-0}" = "1" ]; then
  say "extseed: NOWAIT -> starting seeds 4,5 concurrently (unified governor handles heat)"
else
  say "extseed: waiting for TIER2_DONE (GPU free)"
  while ! grep -q TIER2_DONE tier2_master.log 2>/dev/null; do sleep 60; done
  say "extseed: GPU free -> starting seeds 4,5"
fi

for S in 4 5; do
  run seed_base_s$S --arch aniso_cfg --seed $S
  run seed_gfb_s$S  --arch aniso_cfg --stem tall --fulres --resk 5 --resdil --seed $S
done

say "=== pooled analysis over all 6 seeds ==="
$PY multiseed_pooled.py > multiseed_pooled_final.txt 2>&1
say "EXTSEED_DONE"
cat multiseed_pooled_final.txt | tee -a extseed_master.log
