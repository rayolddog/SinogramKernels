# Angular-robustness flip — Tier-2 factorial results

Retention = macro-AUC(wedge)/macro-AUC(0). Higher = more robust to lost angular data. All cells: aniso_cfg, 200k, 20 ep, seed 0.

| cell | domain | aug | whiten | macro@0 | ret@0.3 | ret@0.5 |
|---|---|---|---|---|---|---|
| sino none off | sino | none | off | 0.8858 | 0.541 | 0.526 |
| sino wedge off | sino | wedge | off | 0.8921 | 0.983 | 0.942 |
| recon none off | recon | none | off | 0.9113 | 0.674 | 0.420 |
| recon wedge off | recon | wedge | off | 0.9054 | 0.975 | 0.946 |
| sino none on | sino | none | on | 0.8959 | 0.558 | 0.532 |
| sino wedge on | sino | wedge | on | 0.8935 | 0.926 | 0.884 |
| recon none on | recon | none | on | 0.8987 | 0.901 | 0.873 |
| recon wedge on | recon | wedge | on | _pending_ | _–_ | _–_ |

## AUG effect on retention@0.3 (within each domain×whiten)

| domain | whiten | ret none | ret wedge | Δ(wedge−none) |
|---|---|---|---|---|
| sino | off | 0.541 | 0.983 | +0.442 |
| sino | on | 0.558 | 0.926 | +0.367 |
| recon | off | 0.674 | 0.975 | +0.301 |
| recon | on | 0.901 | – | _pending_ |

## Interpretation

- **H_intrinsic test** (aug=none, whiten=off): sino ret@0.3=0.541 vs recon ret@0.3=0.674. Sino ≈ recon without augmentation → no intrinsic advantage; the flip is explained by augmentation (H_aug).
- sino/off: augmentation lifts retention by +0.442 → wedge robustness is (at least partly) TRAINED here.
- sino/on: augmentation lifts retention by +0.367 → wedge robustness is (at least partly) TRAINED here.
- recon/off: augmentation lifts retention by +0.301 → wedge robustness is (at least partly) TRAINED here.

## Multiseed noise-floor (pooled)
```
seed      base       gfb       gap
   0    0.8772    0.8814   +0.0042
   1    0.8720    0.8807   +0.0087
   2    0.8764    0.8776   +0.0012
   3    0.8741    0.8788   +0.0047
   4      None      None        --  (incomplete)
   5      None      None        --  (incomplete)

--- pooled ---
 base: n=4 mean=0.8749 sd=0.0024 [0.8720, 0.8772]
  gfb: n=4 mean=0.8796 sd=0.0017 [0.8776, 0.8814]

paired gap (gfb-base): mean=+0.0047 sd=0.0031 n=4
paired t=3.050  Cohen's d=1.525
vs t_crit(df=3,.05)=3.182: WITHIN seed noise (gap not distinguishable)
```
