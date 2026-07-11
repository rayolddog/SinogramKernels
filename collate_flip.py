#!/usr/bin/env python3
"""collate_flip.py — assemble the 2x2x2 angular-flip factorial into one table and
write RESULTS_OVERNIGHT.md. Reads {exp}_{domain}_flip.json retention curves.

Factor levels per cell: DOMAIN{sino,recon} x AUG{none,wedge} x WHITEN{off,on}.
The flip is explained by whichever factor retention tracks. Key contrast: within
each (domain,whiten) pair, does AUG lift retention? If retention tracks AUG and
the sino>recon ordering vanishes when AUG is matched -> H_aug (trained, not
intrinsic). If sino retains > recon even at aug=none -> H_intrinsic survives.
"""
import json
import os

# label -> (exp dir, domain-for-eval, DOMAIN factor, AUG factor, WHITEN factor)
CELLS = [
    ("sino none off",  "flip_noaug", "sinogram",    "sino",  "none",  "off"),
    ("sino wedge off",  "abl_base",  "sinogram",    "sino",  "wedge", "off"),
    ("recon none off",  "rc_none_off",  "recon_wedge", "recon", "none",  "off"),
    ("recon wedge off", "rc_wedge_off", "recon_wedge", "recon", "wedge", "off"),
    ("sino none on",   "sn_none_on",  "sinogram",    "sino",  "none",  "on"),
    ("sino wedge on",   "sn_wedge_on", "sinogram",    "sino",  "wedge", "on"),
    ("recon none on",   "rc_none_on",  "recon_wedge", "recon", "none",  "on"),
    ("recon wedge on",  "rc_wedge_on", "recon_wedge", "recon", "wedge", "on"),
]


def load(exp, domain):
    p = f"{exp}_{domain}_flip.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def main():
    rows = []
    for label, exp, dom, D, A, W in CELLS:
        r = load(exp, dom)
        rows.append((label, exp, D, A, W, r))

    lines = []
    lines.append("# Angular-robustness flip — Tier-2 factorial results\n")
    lines.append("Retention = macro-AUC(wedge)/macro-AUC(0). Higher = more robust "
                 "to lost angular data. All cells: aniso_cfg, 200k, 20 ep, seed 0.\n")
    lines.append("| cell | domain | aug | whiten | macro@0 | ret@0.3 | ret@0.5 |")
    lines.append("|---|---|---|---|---|---|---|")
    for label, exp, D, A, W, r in rows:
        if r is None:
            lines.append(f"| {label} | {D} | {A} | {W} | _pending_ | _–_ | _–_ |")
        else:
            m0 = r["macro"][0]
            lines.append(f"| {label} | {D} | {A} | {W} | {m0:.4f} | "
                         f"{r['retention'][3]:.3f} | {r['retention'][5]:.3f} |")

    # AUG effect within each (domain, whiten)
    lines.append("\n## AUG effect on retention@0.3 (within each domain×whiten)\n")
    lines.append("| domain | whiten | ret none | ret wedge | Δ(wedge−none) |")
    lines.append("|---|---|---|---|---|")
    idx = {(D, A, W): r for (_, _, D, A, W, r) in rows}
    verdict_bits = []
    for D in ("sino", "recon"):
        for W in ("off", "on"):
            rn, rw = idx.get((D, "none", W)), idx.get((D, "wedge", W))
            if rn and rw:
                a, b = rn["retention"][3], rw["retention"][3]
                lines.append(f"| {D} | {W} | {a:.3f} | {b:.3f} | {b-a:+.3f} |")
                verdict_bits.append((D, W, a, b))
            else:
                sa = "–" if not rn else f"{rn['retention'][3]:.3f}"
                sb = "–" if not rw else f"{rw['retention'][3]:.3f}"
                lines.append(f"| {D} | {W} | {sa} | {sb} | _pending_ |")

    # DOMAIN ordering at aug=none, whiten=off (the H_intrinsic test)
    sn = idx.get(("sino", "none", "off"))
    rc = idx.get(("recon", "none", "off"))
    lines.append("\n## Interpretation\n")
    if sn and rc:
        s, c = sn["retention"][3], rc["retention"][3]
        lines.append(f"- **H_intrinsic test** (aug=none, whiten=off): "
                     f"sino ret@0.3={s:.3f} vs recon ret@0.3={c:.3f}. "
                     + ("Sino > recon WITHOUT augmentation → intrinsic-domain "
                        "robustness survives the control (H_intrinsic supported)."
                        if s - c > 0.03 else
                        "Sino ≈ recon without augmentation → no intrinsic advantage; "
                        "the flip is explained by augmentation (H_aug)."))
    else:
        lines.append("- H_intrinsic test: pending (needs sino/none/off and "
                     "recon/none/off).")
    for D, W, a, b in verdict_bits:
        if b - a > 0.05:
            lines.append(f"- {D}/{W}: augmentation lifts retention by {b-a:+.3f} "
                         f"→ wedge robustness is (at least partly) TRAINED here.")

    out = "RESULTS_OVERNIGHT.md"
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[collate] -> {out}")


if __name__ == "__main__":
    main()
