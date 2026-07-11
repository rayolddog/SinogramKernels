#!/usr/bin/env python3
"""make_verdict.py — assemble the one-paragraph angular-flip verdict from whatever
results exist by wake time. Prints a spoken-friendly first line, then detail.
"""
import json
import os


def ret(exp, dom, k=3):
    p = f"{exp}_{dom}_flip.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))["retention"][k]


def pooled_line():
    # cheap re-read of multiseed logs via multiseed_pooled if present
    import subprocess
    try:
        out = subprocess.run(["python3", "multiseed_pooled.py"],
                             capture_output=True, text=True, timeout=60).stdout
        for ln in out.splitlines():
            if "SEPARATES" in ln or "WITHIN seed noise" in ln:
                return ln.strip()
    except Exception:
        pass
    return None


def main():
    augon = ret("abl_base", "sinogram")      # aug-ON sino (0.983 expected)
    augoff = ret("flip_noaug", "sinogram")   # aug-OFF sino (decisive)
    rc_off = ret("rc_none_off", "recon_wedge")
    rc_on = ret("rc_wedge_off", "recon_wedge")

    spoken = "Sinogram experiment verdict."
    detail = []

    if augoff is None:
        spoken = ("The overnight run did not finish the decisive cell. "
                  "Tier one aug-off is still pending.")
        detail.append("Tier-1 aug-OFF (flip_noaug) retention not found — check "
                      "flip_noaug/stdout.log; run may have stalled.")
    else:
        a_on = augon if augon is not None else 0.983
        drop = a_on - augoff
        if augoff >= 0.95:
            spoken = (f"Intrinsic. The sinogram stays robust without augmentation, "
                      f"retaining {augoff*100:.0f} percent at a thirty percent wedge. "
                      f"The grant's hypothesis survives the control.")
            hyp = "H_intrinsic SUPPORTED"
        elif drop >= 0.05:
            spoken = (f"Trained, not intrinsic. Without augmentation the sinogram "
                      f"retention falls to {augoff*100:.0f} percent, versus "
                      f"{a_on*100:.0f} percent with augmentation. The robustness was "
                      f"learned from the wedge augmentation. The grant's spread-"
                      f"information claim needs reframing.")
            hyp = "H_aug SUPPORTED (robustness is trained)"
        else:
            spoken = (f"Mixed. Aug-off retention {augoff*100:.0f} percent versus "
                      f"aug-on {a_on*100:.0f} percent — a small gap; augmentation "
                      f"helps modestly but does not fully explain the flip.")
            hyp = "AMBIGUOUS (small aug effect)"
        detail.append(f"VERDICT: {hyp}")
        detail.append(f"  sino retention@0.3  aug-ON={a_on:.3f}  aug-OFF={augoff:.3f}"
                      f"  (Δ={a_on-augoff:+.3f})")

    if rc_off is not None:
        detail.append(f"  recon retention@0.3 aug-OFF={rc_off:.3f}"
                      + (f"  aug-ON={rc_on:.3f}" if rc_on is not None else "  aug-ON=pending"))
    else:
        detail.append("  recon cells: pending (Tier-2 factorial incomplete)")

    pl = pooled_line()
    if pl:
        detail.append(f"  multiseed: {pl}")

    print("SPOKEN:: " + spoken)
    print("\n".join(detail))


if __name__ == "__main__":
    main()
