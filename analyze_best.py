#!/usr/bin/env python3
"""analyze_best.py EXPDIR [TAG] — one-line effect-size summary from the val-optimal
(best.pt) kernels of a twin run, appended to the overnight log."""
import sys
import numpy as np
import torch
import analyze_kernels as ak


def main():
    exp = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else exp
    res = {}
    for dom in ["recon", "sinogram"]:
        ck = torch.load(f"{exp}/{dom}/best.pt", map_location="cpu", weights_only=False)
        res[dom] = (ak.describe(ck["conv1"][:, 0]), ck["epoch"], ck["val_auc"])
    line = (f"- **{tag}**: recon ep{res['recon'][1]}/AUC{res['recon'][2]:.4f}  "
            f"sino ep{res['sinogram'][1]}/AUC{res['sinogram'][2]:.4f}  |  ")
    for k in ["cent", "det_ratio", "aniso"]:
        a = np.array([r[k] for r in res["recon"][0]])
        b = np.array([r[k] for r in res["sinogram"][0]])
        line += f"{k} d={ak.cohens_d(b, a):+.2f}  "
    print(line)


if __name__ == "__main__":
    main()
