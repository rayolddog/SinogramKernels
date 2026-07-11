#!/usr/bin/env python3
"""plot_angular_1a.py — track 1(a) result (from the completed frac 0.0-0.4 runs).
A clean-trained sinogram model is BRITTLE to zeroed angular wedges (out-of-
distribution), while the recon model degrades gracefully. The information is
spread through the remaining angles, but a model never trained on missing angles
can't exploit it — the robustness advantage is not automatic."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

frac = np.array([0, 10, 20, 30, 40])
sino = np.array([0.8632, 0.5200, 0.5205, 0.5366, 0.4601])
recon = np.array([0.8258, 0.8112, 0.7978, 0.7879, 0.7894])

fig, ax = plt.subplots(figsize=(7.5, 5.2))
ax.plot(frac, sino, "o-", color="#c0392b", lw=2, label="sinogram model (clean-trained)")
ax.plot(frac, recon, "s-", color="#2c7fb8", lw=2, label="recon model")
ax.axhspan(0.5, 0.55, color="0.85", alpha=0.7, zorder=0)
ax.text(40, 0.505, "chance", fontsize=8, color="0.4", ha="right")
ax.set_xlabel("angular wedge removed (% of 180°)")
ax.set_ylabel("macro-AUC")
ax.set_ylim(0.44, 0.9)
ax.set_title("Track 1(a): clean-trained sinogram model is BRITTLE to lost angles\n"
             "(refutes the naive 'information-spread → robust' hypothesis;\n"
             "fix = train with angular-dropout augmentation, then re-test)",
             fontsize=10)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("angular_robustness_1a.png", dpi=150)
print("wrote angular_robustness_1a.png")
