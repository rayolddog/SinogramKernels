#!/usr/bin/env python3
"""plot_robust_se200k.py — angular-dropout robustness of the SE-ResNeXt50 models
(se_200k). The angular-dropout-AUGMENTED sinogram model degrades gracefully; the
un-augmented sinogram (track 1a) collapsed to chance. Recon here got no corruption
augmentation and is fed limited-angle FBP reconstructions, so its collapse is an
unfair comparison — shown for context, not as a clean sinogram-beats-recon claim."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

f = np.array([0, 10, 20, 30, 40])
sino_aug = np.array([0.9111, 0.9079, 0.9059, 0.8996, 0.8950])   # se_200k, augmented
recon = np.array([0.9069, 0.5207, 0.4838, 0.4756, 0.4801])       # se_200k, no aug
sino_unaug = np.array([0.8632, 0.5200, 0.5205, 0.5366, 0.4601])  # track 1a (no aug)

fig, ax = plt.subplots(figsize=(8.4, 5.6))
ax.plot(f, sino_aug, "o-", color="#c0392b", lw=2.4,
        label="sinogram + angular-dropout (augmented)")
ax.plot(f, sino_unaug, "^--", color="#7f8c8d", lw=1.8,
        label="sinogram, NO augmentation (track 1a)")
ax.plot(f, recon, "s-", color="#2c7fb8", lw=2.0,
        label="recon, no augmentation (fed limited-angle FBP)")
ax.axhspan(0.5, 0.53, color="0.85", alpha=0.7, zorder=0)
ax.text(40, 0.505, "chance", fontsize=8, color="0.4", ha="right")
ax.set_xlabel("angular wedge removed (% of 180°)")
ax.set_ylabel("macro-AUC")
ax.set_ylim(0.44, 0.94)
ax.set_title("Angular-dropout augmentation makes the sinogram model\n"
             "robust to lost angles (0.91→0.90 across 0–40% removed)",
             fontsize=11)
ax.legend(fontsize=8.5, loc="center right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("angular_robustness_se200k.png", dpi=150)
print("wrote angular_robustness_se200k.png")
