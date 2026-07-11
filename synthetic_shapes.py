#!/usr/bin/env python3
"""synthetic_shapes.py — a parametric shape generator to probe an architecture's
representational efficiency by SPATIAL SCALE and shape class.

Controllable factors (each independently variable → a factorial stimulus set):
  size (radius), border thickness, eccentricity/orientation, curvature, contrast.
Shape vocabulary chosen for the ICH problem:
  disk, ellipse, annulus (border thickness), crescent (subdural lune),
  branching (subarachnoid, thin material in sulci), gyri (folded ribbons),
  ventricles (paired dark curves).

Every shape can be Radon-transformed → paired image/sinogram synthetic data with
KNOWN ground truth, so the same primitive can be studied in both domains and used
to design/compare sinogram-adapted architectures cheaply before full ICH training.
"""
import numpy as np
from skimage.draw import disk as sk_disk, ellipse as sk_ellipse, line as sk_line
from skimage.transform import radon
from scipy.ndimage import binary_dilation

N = 192


def blank():
    return np.zeros((N, N), float)


def add_disk(img, cy, cx, r, val):
    rr, cc = sk_disk((cy, cx), r, shape=img.shape)
    img[rr, cc] = val
    return img


def shape_disk(r=24, val=1.0):
    return add_disk(blank(), N // 2, N // 2, r, val)


def shape_ellipse(a=36, b=18, ang=30, val=1.0):
    img = blank()
    rr, cc = sk_ellipse(N // 2, N // 2, a, b, rotation=np.deg2rad(ang), shape=img.shape)
    img[rr, cc] = val
    return img


def shape_annulus(r=36, thick=5, val=1.0):
    img = add_disk(blank(), N // 2, N // 2, r, val)
    return add_disk(img, N // 2, N // 2, r - thick, 0.0)


def shape_crescent(r=42, offset=15, val=1.0):
    """Subdural lune: a disk minus a shifted disk → crescent hugging a curve."""
    img = add_disk(blank(), N // 2 + 6, N // 2, r, val)
    return add_disk(img, N // 2 + 6 - offset, N // 2, r, 0.0)


def shape_branching(val=1.0, seed=1):
    """Subarachnoid-like: thin bright material branching through sulci."""
    img = blank()
    rng = np.random.default_rng(seed)

    def branch(y, x, ang, length, depth):
        if depth == 0 or length < 7:
            return
        y2 = int(np.clip(y + length * np.sin(ang), 1, N - 2))
        x2 = int(np.clip(x + length * np.cos(ang), 1, N - 2))
        rr, cc = sk_line(int(y), int(x), y2, x2)
        img[rr, cc] = 1.0
        branch(y2, x2, ang + rng.uniform(0.3, 0.8), length * 0.72, depth - 1)
        branch(y2, x2, ang - rng.uniform(0.3, 0.8), length * 0.72, depth - 1)

    for a0 in np.linspace(0, 2 * np.pi, 5, endpoint=False):
        branch(N // 2, N // 2, a0, 34, 4)
    return binary_dilation(img > 0, iterations=1).astype(float) * val


def shape_gyri(val=0.85):
    """Folded cortical ribbons: wavy high-frequency bands inside a brain ellipse."""
    y, x = np.mgrid[0:N, 0:N]
    band = np.sin(x / 6.0 + 3.0 * np.sin(y / 22.0))
    ridges = (np.abs(band) > 0.7).astype(float) * val
    mask = blank()
    rr, cc = sk_ellipse(N // 2, N // 2, 72, 58, shape=(N, N))
    mask[rr, cc] = 1.0
    return ridges * mask


def shape_ventricles(brain=0.35, val=0.0):
    """Paired dark curved ventricles on a gray brain."""
    img = blank()
    rr, cc = sk_ellipse(N // 2, N // 2, 72, 58, shape=(N, N))
    img[rr, cc] = brain
    for sgn in (-1, 1):
        rr, cc = sk_ellipse(N // 2, N // 2 + sgn * 20, 30, 8,
                            rotation=np.deg2rad(sgn * 25), shape=(N, N))
        img[rr, cc] = val
    return img


PRIMITIVES = [
    ("disk", shape_disk()), ("ellipse", shape_ellipse()),
    ("annulus", shape_annulus()), ("crescent (SDH)", shape_crescent()),
    ("branching (SAH)", shape_branching()), ("gyri", shape_gyri()),
    ("ventricles", shape_ventricles()),
]


def sino(img):
    return radon(img, theta=np.linspace(0, 180, 180, endpoint=False), circle=False)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 7, figsize=(15.5, 7.2))
    for j, (name, img) in enumerate(PRIMITIVES):
        ax[0, j].imshow(img, cmap="gray"); ax[0, j].set_title(name, fontsize=9)
        ax[0, j].axis("off")

    # row 2: SIZE sweep (the scale-selectivity axis) + a crescent image/sinogram pair
    for j, r in enumerate([8, 16, 24, 32, 40]):
        ax[1, j].imshow(shape_disk(r), cmap="gray")
        ax[1, j].set_title(f"disk r={r}", fontsize=9); ax[1, j].axis("off")
    cr = shape_crescent()
    ax[1, 5].imshow(cr, cmap="gray"); ax[1, 5].set_title("crescent", fontsize=9); ax[1, 5].axis("off")
    ax[1, 6].imshow(sino(cr), cmap="gray", aspect="auto")
    ax[1, 6].set_title("→ its sinogram", fontsize=9); ax[1, 6].axis("off")

    # row 3: a gyri image/sinogram pair + BORDER-THICKNESS sweep
    gy = shape_gyri()
    ax[2, 0].imshow(gy, cmap="gray"); ax[2, 0].set_title("gyri", fontsize=9); ax[2, 0].axis("off")
    ax[2, 1].imshow(sino(gy), cmap="gray", aspect="auto")
    ax[2, 1].set_title("→ its sinogram", fontsize=9); ax[2, 1].axis("off")
    for j, th in enumerate([2, 4, 6, 8, 10]):
        ax[2, 2 + j].imshow(shape_annulus(thick=th), cmap="gray")
        ax[2, 2 + j].set_title(f"border={th}px", fontsize=9); ax[2, 2 + j].axis("off")

    fig.suptitle("Synthetic shape benchmark — controlled probes of scale, border "
                 "thickness, curvature, and shape class (each also Radon-transformable)",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("synthetic_shapes_gallery.png", dpi=140)
    print("wrote synthetic_shapes_gallery.png")


if __name__ == "__main__":
    main()
