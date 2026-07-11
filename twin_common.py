#!/usr/bin/env python3
"""
twin_common.py — shared model, dataset, and split for the twin-CNN kernel study.

Two IDENTICAL CNNs are trained on the SAME slices in two domains — reconstructed
image vs. raw sinogram (Radon transform) — produced from one source image by
data_pipeline.py. The classification task (6-way multilabel ICH) is only a means
to force layer-1 to become a useful feature detector; the object of study is the
first convolutional layer's kernels. Postulate: sinogram-domain layer-1 kernels
should be more anisotropic, more oscillatory/high-pass (rediscovering the FBP
ramp the RAW sinogram lacks), and more detector-axis oriented than recon kernels.

Convention: stored sinogram is skimage.radon output, axis0 = DETECTOR position
(where the ramp acts), axis1 = projection ANGLE. So in a conv kernel weight
[out,in,kH,kW], kH indexes the detector axis and kW the angle axis.
"""
import csv
import hashlib
import os

import numpy as np
import torch
import torch.nn as nn

LABEL_COLS = ["epidural", "intraparenchymal", "intraventricular",
              "subarachnoid", "subdural", "any"]
# cache dir is overridable so the same study can run on a larger tensor set
CACHE = os.environ.get("SINO_CACHE", "/mnt/mlcache/subset50k")

# --- large, interpretable first layer -------------------------------------
K1 = 9          # conv1 kernel size (big enough to show ramp/sidelobe structure)
C1 = 48         # conv1 filters (a population to analyze)


class SmallCNN(nn.Module):
    """Deliberately plain CNN: one large visualizable conv1, then a compact
    downsampling trunk to a 6-logit head. No attention/residual confounds so the
    layer-1 comparison is clean. conv1 geometry (k1,c1,s1) is configurable so the
    same study can be run at different first-layer spatial SCALES."""

    def __init__(self, n_out=6, k1=K1, c1=C1, s1=2, dropout=0.2):
        super().__init__()
        self.k1, self.c1, self.s1 = k1, c1, s1
        self.conv1 = nn.Conv2d(1, c1, k1, stride=s1, padding=k1 // 2, bias=True)
        self.bn1 = nn.BatchNorm2d(c1)
        self.act = nn.ReLU(inplace=True)

        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True))

        self.trunk = nn.Sequential(block(c1, 96), block(96, 192), block(192, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(256, n_out))

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        return self.head(self.trunk(x))


class ResBlock(nn.Module):
    """Basic residual block; kernel size k and dilation configurable (k=3,dil=1 is
    the standard 3x3). Larger k or dilation enlarge the effective receptive field
    to capture the sinogram's spread-out structure."""
    def __init__(self, c, k=3, dilation=1):
        super().__init__()
        pad = dilation * (k // 2)
        self.c1 = nn.Conv2d(c, c, k, padding=pad, dilation=dilation, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, k, padding=pad, dilation=dilation, bias=False)
        self.b2 = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        r = x
        x = self.act(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        return self.act(x + r)


# anisotropic stem cascade: (kH, kW) = (detector rows, angle cols). ANGLE-WIDE:
# short in detector, wide in angle — matched to the trace geometry (a point traces
# a wave running across the FULL angle axis; the image center → a horizontal line
# at constant detector). Aspect 4 -> 3.75 -> 2.5 -> 2, then an isotropic 3x3 ResNet.
ANISO_STEM = [(15, 60), (8, 30), (6, 15), (4, 8)]

# ISOTROPIC control for the ablation: SQUARE kernels with the SAME AREA as each
# anisotropic kernel (√900≈30, √240≈15, √90≈9, √32≈6), same channel widths, same
# body. Equal area → equal parameters, so the ONLY difference vs ANISO_STEM is the
# kernel aspect ratio (shape). Any AUC gap is attributable to anisotropy alone.
ISO_STEM = [(30, 30), (15, 15), (9, 9), (6, 6)]

# aniso2: taller first layer (30 detector, was 15) so a more complete trace CURVE
# fits in the kernel before it exits; both dims halve down the cascade.
ANISO2_STEM = [(30, 60), (15, 30), (8, 15), (4, 8)]

# aniso3: bigger stem throughout to capture more horizontal (angle) trace info in
# the encoding phase; paired with a wider-RF dilated body (2,4,8 -> RF 9,17,33).
ANISO3_STEM = [(30, 60), (16, 30), (12, 20), (8, 12)]


class AnisoStemNet(nn.Module):
    """Sinogram-adapted architecture: a cascade of decreasingly-anisotropic conv
    kernels (tall in the detector axis, narrow in the angle axis — matched to the
    elongated trace geometry) feeding a 3x3 ResNet. The stem convs are exposed as
    self.stem_convs for kernel visualization."""

    def __init__(self, n_out=6, widths=(48, 96, 160, 256), n_res=3, stem_spec=None,
                 stem_strides=None, res_k=3, res_dilations=None):
        super().__init__()
        stem_spec = ANISO_STEM if stem_spec is None else stem_spec
        stem_strides = [2] * len(stem_spec) if stem_strides is None else stem_strides
        layers, self.stem_convs, cin = [], [], 1
        for (kh, kw), cout, st in zip(stem_spec, widths, stem_strides):
            conv = nn.Conv2d(cin, cout, (kh, kw), stride=st,
                             padding=(kh // 2, kw // 2), bias=False)
            self.stem_convs.append(conv)
            layers += [conv, nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            cin = cout
        self.stem = nn.Sequential(*layers)
        res_dilations = [1] * n_res if res_dilations is None else res_dilations
        self.res = nn.Sequential(*[ResBlock(cin, k=res_k, dilation=res_dilations[i])
                                   for i in range(n_res)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.2), nn.Linear(cin, n_out))

    def forward(self, x):
        return self.head(self.res(self.stem(x)))


# --- data ------------------------------------------------------------------
def load_labels():
    per = {}
    with open(os.path.join(CACHE, "labels.csv")) as f:
        for row in csv.DictReader(f):
            per[row["id"]] = [int(row[c]) for c in LABEL_COLS]
    return per


def split_ids(ids, val_every=10):
    """Deterministic 90/10 split by id hash (identical for both domains)."""
    tr, va = [], []
    for i in ids:
        h = int(hashlib.md5(i.encode()).hexdigest(), 16)
        (va if h % val_every == 0 else tr).append(i)
    return sorted(tr), sorted(va)


# each DataLoader worker imports this module fresh -> its own augmentation stream
_AUG_RNG = np.random.default_rng()


def recon_wedge_pick(bank, p=0.5, rng=None):
    """Augment for the iradon-recon domain. `bank` is a (1+K, H, W) stack:
    row 0 = iradon(full sino), rows 1..K = iradon(wedged sino). With prob (1-p)
    return the clean row 0; else a random wedged row. Mirrors angular_dropout(p)
    so the recon branch gets the SAME physical corruption as the sinogram branch —
    the parity fix that makes the AUG factor separable from DOMAIN in the Tier-2
    factorial."""
    rng = rng or _AUG_RNG
    if bank.ndim != 3:                 # not a bank (safety): pass through
        return bank
    if rng.random() > p or bank.shape[0] < 2:
        return bank[0]
    return bank[int(rng.integers(1, bank.shape[0]))]


def angular_dropout(sino, p=0.5, max_frac=0.4, rng=None):
    """Zero a contiguous angular wedge (columns = angle axis) of the RAW sinogram,
    at a random phase and width — trains robustness to lost/garbage projections
    (the motion case). Matches the test-time corruption in angular_robustness.py."""
    rng = rng or _AUG_RNG
    if rng.random() > p:
        return sino
    A = sino.shape[1]
    w = int(rng.uniform(0.05, max_frac) * A)
    if w <= 0:
        return sino
    s = sino.copy()
    start = int(rng.integers(0, A))
    end = start + w
    if end <= A:
        s[:, start:end] = 0.0
    else:                                   # wrap-around wedge
        s[:, start:] = 0.0
        s[:, :end - A] = 0.0
    return s


class DomainDS(torch.utils.data.Dataset):
    def __init__(self, domain, ids, labels, mean, std, augment=None):
        self.dir = os.path.join(CACHE, domain)
        self.ids = ids
        self.labels = labels
        self.mean, self.std = float(mean), float(std)
        self.augment = augment            # callable(raw_arr)->raw_arr, train only

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        stem = self.ids[i]
        arr = np.load(os.path.join(self.dir, stem + ".npy")).astype(np.float32)
        if arr.ndim == 3:                    # variant bank (recon_wedge): (1+K,H,W)
            arr = self.augment(arr) if self.augment is not None else arr[0]
        elif self.augment is not None:
            arr = self.augment(arr)
        x = (arr - self.mean) / self.std
        y = np.asarray(self.labels[stem], dtype=np.float32)
        return torch.from_numpy(x)[None], torch.from_numpy(y)


def domain_stats(domain, ids, n=800, seed=0):
    """Per-domain mean/std over a random sample of slices (standardization)."""
    rng = np.random.default_rng(seed)
    pick = ids if len(ids) <= n else [ids[j] for j in
                                      rng.choice(len(ids), n, replace=False)]
    def _clean(s):
        a = np.load(os.path.join(CACHE, domain, s + ".npy")).astype(np.float32)
        return (a[0] if a.ndim == 3 else a).ravel()   # bank -> clean variant 0
    vals = [_clean(s) for s in pick]
    v = np.concatenate(vals)
    return float(v.mean()), float(v.std() + 1e-6)


def whitening_filter(domain, ids, mean, std, n=400, f_lp=0.8, eps=1e-2, seed=0):
    """Per-domain spectral whitening filter W(f) = lowpass(f) / sqrt(<|F|^2>),
    estimated from a sample of standardized slices. Flattening the average power
    spectrum removes the low-order (pairwise/1-over-f) structure the network would
    otherwise learn first, so learned kernels reflect signal shape rather than a
    generic whitening filter. Returned in the UNSHIFTED fft2 layout; applied on
    GPU in the training loop. The radial low-pass prevents Nyquist noise blow-up."""
    rng = np.random.default_rng(seed)
    pick = ids if len(ids) <= n else [ids[j] for j in
                                      rng.choice(len(ids), n, replace=False)]
    acc, cnt = None, 0
    for s in pick:
        arr = np.load(os.path.join(CACHE, domain, s + ".npy")).astype(np.float32)
        if arr.ndim == 3:                    # bank -> whiten on clean variant 0
            arr = arr[0]
        x = (arr - mean) / std
        p = np.abs(np.fft.fft2(x)) ** 2
        acc = p if acc is None else acc + p
        cnt += 1
    P = acc / cnt
    H, W = P.shape
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2) / 0.5              # Nyquist -> 1
    lp = np.exp(-(r / f_lp) ** 4)
    Wf = lp / np.sqrt(P + eps * P.mean())
    Wf /= np.sqrt((Wf ** 2).mean())                   # unit average power gain
    return Wf.astype(np.float32)
