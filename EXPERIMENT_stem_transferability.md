# Experiment: Transferability of the sinogram stem — ResNet → Transformer

**Status:** code ready (2026-07-14) — `train_stem_transfer.py`, launch via
`~/sino_work/run_arm{1,2,3}.sh`; smoke-tested, training deferred to 2026-07-15 evening.
**Machine:** DGX Spark. **Authors:** J. Bramble, MD + Claude (Anthropic Fable 5).

> **Implemented graft (differs from the generic token diagram below):** the
> transformer is timm **MaxViT-tiny** (reusing the ich-maxvit choice). The arm0
> stem (ANISO2, widths 48→256) deploys at the existing `--fulres` strides
> (2,2,2,1) → 34×34×256, center-cropped to **32×32** — exactly MaxViT stage-2's
> input grid at img_size 256 — then a learnable 1×1 conv (256→128, same fixed
> init in all arms) feeds **stages 2–3 + head** (7 of 11 blocks). MaxViT's
> relative-position attention replaces the explicit positional encoding.
> Transformer side is initialized once per `--tf-seed` (`tf_init_s0.pt`) and
> bit-identical across arms; verified by hash in the smoke test. Source stem:
> arm0 best.pt, val macro-AUC 0.8714 (epoch 22/30, 2026-07-14).

## Question

Does the anisotropic **stem** (the early *encoding / input* conv layers), trained end-to-end
as the front-end of a ResNet on sinograms, transfer as the front-end for a **transformer**?

> Terminology: "stem" = the input/encoding layers. NOT "top layers" — in DL "top" = the
> output head (Keras `include_top`). Say *stem / encoder / front-end*.

## Hypothesis (genuinely open)

Geometry-matched low-level sinogram features are a property of the **domain** (Radon-space
structure), not the downstream head → they *should* transfer across architecture families.
But the stem was co-trained with the ResNet, so some features may be ResNet-specific and
fail to transfer. Either outcome is informative. Working bet: **Arm 2 (fine-tuned)** most
likely to succeed.

## Design — the isolating comparison

Compare *transfer* vs *no-transfer* with **everything else identical**. Only the stem's
initialization (and frozen/trainable state) varies.

| Arm | Stem weights | Stem during transformer training |
|-----|--------------|----------------------------------|
| 1 | pretrained (from ResNet) | frozen |
| 2 | pretrained (from ResNet) | fine-tuned (unfrozen after warm-up) |
| 3 (control) | random init, **same architecture** | trained from scratch |

**Critical comparison: Arm 1/2 vs Arm 3.** Same stem *architecture*, same
transformer / tokenizer / recipe / data / split. The ONLY difference is pretrained vs
random stem weights. Pretrained > random ⇒ the stem transfers.

## What to measure — the learning CURVE, not just final accuracy

Transfer shows up as **faster convergence** and **better data-efficiency**, and often
**vanishes at large N / long training**. So:

- Run as a **data-size sweep** (reuse existing methodology): each arm at several N
  (e.g. 10k, 25k, 50k, 100k, 200k).
- Track **accuracy vs epoch** and **accuracy vs N**.
- The transfer signal is the **gap at small N / early epochs**. Comparing only final
  accuracy at full data can hide a transfer that saved most of the data.
- Also track **training stability** — divergence rate, sensitivity to LR / warmup. If the
  pretrained stem makes training more forgiving, that is a distinct, reportable practical
  win, separate from final accuracy.

## Controls

- Fix the **transformer's init (same seed)** across arms → the stem is the only variable.
- **Multi-seed** each arm (reuse the go-for-broke multi-seed harness); compare curves
  against the seed spread to separate real gaps from seed noise.
- Hold constant: transformer, tokenizer interface, positional encoding, optimizer/recipe,
  data, split, input size.

## Stem → transformer interface

```
stem(sinogram)             -> feature map   [C, H, W]     # frozen (arm 1) or trainable (arms 2,3)
reshape / flatten spatial  -> token seq     [N=H*W, C]    # NOT learnable (a reshape)
linear proj  C -> D  (shared per token)     [N, D]        # learnable FC == 1x1 conv;
                                                          #   do NOT flatten to a single token
+ positional encoding       [N, D]                        # learnable or sinusoidal;
                                                          #   consider PERIODIC on the angle (theta) axis
(+ optional learnable [CLS] token)
-> transformer encoder -> classification head
```

Keep the `H*W` tokens separate — a single FC that flattens the whole feature map collapses
it to one token and defeats the transformer.

## Staged training

- **Stage 1** — freeze stem, warm up the transformer. Protects the pretrained features from
  the random transformer's early large gradients. A frozen stem's output is deterministic,
  so **precompute + cache the stem features once** and train the transformer on the cached
  features → large speed/memory win, pairs with the Zig loader.
- **Stage 2** — unfreeze stem, low-LR end-to-end fine-tune (gives up the cache; lets the
  stem co-adapt to attention). This is **Arm 2**.

## Practical contribution (translational angle)

If Arm 1/2 succeeds: a **pretrained, geometry-matched stem makes sinogram-transformers
cheaper/easier to train** — train the easy CNN first, bootstrap the hard transformer.
Connects to the known result that conv stems stabilize transformer training (Xiao et al.
2021, *Early Convolutions Help Transformers See Better*). Novelty here = a *pretrained,
domain-specific* stem transferring *across architecture families*, plus a **reusable
front-end** others can grab. Biggest win in the **low-data / low-compute** regime. Honest
boundary: this is a domain-specific demonstration + reusable resource, not a new ML mechanism.

## Notes

- **ICH-MaxViT** (image domain) is **NOT a controlled arm** — comparing it to a different
  sinogram architecture confounds *domain* with *architecture*. Watch-only; it can serve as
  an *informal* reference for the training-ease claim ("transformer with a *from-scratch*
  conv stem" vs our "*pretrained geometry-matched* stem").
- **Input size:** SinogramKernels pipeline = **256**; ich-maxvit likely **224**. Confirm with
  `np.load(path).shape` — don't guess. Monochrome = **1 channel**.
- **DGX Spark memory gotcha:** the unified-memory system monitor misreports GPU allocation.
  Find batch size **empirically** (ramp to OOM, back off), watch `torch.cuda.memory_reserved()`,
  try `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and use gradient accumulation if the
  physical batch must shrink.
- Reuse existing infra: `twin_common.py` / `train_twin.py`, the data-size sweep, the
  multi-seed harness.
