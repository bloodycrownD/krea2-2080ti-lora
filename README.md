# Krea 2 LoRA Training on a GTX 1080Ti-class GPU (Turing, fp16-only)

Train character LoRAs for **Krea 2** (12B DiT) on an **RTX 2080 Ti (22 GB, Turing sm_75)** —
a card with fp16 tensor cores but **no bf16 and no fp8** — at fp16 speed (~5.5 s/step @ 512px),
without switching to emulated bf16 (~5x slower) or a different machine.

This repository is the distilled, reproducible outcome of a 9-run debugging campaign.
The final recipe produced a working character LoRA on the first clean run:

| metric | before (runs v1–v8) | after (run v9) |
|---|---|---|
| training loss vs. base model on its own dataset | always **worse** (1.0–3.6x) | **better** (0.0590 vs 0.0599) |
| trigger-word generation | generic person / style drift, never the character | **character appears** |
| LoRA @ strength 1.0 | purple noise by step ~600 | coherent |

## The root cause (and why it is not the usual suspects)

**99.99% of the LoRA gradient components were smaller than fp16's minimum normal
(6.1e-5) and were flushed to zero/subnormals in the fp16 backward pass** — measured
in-run, not theorized:

```
GRADLOG step 1  zero-frac 0.00002  would-underflow-frac 0.99998  max 0.000220
```

The optimizer only ever saw the loudest ~0.01% of each gradient: global style /
dataset-mean directions. Identity- and caption-conditioned components (small
per-feature signals) never reached the weights. This single mechanism explains
every observed failure mode at once:

- training loss never descended at any learning rate (it rose or flatlined),
- style drift visible, identity never appeared, at any strength, any prompt,
- noise-like artifacts growing with steps (subnormal rounding garbage),
- and why community recipes on bf16-native cards worked on the same data.

The fix is the classic one — **loss scaling** — plus forward-side armor so scaling
can work at all. Full story: [`docs/postmortem.zh.md`](docs/postmortem.zh.md)
(Chinese, detailed). The transferable method for the next model (Krea 3, etc.):
[`docs/methodology.md`](docs/methodology.md).

## The recipe (v9, all three changes at once)

1. **GradScaler-style loss scaling** — backward on `loss * 65536`, unscale before
   clipping/stepping. Post-fix zero-frac: 0.00002. (Requires 2 and 3 as
   preconditions: loss scaling cannot rescue a forward pass that overflows.)
2. **LoRA targets narrowed to attention + FFN** — `to_q/k/v/out.0/gate, ff.up, ff.down`.
   The default "full" scope (464 modules incl. `text_fusion`, `img_in`,
   `final_layer`) hands gradient descent cheap global knobs that get hijacked for
   unconditional style drift before identity ever forms. Community-verified Krea 2
   LoRAs train 264 linears, not 464.
3. **Sigma sampling: shift 2.5** — `sigma = 2.5*u / (1 + 1.5*u)`, u~U(0,1), matching
   the community-verified timestep distribution instead of unshifted logit-normal.

Forward-side fp16 armor (preconditions, all kept from earlier runs):

- **fp32 attention compute** — Krea 2's QK-norm learnable scales reach ~51.9,
  scaling q·k logits ~2700x past fp16's 65504 ceiling (NaN source #1),
- **fp32 residual stream** (block-to-block), fp16 inside matmuls — late-block
  residual accumulation rides at ~30% of the fp16 ceiling and spikes over on
  unlucky samples around step 550–800 at lr 1e-4 (NaN source #2, located
  precisely by a fail-fast run at lr 5e-4: block 7, 52.5k → inf at step 223),
- fp32 LoRA master weights, grad-norm clip 1.0, fp32 AdamW.

Result: ~5.5 s/step at 512px, 13.6 GB VRAM, rank 32 / alpha 32, lr 1e-4, batch 1.

## Repository layout

```
train/            run_lora.py (all patches, heavily commented), manifest example,
                  trainer patch (scaler + GRADLOG)
probes/           measurement instruments - run grad_probe.py FIRST on any new
                  model/stack (fp16 vs bf16 vs fp16+scaler gradient comparison)
eval/             clean-prompt acceptance suite, LoRA format conversion, gate /
                  monitor / checkpoint-rotation utilities
dataset/          tight-crop dataset builder used for the character run (optional)
docs/             postmortem.zh.md (full 9-run failure chain), methodology.md
```

## Quickstart

```bash
# 0) probe the numerics of your stack before training anything
python probes/grad_probe.py --help

# 1) train
python train/run_lora.py train/manifest.example.json

# 2) convert a snapshot for ComfyUI
python eval/convert_lora.py run/snapshots/step-000600.safetensors out.safetensors --rank 32

# 3) judge with CLEAN prompts only (see docs/methodology.md - prompt confounds)
python eval/acceptance.py q21char/q21char_v9_s600.safetensors
```

Paths inside scripts are environment-variable driven with documented defaults —
see the header of each file.

## Status / caveats

- Verified on one character (123 tight-crop images, single GPU, RAW train →
  Turbo inference). Steps beyond 600 (community budget: 2000–2500) pending.
- The 12B base is trained NF4-quantized (bitsandbytes) with fp32 LoRA on top;
  512px is the practical resolution ceiling on 22 GB at this stack (768 needs TE
  quantization, 11.5 s/step).
- OmniChar is the training harness here; the *method* (probe → armor → scale →
  narrow → align sigmas) is harness-agnostic.

## Credits

- Community reference recipe that worked on a 12 GB bf16-native card:
  note.com/sepiablue — 【VRAM 12GB】Krea2 キャラ LoRA 完全ガイド (musubi-tuner).
- OmniChar / DiffSynth-Studio / sd-scripts communities.
