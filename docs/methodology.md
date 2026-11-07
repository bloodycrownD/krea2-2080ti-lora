# Methodology: training a large DiT LoRA on a no-bf16 GPU

The Krea 2 recipe in this repo is one instance of a general procedure. When the
next model arrives (Krea 3, or any 10B+ flow-matching DiT) on the same Turing
card, run the same ladder. Each step is cheap; each step kills a class of
failure before it costs you a training run.

## Step 0 — Probe gradients before training anything

`probes/grad_probe.py` runs ONE batch's forward+backward three times:
fp16 (your fast path), bf16 (ground truth on this card, slow but exact), and
fp16 + loss scaling. It reports the zero fraction and subnormal fraction of the
LoRA gradients and their cosine similarity against the bf16 ground truth.

Interpretation:

| observation | meaning |
|---|---|
| fp16 vs bf16 cosine > 0.98, zero-frac < 1% | numerics healthy, look elsewhere |
| large `would-underflow-frac` (ours: 0.9999) | **backward is destroying signal — add loss scaling** |
| NaN/inf in backward | forward overflow — go to step 1 first; scaling makes overflow worse |

On a 12B model, LoRA gradient magnitudes land in the 1e-6..1e-4 range; fp16's
minimum normal is 6.1e-5. This is not an edge case — it is the default situation
for LoRA on big DiTs in fp16. bf16-native cards never see it.

## Step 1 — Kill the forward overflow sites (preconditions for scaling)

fp16 overflows at 65504. On Krea 2 there were exactly two classes of site,
found by a fail-fast run (5x learning rate + per-module overflow hooks, which
prints the module name the first time any activation goes non-finite):

1. **Attention QK-norm** — learnable per-channel scales multiply the q·k logits.
   With scales ~52 the product inflates ~2700x. Fix: compute attention in fp32,
   cast in/out to fp16. Cost: ~25% step time.
2. **Residual stream** — 28 blocks of accumulation ride at ~30% of the ceiling;
   LoRA drift pushes unlucky samples over around step 550–800 at lr 1e-4.
   Fix: fp32 between blocks (residual adds, modulation), fp16 inside the big
   matmuls. Cost: ~0%.

A loss scaler cannot rescue a forward that produces inf — do this first.

## Step 2 — Scale the backward

`loss * 65536` → backward → unscale → clip(1.0) → step. Verify with the probe:
post-scaling zero-frac should collapse to the structural baseline (ours: 2e-5).

## Step 3 — Narrow the LoRA target set

Default "full" scopes attach to every Linear, including input/output projections
and text-fusion blocks. Those are cheap global knobs; gradient descent exploits
them first and learns unconditional style/bias drift, starving the identity
signal in the attention/FFN cores. Match the community-verified set for the
model (Krea 2: 264 linears = attention projections + FFN up/down).

Symptom signature of a hijacked scope: style drifts, identity never appears,
weights concentrated in early blocks / non-attention modules, loss rises.

## Step 4 — Align the training sigma distribution with the verified recipe

Flow-matching models learn global structure (who is in the image) at high noise
and detail at low noise. If your harness samples unshifted logit-normal but the
community recipe uses `shift` sampling (Krea 2: discrete_flow_shift 2.5), the
training budget lands on the wrong noise levels. One line to change; verified
recipes exist per model.

## Step 5 — Calibrate expectations BEFORE judging a run

Healthy character-LoRA evolution (lr 1e-4, batch 1, ~100 images):

- 0–500 steps (1–4 epochs): global appearance bleed — palette, hair, garment
  type. Imperfect, often malformed. **This is what "learning" looks like.**
- 500–1500: signature features consolidate.
- 1500+: refinement, then overfitting risk.

Death signatures, distinct from "too early":

- zero change at every strength → nothing learned (underflow / lr too small),
- change but wrong person + pure style drift → unconditional offset
  (scope hijack / captions never earned gradient),
- **noise at strength 1.0 → pathological, always.** Strength 1.0 is the design
  operating point; a healthy LoRA never disintegrates there at any step count.

## Step 6 — Judge only with clean prompts + no-LoRA controls

Two confounds bit us repeatedly:

- Any attribute word in the prompt ("helmet") lets the BASE model draw that
  attribute itself; the LoRA's contribution is invisible without a same-seed
  no-LoRA control.
- On distilled (cfg=1) models, every extra prompt token is an active pull toward
  base-model priors ("standing, full body, white background" = stock-photo mode)
  and dilutes the trigger word. **A bare trigger word gives maximum likeness**;
  scene words trade likeness for scene.

The acceptance suite (`eval/acceptance.py`) embodies this: bare trigger word at
several strengths + one no-LoRA control, same seed, nothing else.

## Instrument checklist

| instrument | answers | cost |
|---|---|---|
| grad_probe (3-arm) | is the backward destroying signal? | minutes, no training |
| fail-fast run (5x lr + overflow hooks) | where does the forward overflow? | ~30 min |
| text-binding probe | does the adapter read captions at all? | minutes |
| loss probe | does the adapter fit the training set better than the base? | minutes |
| bare-trigger acceptance | is identity really bound to the trigger? | minutes |
