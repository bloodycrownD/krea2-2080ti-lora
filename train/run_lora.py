"""Krea 2 LoRA trainer for fp16-only GPUs (Turing): the v9 working recipe.

Usage:
    python run_lora.py <manifest.json>

Environment:
    OMNI_CORE   path to OmniChar's inline_core source root
                (the `inline_core` package must be importable; this env var is
                 prepended to sys.path when set)

The manifest (see manifest.example.json) carries every path: dataset, models,
checkpoints, snapshots, hyperparameters. This launcher applies the numerics
patches that made character training work on a 2080 Ti:

  A. fp32 attention compute   - Krea 2 QK-norm learnable scales reach ~52, the
     q.k logits inflate ~2700x and overflow fp16 (>65504) into softmax NaN.
  B. fp32 residual stream between blocks (fp16 inside matmuls) - 28 blocks of
     residual accumulation ride at ~30% of the fp16 ceiling; LoRA drift pushes
     unlucky samples over around step 550-800 at lr 1e-4 (fail-fast located:
     block 7, 52.5k -> inf at step 223 under lr 5e-4).
  C. LoRA targets narrowed to attention+ff - the default "full" scope (464
     modules incl. text_fusion / img_in / final_layer) gives gradient descent
     cheap global knobs that get hijacked for unconditional style drift before
     identity ever forms. Community-verified Krea 2 LoRAs train 264 linears.
  D. sigma sampling shift 2.5 (2.5*u/(1+1.5*u), u~U(0,1)) - aligns training
     noise levels with the community-verified timestep distribution.
  E. activation-overflow watchdog - per-step non-finite scan of every module
     output; on first hit prints the module name and exits cooperatively.

The backward-side fix (loss scaling + GRADLOG gradient telemetry) lives in the
trainer itself; apply it with patch_scaler.py once per OmniChar checkout.
Background: docs/postmortem.zh.md, docs/methodology.md.
"""

import os
import signal
import sys
from collections import deque

if os.environ.get("OMNI_CORE"):
    sys.path.insert(0, os.environ["OMNI_CORE"])

import torch
from diffusers.models.transformers import transformer_krea2 as tk

import inline_core.training.models as models

models.compute_dtype = lambda: torch.float16

# --- patch C: narrow Krea2 LoRA targets to attention + ff only ---
import inline_core.training.arch as arch_mod  # noqa: E402

_krea = arch_mod.ARCHS["krea2"]
_NARROW = ["to_q", "to_k", "to_v", "to_out.0", "to_gate", "ff.up", "ff.down"]
object.__setattr__(_krea, "target_modules", _NARROW)

# --- patch D: shift-2.5 sigma sampling (verified recipe) ---
def _shifted_sigma(device, shift):
    u = torch.rand((), device=device)
    return 2.5 * u / (1.0 + 1.5 * u)

object.__setattr__(_krea, "sigma", _shifted_sigma)

# --- patch A: fp32 attention ---
_orig_dispatch = tk.dispatch_attention_fn


def _fp32_attention(query, key, value, *args, **kwargs):
    if query.dtype == torch.float16:
        mask = kwargs.get("attn_mask")
        if mask is not None and mask.is_floating_point() and mask.dtype != torch.float32:
            kwargs["attn_mask"] = mask.float()
        out = _orig_dispatch(query.float(), key.float(), value.float(), *args, **kwargs)
        return out.to(torch.float16)
    return _orig_dispatch(query, key, value, *args, **kwargs)


tk.dispatch_attention_fn = _fp32_attention

# --- patch B: fp32 residual skeleton ---
def _fp32res_block_forward(self, hidden_states, temb, image_rotary_emb, attention_mask=None):
    modulation = temb.float().unflatten(-1, (6, -1)) + self.scale_shift_table.float()
    prescale, preshift, pregate, postscale, postshift, postgate = modulation.unbind(-2)
    x = hidden_states.float()
    attn_in = (1.0 + prescale) * self.norm1(x) + preshift
    attn_out = self.attn(attn_in.to(torch.float16), attention_mask=attention_mask,
                         image_rotary_emb=image_rotary_emb)
    x = x + pregate * attn_out.float()
    ff_in = (1.0 + postscale) * self.norm2(x) + postshift
    ff_out = self.ff(ff_in.to(torch.float16))
    x = x + postgate * ff_out.float()
    return x


tk.Krea2TransformerBlock.forward = _fp32res_block_forward

# --- patch E: activation-overflow watchdog ---
_orig_fwd = _krea.forward
STATE = {"step": 0, "hooked": set(), "hit": None, "blockmax": {}}


def _hook_for(name):
    def hook(module, inputs, output):
        outs = output if isinstance(output, (tuple, list)) else (output,)
        for t in outs:
            if not torch.is_tensor(t) or not t.is_floating_point() or t.numel() == 0:
                continue
            finite = torch.isfinite(t)
            if not bool(finite.all()):
                if STATE["hit"] is None:
                    m = float("nan")
                    if bool(finite.any()):
                        m = float(t[finite].abs().max())
                    STATE["hit"] = (name, STATE["step"], m)
                return
            mx = float(t.abs().max())
            if name in STATE["blockmax"]:
                STATE["blockmax"][name] = max(STATE["blockmax"][name], mx)
            elif len(STATE["blockmax"]) < 64 and name.count(".") <= 1:
                STATE["blockmax"][name] = mx
    return hook


def _install(tf):
    for name, mod in tf.named_modules():
        if name:
            mod.register_forward_hook(_hook_for(name))


def _fwd(tf, *args, **kwargs):
    if id(tf) not in STATE["hooked"]:
        _install(tf)
        STATE["hooked"].add(id(tf))
    STATE["step"] += 1
    out = _orig_fwd(tf, *args, **kwargs)
    if STATE["hit"] is not None:
        name, step, mx = STATE["hit"]
        print(f"ACTOVERFLOW step {step} module {name} finite-max {mx}", flush=True)
        print(f"ACTOVERFLOW-DIE first non-finite output at {name}", flush=True)
        os.kill(os.getpid(), signal.SIGTERM)
        return out
    if STATE["step"] % 25 == 0:
        top = sorted(STATE["blockmax"].items(), key=lambda kv: -kv[1])[:5]
        print(f"ACTTRAJ step {STATE['step']} top " + ", ".join(f"{n}={v:.0f}" for n, v in top), flush=True)
        STATE["blockmax"] = {}
    return out


object.__setattr__(_krea, "forward", _fwd)

from inline_core.training.__main__ import main  # noqa: E402,E408

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
