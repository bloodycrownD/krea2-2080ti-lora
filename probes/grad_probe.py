"""Gradient-underflow probe: is your fp16 backward destroying LoRA gradients?

THE first instrument to run on any new model / training stack on a no-bf16 GPU.
Runs ONE batch's forward+backward three times - fp16 (your fast path), bf16
(ground truth), fp16 with loss scaling - then compares zero-fraction and cosine
similarity of the LoRA gradients. The one-arm-per-process layout avoids OOM.

usage:
    python grad_probe.py fp16   ADAPTER  [ITEM_INDEX]
    python grad_probe.py bf16   ADAPTER  [ITEM_INDEX]
    python grad_probe.py fp16sc ADAPTER  [ITEM_INDEX]   # loss * 65536 backward
    python grad_probe.py compare

environment:
    OMNI_CORE  OmniChar inline_core source root   (default: $OMNI_CORE)
    K2_MODELS  models dir with diffusion_models/krea2_raw_bf16.safetensors
    K2_CACHE   a precache dir (index.json + item tensors) from any run
    TMPDIR     grad scratch dir                    (default /tmp)

interpretation (docs/methodology.md step 0):
    fp16-vs-bf16 cosine > 0.98 and zero-frac < 1%  -> numerics healthy
    would-underflow-frac large (ours: 0.9999)      -> add loss scaling NOW
"""

import json
import os
import pathlib
import sys

_CORE = os.environ.get("OMNI_CORE", "")
if _CORE:
    sys.path.insert(0, _CORE)
import torch
from safetensors.torch import load_file

MODELS = os.environ.get("K2_MODELS", "/home/bloodycrown/train/omni/models")
CACHE = os.environ.get("K2_CACHE",
                       "/home/bloodycrown/train/omni/run/precache_v7/db144b6617fdf77d")
_TMP = os.environ.get("TMPDIR", "/tmp")

arm = sys.argv[1]
ADAPTER = sys.argv[2] if len(sys.argv) > 2 else "none"
if arm == "compare":
    gs = {a: torch.load(f"{_TMP}/grad_{a}.pt") for a in ("fp16", "bf16", "fp16sc")}
    for tag in ("fp16", "fp16sc"):
        zeros = sub = elems = 0
        for t in gs[tag].values():
            elems += t.numel()
            zeros += int((t == 0).sum())
            sub += int(((t != 0) & (t.abs() < 6.1e-5)).sum())
        print(f"{tag}: zero-frac {zeros/elems:.5f}  subnormal-frac {sub/elems:.6f}", flush=True)
    for tag in ("fp16", "fp16sc"):
        names = sorted(set(gs[tag]) & set(gs["bf16"]))
        dot = norm = 0.0
        worst = (1.0, "")
        for n in names:
            a, b = gs[tag][n].flatten(), gs["bf16"][n].flatten()
            d = float((a * b).sum())
            na = float(a.norm()) * float(b.norm()) + 1e-30
            dot += d
            norm += na
            c = d / na
            if c < worst[0]:
                worst = (c, n)
        print(f"{tag} vs bf16: global cosine {dot/norm:.4f}   worst {worst[1]} {worst[0]:.4f}", flush=True)
    sys.exit(0)

from diffusers.models.transformers import transformer_krea2 as tk  # noqa: E402

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

import inline_core.training.arch as arch_mod  # noqa: E402
from inline_core.device.policy import Quantization  # noqa: E402
from inline_core.graph.loader_runners import LoraRef  # noqa: E402
from inline_core.models import loaders  # noqa: E402

arch = arch_mod.ARCHS["krea2"]
index = json.loads((pathlib.Path(CACHE) / "index.json").read_text())
sd = load_file(f"{CACHE}/{index['items'][3]}")

dtype = torch.bfloat16 if arm == "bf16" else torch.float16
scale = 65536.0 if arm == "fp16sc" else None

tf = loaders.load_krea2_transformer(
    f"{MODELS}/diffusion_models/krea2_raw_bf16.safetensors",
    dtype, Quantization.NF4, device="cuda",
    loras=(LoraRef(file=ADAPTER, strength=1.0),),
)
tf.train()
if hasattr(tf, "enable_gradient_checkpointing"):
    tf.enable_gradient_checkpointing()
lora_params = {n: p for n, p in tf.named_parameters() if "lora_" in n and p.requires_grad}
latent = sd["latent"].to("cuda", dtype)
item = {"embed": sd["embed"].to("cuda", dtype), "mask": sd["mask"].to("cuda")}
g = torch.Generator(device="cpu").manual_seed(1234)
noise = torch.randn(latent.shape, generator=g).to("cuda", dtype)
sigma = torch.tensor(0.55, device="cuda", dtype=dtype)
noisy = (1 - sigma) * latent + sigma * noise
target = arch.target(latent, noise)
pred = arch.forward(tf, noisy, arch.timestep(sigma), item)
loss = torch.nn.functional.mse_loss(pred.float(), target.float())
if scale is not None:
    (loss * scale).backward()
else:
    loss.backward()
grads = {n: p.grad.detach().float().cpu() for n, p in lora_params.items() if p.grad is not None}
torch.save(grads, f"{_TMP}/grad_{arm}.pt")
print(f"{arm}: loss {loss.item():.5f}  saved {len(grads)} grad tensors", flush=True)
