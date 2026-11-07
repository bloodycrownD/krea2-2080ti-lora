"""Paired probe in a chosen compute dtype: does the adapter lower the loss when the forward pass
is not fp16?

The fp16 runs diverge, and the probe that called the v1 adapter "3.6x worse than base" also ran in
fp16. If the corruption is in the fp16 forward (activations overflowing as the adapter grows) rather
than in the learning itself, then measuring in bf16 — same range as fp32 — separates the two.

usage: python loss_probe4.py <lora_path_or_none> <strength> <bf16|fp16>
"""

import json
import pathlib
import sys

import torch
from safetensors.torch import load_file

from inline_core.device.policy import Quantization
from inline_core.graph.loader_runners import LoraRef
from inline_core.models import loaders
from inline_core.training import arch as arch_mod

MODELS = "/home/bloodycrown/train/omni/models"
CACHE = "/home/bloodycrown/train/omni/run/precache/4c811d9a4808fa31"

lora_path = None if sys.argv[1] == "none" else sys.argv[1]
strength = float(sys.argv[2])
dtype = torch.bfloat16 if sys.argv[3] == "bf16" else torch.float16

arch = arch_mod.ARCHS["krea2"]
index = json.loads((pathlib.Path(CACHE) / "index.json").read_text())
items = index["items"]

loras = () if lora_path is None else (LoraRef(file=lora_path, strength=strength),)
tf = loaders.load_krea2_transformer(
    f"{MODELS}/diffusion_models/krea2_raw_bf16.safetensors",
    dtype,
    Quantization.NF4,
    device="cuda",
    loras=loras,
)
tf.eval()

losses = []
for i in range(6):
    sd = load_file(f"{CACHE}/{items[i]}")
    latent = sd["latent"].to("cuda", dtype)
    item = {"embed": sd["embed"].to("cuda", dtype), "mask": sd["mask"].to("cuda")}
    g = torch.Generator(device="cpu").manual_seed(7000 + i)
    noise = torch.randn(latent.shape, generator=g).to("cuda", dtype)
    sigma = torch.tensor(0.35 + 0.1 * i, device="cuda", dtype=dtype)
    noisy = (1 - sigma) * latent + sigma * noise
    target = arch.target(latent, noise)
    with torch.no_grad():
        pred = arch.forward(tf, noisy, arch.timestep(sigma), item)
        losses.append(torch.nn.functional.mse_loss(pred.float(), target.float()).item())

tag = "base" if lora_path is None else f"{pathlib.Path(lora_path).name}@{strength}"
print(f"{tag} [{sys.argv[3]}]: loss {sum(losses) / len(losses):.5f}  per-item {[round(x, 4) for x in losses]}",
      flush=True)
