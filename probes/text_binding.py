"""Is the trained adapter text-conditioned at all?

Same cached latent, same noise, two conditionings: the item's own caption embedding vs the cached
unconditional embedding. Run for the base model and for the best trained adapter.

  adapter loss(caption) << loss(empty)  -> the LoRA learned to listen to text; the trigger carries
                                          signal, and the failure is downstream (inference bridge)
  adapter loss(caption) ~= loss(empty)  -> the LoRA learned a text-independent shift; caption
                                          conditioning never became predictive, which would explain
                                          a character that appears at no strength
The base model row is the control: the base must be text-sensitive or the probe itself is broken.
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
CACHE = pathlib.Path("/home/bloodycrown/train/omni/run/precache_v7/db144b6617fdf77d")

lora_path = None if sys.argv[1] == "none" else sys.argv[1]

arch = arch_mod.ARCHS["krea2"]
index = json.loads((CACHE / "index.json").read_text())

loras = () if lora_path is None else (LoraRef(file=lora_path, strength=1.0),)
tf = loaders.load_krea2_transformer(
    f"{MODELS}/diffusion_models/krea2_raw_bf16.safetensors",
    torch.float16, Quantization.NF4, device="cuda", loras=loras,
)
tf.eval()

uncond = load_file(str(pathlib.Path("/home/bloodycrown/train/omni/run/precache_v5/23e8d8db81e0086a") / "unconditional.safetensors"))

for i in (0, 1, 20):
    sd = load_file(str(CACHE / index["items"][i]))
    latent = sd["latent"].to("cuda", torch.float16)
    noise = torch.randn(latent.shape, generator=torch.Generator(device="cpu").manual_seed(7000 + i)).to("cuda", torch.float16)
    sigma = torch.tensor(0.85, device="cuda", dtype=torch.float16)
    noisy = (1 - sigma) * latent + sigma * noise
    target = arch.target(latent, noise)

    for tag, src in (("caption", sd), ("empty  ", uncond)):
        item = {"embed": src["embed"].to("cuda", torch.float16), "mask": src["mask"].to("cuda")}
        with torch.no_grad():
            pred = arch.forward(tf, noisy, arch.timestep(sigma), item)
            loss = torch.nn.functional.mse_loss(pred.float(), target.float()).item()
        print(f"item {i:>2}  {tag}: {loss:.5f}", flush=True)
