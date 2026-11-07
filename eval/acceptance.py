"""Clean-prompt acceptance suite: judge a character LoRA without prompt confounds.

Bare trigger word at several strengths + one no-LoRA control, same seed.
Any attribute word in a prompt lets the BASE model draw that attribute itself,
and on distilled (cfg=1) models every extra token pulls toward base priors and
dilutes the trigger - so likeness is judged on the bare-trigger images only
(docs/methodology.md step 6).

usage: python acceptance.py <lora_name_in_comfy> [trigger] [strengths...]
       python acceptance.py q21char/q21char_v9_s600.safetensors q21char 0.5 0.8 1.0
env:   COMFY_URL  (default http://127.0.0.1:8189)
       SEED       (default 20261110)
"""

import json
import os
import sys
import time
import urllib.request
import uuid

BASE = os.environ.get("COMFY_URL", "http://127.0.0.1:8189")
SEED = int(os.environ.get("SEED", "20261110"))
LORA = sys.argv[1]
TRIGGER = sys.argv[2] if len(sys.argv) > 2 else "q21char"
TRIG = TRIGGER
FB = f"{TRIGGER}, standing three-quarter view, full body, plain white background."
JOBS = [
    (TRIG, "acc_trig_s080", 0.8),
    (TRIG, "acc_trig_s100", 1.0),
    (FB, "acc_fb_s050", 0.5),
    (FB, "acc_fb_s080", 0.8),
    (FB, "acc_fb_s100", 1.0),
    (TRIG, "acc_trig_nolora", None),
]


def graph(prompt, prefix, strength, seed=SEED):
    g = {}
    g["10"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "krea2_turbo_int8_convrot.safetensors", "weight_dtype": "default"}}
    g["11"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_bf16.safetensors", "type": "krea2", "device": "default"}}
    g["12"] = {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}}
    g["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["11", 0]}}
    g["13"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}}
    g["5"] = {"class_type": "EmptyLatentImage", "inputs": {"width": 896, "height": 1152, "batch_size": 1}}
    model = ["10", 0]
    if strength is not None:
        g["15"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["10", 0], "lora_name": LORA, "strength_model": strength}}
        model = ["15", 0]
    g["3"] = {"class_type": "KSampler", "inputs": {"model": model, "seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "positive": ["6", 0], "negative": ["13", 0], "latent_image": ["5", 0], "denoise": 1.0}}
    g["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["12", 0]}}
    g["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return g


pids = []
for prompt, prefix, st in JOBS:
    data = json.dumps({"prompt": graph(prompt, prefix, st), "client_id": uuid.uuid4().hex}).encode()
    req = urllib.request.Request(BASE + "/prompt", data=data, headers={"Content-Type": "application/json"})
    pids.append((prefix, json.load(urllib.request.urlopen(req))["prompt_id"]))

for prefix, pid in pids:
    t0 = time.time()
    while time.time() - t0 < 300:
        h = json.load(urllib.request.urlopen(f"{BASE}/history/{pid}"))
        if pid in h and h[pid].get("status", {}).get("completed"):
            print("done", prefix, flush=True)
            break
        time.sleep(3)
    else:
        print("timeout", prefix, flush=True)
