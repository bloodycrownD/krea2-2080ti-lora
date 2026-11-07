"""Pure-trigger binding test: prompt is ONLY the trigger word (no identity words at all).
Anything tactical that appears with the LoRA and is absent without it = LoRA's own contribution.
"""

import json
import time
import urllib.request
import uuid

BASE = "http://127.0.0.1:8189"
LORA = "q21char/q21char_v8_s600.safetensors"
JOBS = [
    ("q21char", "pt600_lora_s050", 0.5),
    ("q21char", "pt600_lora_s100", 1.0),
    ("q21char", "pt600_nolora", None),
    ("a woman", "pt600_woman_lora_s050", 0.5),
]


def graph(prompt, prefix, strength, seed=20261110):
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
    while time.time() - t0 < 240:
        h = json.load(urllib.request.urlopen(f"{BASE}/history/{pid}"))
        if pid in h and h[pid].get("status", {}).get("completed"):
            print("done", prefix, flush=True)
            break
        time.sleep(3)
    else:
        print("timeout", prefix, flush=True)
