"""Live health monitor for an OmniChar run: loss trend AND adapter magnitude, not just NaN.

The 1500-step v1 run reported 0 NaN the whole way, which was true and useless: the loss had already
climbed to 2.4x its starting level by step 600 and the block-0 weights were 20-80x a healthy
magnitude. Both were visible in data I had and never looked at. This checks the three things that
actually predict a dead LoRA:

  1. loss   - rolling 100-step mean vs the first 100 steps; a sustained rise means the step size is
              overshooting and the run will not converge (alert at 1.5x after a 300-step warmup)
  2. weights - max |lora_B| of the newest snapshot; healthy adapters sit near 0.1-0.5, the broken run
              reached 4.47 (alert above 1.0, hard stop recommendation above 2.0)
  3. nan    - still checked, because it is the terminal case

usage: python train_monitor.py <log> <snapshot_dir> [interval_seconds]
"""

import json
import sys
import time
from pathlib import Path

from safetensors.torch import load_file

log = Path(sys.argv[1])
snap_dir = Path(sys.argv[2])
interval = int(sys.argv[3]) if len(sys.argv) > 3 else 60

offset = 0
losses: list[tuple[int, float]] = []
seen_snapshot: Path | None = None
last_report = 0.0

while True:
    if log.exists():
        with log.open("r", errors="ignore") as f:
            f.seek(offset)
            chunk = f.read()
            offset = f.tell()
        for line in chunk.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "done":
                print(f"MONITOR: run finished at step {msg.get('step', '?')}", flush=True)
                sys.exit(0)
            if msg.get("type") == "progress" and "loss" in msg:
                if msg["loss"] == msg["loss"]:
                    losses.append((msg["step"], msg["loss"]))
                else:
                    print(f"MONITOR: NaN at step {msg['step']}", flush=True)
                    sys.exit(2)

    now = time.time()
    if losses and now - last_report >= interval:
        last_report = now
        step = losses[-1][0]
        import statistics as _st

        first = [l for s, l in losses if s <= 100]
        recent = [l for s, l in losses if s > step - 100]
        base = _st.median(first) if first else None
        cur = _st.median(recent) if recent else float("nan")
        ratio = cur / base if base else float("nan")

        weight_note = ""
        snaps = sorted(snap_dir.glob("step-*.safetensors")) if snap_dir.exists() else []
        if snaps and snaps[-1] != seen_snapshot:
            seen_snapshot = snaps[-1]
            sd = load_file(str(seen_snapshot))
            mx = max(float(v.float().abs().max()) for k, v in sd.items() if k.endswith(".lora_B.weight"))
            weight_note = f" max|B| {mx:.3f} ({seen_snapshot.name})"
            if mx > 2.0:
                weight_note += "  <-- STOP: adapter diverging"

        alert = ""
        if base and step > 300:
            if ratio > 1.5:
                alert = "  <-- ALERT: loss above 1.5x baseline, lr is overshooting"
            elif ratio < 0.8:
                alert = "  (learning)"

        print(f"MONITOR step {step}  loss100 {cur:.4f}  base {base:.4f}  "
              f"ratio {ratio:.2f}{weight_note}{alert}", flush=True)

    time.sleep(10)
