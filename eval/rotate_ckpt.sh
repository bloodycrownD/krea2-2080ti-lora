#!/usr/bin/env bash
# Rotate the single-checkpoint dir into a timestamped history, keep newest 5.
# OmniChar overwrites checkpointDir at every cooperative stop; without rotation any
# resume bug destroys the only optimizer state (this already cost us 658 steps once).
CKPT=/home/bloodycrown/train/omni/run/ckpt_v8
HIST=/home/bloodycrown/train/omni/run/ckpt_v8_history
KEEP=5

[ -f "$CKPT/step.json" ] || { echo "rotate: no live checkpoint, skip"; exit 0; }
step=$(python3 -c "import json;print(json.load(open('$CKPT/step.json'))['step'])" 2>/dev/null || echo "?")
mkdir -p "$HIST"
dst="$HIST/ckpt_step${step}_$(date +%m%d_%H%M%S)"
cp -r "$CKPT" "$dst"
echo "rotate: saved $dst"
ls -1dt "$HIST"/ckpt_* 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -rf
echo "rotate: keeping $(ls -1d "$HIST"/ckpt_* 2>/dev/null | wc -l) history checkpoints"
