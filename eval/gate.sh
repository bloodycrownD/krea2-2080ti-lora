#!/usr/bin/env bash
# v7 gate N: wait for snapshot N, pause training, convert, render gate images, resume training.
set -u
STEP=$1
SNAP=/home/bloodycrown/train/omni/run/snapshots_v7/step-$(printf '%06d' "$STEP").safetensors
LORA_OUT=/home/bloodycrown/Dev/ComfyUI/models/loras/q21char/q21char_v7_s${STEP}.safetensors
VENV=/home/bloodycrown/train/venv/bin/python
export PYTHONPATH=/home/bloodycrown/train/omni_libs/OmniChar/core/src HF_ENDPOINT=https://hf-mirror.com PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "waiting for $SNAP"
while [ ! -f "$SNAP" ]; do sleep 20; done
sleep 8

pkill -TERM -f omni_fp16_attn32
for _ in $(seq 1 30); do pgrep -f omni_fp16_attn32 >/dev/null || break; sleep 3; done
sleep 4
echo "training paused at step $STEP"

"$VENV" /tmp/convert_snap.py "$SNAP" "$LORA_OUT" || { echo "convert failed"; exit 1; }

systemctl --user start comfyui-cu130
sleep 45
"$VENV" /home/bloodycrown/train/omni/scripts/submit_gate.py "$STEP"
systemctl --user stop comfyui-cu130
sleep 6
echo "images done, resuming training"

cd /home/bloodycrown/train/omni
setsid nohup "$VENV" /home/bloodycrown/train/omni/scripts/omni_fp16_attn32.py /home/bloodycrown/train/omni/scripts/omni_manifest_v7r.json > /home/bloodycrown/train/omni/logs/omni_v7.log 2>&1 < /dev/null &
sleep 2
setsid nohup "$VENV" /home/bloodycrown/train/omni/scripts/train_monitor_v6.py /home/bloodycrown/train/omni/logs/omni_v7.log /home/bloodycrown/train/omni/run/snapshots_v7 120 > /home/bloodycrown/train/omni/logs/omni_v7_monitor.log 2>&1 < /dev/null &
sleep 60
tail -1 /home/bloodycrown/train/omni/logs/omni_v7_monitor.log
echo "gate $STEP complete"
