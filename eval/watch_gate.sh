#!/usr/bin/env bash
# Wait for v7 gate N to finish (images rendered + training resumed), then notify by exiting.
# Server layout after 2026-11-07 cleanup: scripts and logs live under /home/bloodycrown/train/omni/.
HOST=bloodycrown@192.168.1.144
OM=/home/bloodycrown/train/omni
N=$1
MAX_MINUTES=${2:-20}
start=$(date +%s)
while true; do
    if [ $(( ($(date +%s) - start) / 60 )) -ge "$MAX_MINUTES" ]; then echo "gate $N: timeout"; exit 0; fi
    out=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" \
        "tail -1 $OM/logs/gate${N}.log 2>/dev/null; ls /home/bloodycrown/Dev/ComfyUI/output/ 2>/dev/null | grep -c v7_s${N}_" 2>/dev/null)
    line=$(echo "$out" | sed -n '1p'); cnt=$(echo "$out" | sed -n '2p')
    if [ "$line" = "gate $N complete" ] && [ "${cnt:-0}" -ge 4 ]; then echo "gate $N: images ready"; exit 0; fi
    sleep 60
done
