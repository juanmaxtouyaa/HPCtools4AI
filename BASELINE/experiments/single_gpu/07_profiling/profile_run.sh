#!/bin/bash

set -euo pipefail

OUTDIR="$1"
PROFILE_CSV="$OUTDIR/system_profile.csv"

mkdir -p "$OUTDIR"

echo "timestamp,gpu_uuid,gpu_util_percent,gpu_memory_mib,power_w,temperature_c,sm_clock_mhz,cpu_rss_kib" \
    > "$PROFILE_CSV"

python train.py \
    --max-train-samples 32768 \
    --epochs 1 \
    --batch-size 160 \
    --learning-rate 3e-5 \
    --dataloader-num-workers 0 \
    --bf16 \
    --skip-save \
    --output-dir "$OUTDIR" &

TRAIN_PID=$!

echo "Training PID: $TRAIN_PID"

GPU_UUID=""

while kill -0 "$TRAIN_PID" 2>/dev/null; do

    if [[ -z "$GPU_UUID" ]]; then
        GPU_UUID="$(
            nvidia-smi \
                --query-compute-apps=pid,gpu_uuid \
                --format=csv,noheader,nounits 2>/dev/null \
            | awk -F', ' -v pid="$TRAIN_PID" \
                '$1 == pid {print $2; exit}'
        )" || true

        if [[ -n "$GPU_UUID" ]]; then
            echo "Detected training GPU UUID: $GPU_UUID"
        fi
    fi

    TIMESTAMP="$(date --iso-8601=seconds)"

    CPU_RSS="$(
        awk '/VmRSS:/ {print $2}' \
            "/proc/$TRAIN_PID/status" 2>/dev/null \
        || echo ""
    )"

    if [[ -n "$GPU_UUID" ]]; then
        GPU_STATS="$(
            nvidia-smi \
                -i "$GPU_UUID" \
                --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm \
                --format=csv,noheader,nounits 2>/dev/null \
            || echo ",,,,"
        )"

        echo "$TIMESTAMP,$GPU_UUID,$GPU_STATS,$CPU_RSS" \
            >> "$PROFILE_CSV"
    else
        echo "$TIMESTAMP,NA,NA,NA,NA,NA,NA,$CPU_RSS" \
            >> "$PROFILE_CSV"
    fi

    sleep 1
done

set +e
wait "$TRAIN_PID"
STATUS=$?
set -e

echo "Training exit status: $STATUS"
echo "Profile written to: $PROFILE_CSV"

exit "$STATUS"
