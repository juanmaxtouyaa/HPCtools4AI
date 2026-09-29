#!/bin/bash

set -uo pipefail

OUTDIR="$1"
shift

mkdir -p "$OUTDIR"

PROFILE="$OUTDIR/system_profile.csv"

echo \
"timestamp,phase,gpu_uuid,gpu_util_percent,gpu_memory_util_percent,gpu_memory_used_mib,gpu_memory_total_mib,power_w,temperature_c,sm_clock_mhz,memory_clock_mhz,pstate,process_tree_cpu_percent,process_tree_rss_kib,process_tree_vsz_kib,process_tree_threads,node_load_1m,node_load_5m,node_load_15m,node_mem_available_kib" \
> "$PROFILE"

echo "=== Runtime affinity ==="
echo "PID: $$"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NA}"

taskset -pc $$ || true

if command -v numactl >/dev/null 2>&1; then
    numactl --show || true
fi

python train_v3_validation.py "$@" &

TRAIN_PID=$!

echo "Training PID: $TRAIN_PID"


collect_pid_tree()
{
    local root="$1"
    local queue=("$root")
    local result=()

    while ((${#queue[@]} > 0)); do
        local pid="${queue[0]}"
        queue=("${queue[@]:1}")

        if [[ ! -d "/proc/$pid" ]]; then
            continue
        fi

        result+=("$pid")

        while read -r child; do
            if [[ -n "$child" ]]; then
                queue+=("$child")
            fi
        done < <(
            pgrep -P "$pid" 2>/dev/null || true
        )
    done

    local IFS=,
    echo "${result[*]}"
}


GPU_UUID=""

while kill -0 "$TRAIN_PID" 2>/dev/null; do

    TIMESTAMP="$(date --iso-8601=seconds)"

    PHASE="$(
        cat "$OUTDIR/phase.txt" \
        2>/dev/null \
        || echo "startup"
    )"

    if [[ -z "$GPU_UUID" ]]; then
        GPU_UUID="$(
            nvidia-smi \
                --query-compute-apps=pid,gpu_uuid \
                --format=csv,noheader,nounits \
            2>/dev/null \
            | awk -F', *' \
                -v pid="$TRAIN_PID" \
                '$1 == pid {print $2; exit}'
        )"
    fi

    GPU_UTIL="NA"
    GPU_MEM_UTIL="NA"
    GPU_MEM_USED="NA"
    GPU_MEM_TOTAL="NA"
    POWER="NA"
    TEMP="NA"
    SM_CLOCK="NA"
    MEM_CLOCK="NA"
    PSTATE="NA"

    if [[ -n "$GPU_UUID" ]]; then

        GPU_LINE="$(
            nvidia-smi \
                -i "$GPU_UUID" \
                --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem,pstate \
                --format=csv,noheader,nounits \
            2>/dev/null \
            | sed 's/, */,/g'
        )"

        if [[ -n "$GPU_LINE" ]]; then
            IFS=',' read -r \
                GPU_UTIL \
                GPU_MEM_UTIL \
                GPU_MEM_USED \
                GPU_MEM_TOTAL \
                POWER \
                TEMP \
                SM_CLOCK \
                MEM_CLOCK \
                PSTATE \
                <<< "$GPU_LINE"
        fi
    fi

    PIDS="$(collect_pid_tree "$TRAIN_PID")"

    CPU_PCT="NA"
    RSS="NA"
    VSZ="NA"
    THREADS="NA"

    if [[ -n "$PIDS" ]]; then

        PROC_LINE="$(
            ps -p "$PIDS" \
                -o %cpu=,rss=,vsz=,nlwp= \
                2>/dev/null \
            | awk '
                {
                    cpu += $1
                    rss += $2
                    vsz += $3
                    thr += $4
                }
                END {
                    if (NR > 0) {
                        printf "%.2f,%d,%d,%d",
                            cpu,rss,vsz,thr
                    }
                }
            '
        )"

        if [[ -n "$PROC_LINE" ]]; then
            IFS=',' read -r \
                CPU_PCT \
                RSS \
                VSZ \
                THREADS \
                <<< "$PROC_LINE"
        fi
    fi

    read -r \
        LOAD1 \
        LOAD5 \
        LOAD15 \
        _ \
        < /proc/loadavg

    MEM_AVAILABLE="$(
        awk \
            '/^MemAvailable:/ {print $2}' \
            /proc/meminfo
    )"

    echo \
"$TIMESTAMP,$PHASE,${GPU_UUID:-NA},$GPU_UTIL,$GPU_MEM_UTIL,$GPU_MEM_USED,$GPU_MEM_TOTAL,$POWER,$TEMP,$SM_CLOCK,$MEM_CLOCK,$PSTATE,$CPU_PCT,$RSS,$VSZ,$THREADS,$LOAD1,$LOAD5,$LOAD15,$MEM_AVAILABLE" \
    >> "$PROFILE"

    sleep 1
done

set +e
wait "$TRAIN_PID"
STATUS=$?
set -e

echo "Training exit status: $STATUS"
echo "Profile written to: $PROFILE"

exit "$STATUS"
