#!/bin/bash
# Executed once per task by `srun`. Maps the SLURM task ids to the variables
# torch.distributed expects, starts a 1 Hz GPU sampler on the first task of each
# node, then runs the command given as arguments (python train_ddp.py ...).
set -euo pipefail

export RANK="$SLURM_PROCID"
export LOCAL_RANK="$SLURM_LOCALID"
export WORLD_SIZE="$SLURM_NTASKS"

HOST="$(hostname -s)"
OUTDIR="${OUTDIR:?OUTDIR must be exported by the sbatch script}"
MON_PID=""

if [[ "$LOCAL_RANK" == "0" ]]; then
    nvidia-smi topo -m > "$OUTDIR/topo_${HOST}.txt" 2>&1 || true
    if [[ "${GPU_MONITOR:-1}" == "1" ]]; then
        nvidia-smi \
            --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,clocks.sm \
            --format=csv,nounits -l 1 > "$OUTDIR/gpu_${HOST}.csv" 2>/dev/null &
        MON_PID=$!
    fi
fi

set +e
"$@"
STATUS=$?
set -e

if [[ -n "$MON_PID" ]]; then
    kill "$MON_PID" 2>/dev/null || true
fi
exit "$STATUS"
