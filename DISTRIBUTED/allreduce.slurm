#!/bin/bash
#SBATCH --job-name=nccl-allreduce
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-allreduce-%j.out
#
# Same layouts as ddp.slurm (override --nodes / --ntasks-per-node / --gres on the CLI).

set -euo pipefail

module load python/3.10.8

DIST_ROOT="$SLURM_SUBMIT_DIR"
PROJECT_ROOT="$(cd "$DIST_ROOT/.." && pwd)"
cd "$DIST_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((10000 + SLURM_JOB_ID % 10000))"
export WORLD_SIZE="$SLURM_NTASKS"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
export GPU_MONITOR=0

GPN="${SLURM_NTASKS_PER_NODE:-$((SLURM_NTASKS / SLURM_JOB_NUM_NODES))}"
CONFIG="${SLURM_JOB_NUM_NODES}n${GPN}g"
export OUTDIR="$DIST_ROOT/results/allreduce/$CONFIG/job${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

srun --ntasks-per-node="$GPN" \
     --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     --kill-on-bad-exit=1 \
     ./run_ddp_task.sh python allreduce_bench.py --output "$OUTDIR/allreduce.json"
