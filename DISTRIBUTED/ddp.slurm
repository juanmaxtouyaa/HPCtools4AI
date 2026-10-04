#!/bin/bash
#SBATCH --job-name=bert-ddp
#SBATCH --nodes=2                 # 2 A100 nodes
#SBATCH --ntasks-per-node=2       # one process (task) per GPU
#SBATCH --cpus-per-task=32        # FT3 policy: 32 cores per A100
#SBATCH --gres=gpu:a100:2         # GPUs per node
#SBATCH --mem=64G                 # per node
#SBATCH --time=00:20:00
#SBATCH --output=slurm-ddp-%j.out
#
# Default = 2 nodes x 2 A100 (4 ranks). Other configurations by overriding on the CLI:
#   sbatch --nodes=1 --ntasks-per-node=1 --gres=gpu:a100:1 --mem=32G ddp.slurm   # 1n1g
#   sbatch --nodes=1                                                    ddp.slurm   # 1n2g
#   sbatch --nodes=2 --ntasks-per-node=1 --gres=gpu:a100:1 --mem=32G ddp.slurm   # 2n1g
# Optional environment variables (exported to the job by sbatch):
#   RUN_TAG=smoke EXTRA_ARGS="--max-steps 30" NCCL_DEBUG=INFO GPU_MONITOR=0

set -euo pipefail

module load python/3.10.8

DIST_ROOT="$SLURM_SUBMIT_DIR"
PROJECT_ROOT="$(cd "$DIST_ROOT/.." && pwd)"
cd "$DIST_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"

# ---- rendezvous for torch.distributed (env:// initialisation) ----
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((10000 + SLURM_JOB_ID % 10000))"
export WORLD_SIZE="$SLURM_NTASKS"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"   # INFO prints the transport (NET/IB, P2P, SHM)

GPN="${SLURM_NTASKS_PER_NODE:-$((SLURM_NTASKS / SLURM_JOB_NUM_NODES))}"
CONFIG="${SLURM_JOB_NUM_NODES}n${GPN}g"
RUN_TAG="${RUN_TAG:-bench}"
export OUTDIR="$DIST_ROOT/results/runs/$CONFIG/${RUN_TAG}_job${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

echo "========================================"
echo "BERT/SQuAD DISTRIBUTED (PyTorch DDP)"
echo "========================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Config:       $CONFIG  (world size $WORLD_SIZE)"
echo "Nodes:        $SLURM_JOB_NODELIST"
echo "Master:       $MASTER_ADDR:$MASTER_PORT"
echo "NCCL_DEBUG:   $NCCL_DEBUG"
echo "Output:       $OUTDIR"
echo "Extra args:   ${EXTRA_ARGS:-<none>}"
echo "========================================"

# --cpus-per-task must be repeated: srun does not inherit it from sbatch (Slurm >= 22.05)
# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
srun --ntasks-per-node="$GPN" \
     --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     --kill-on-bad-exit=1 \
     ./run_ddp_task.sh python train_ddp.py \
        --epochs 1 \
        --batch-size 160 \
        --learning-rate 3e-5 \
        --num-workers 0 \
        --bf16 \
        --skip-save \
        --tokenized-dir "$DIST_ROOT/data/squad_bert-base-uncased_len384_stride128" \
        --output-dir "$OUTDIR" \
        ${EXTRA_ARGS:-}
