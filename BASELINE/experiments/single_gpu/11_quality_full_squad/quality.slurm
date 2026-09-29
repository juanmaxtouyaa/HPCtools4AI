#!/bin/bash
#SBATCH --job-name=bert-quality
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --output=results/single_gpu/11_quality_full_squad/slurm/slurm-%j.out

set -euo pipefail

module load python/3.10.8

PROJECT_ROOT="$HOME/HPCTools/HPCtools4AI"
BASELINE_ROOT="$PROJECT_ROOT/BASELINE"

cd "$BASELINE_ROOT"

source "$PROJECT_ROOT/.venv/bin/activate"

export PYTHONUNBUFFERED=1


# ------------------------------------------------------------
# Persistent results -> STORE
# Temporary checkpoints -> node-local scratch
# ------------------------------------------------------------

STORE_ROOT="$STORE/HPCtools4AI/BASELINE/11_quality_full_squad"

OUTDIR="$STORE_ROOT/bert_base_squad_quality"

if [[ -e "$OUTDIR" ]]; then
    echo "ERROR: output already exists:"
    echo "$OUTDIR"
    exit 1
fi

mkdir -p "$OUTDIR"


JOB_TMP="${TMPDIR:-/scratch/$SLURM_JOB_ID}"

mkdir -p "$JOB_TMP"

export CHECKPOINT_DIR="$JOB_TMP/quality_checkpoints"

rm -rf "$CHECKPOINT_DIR"
mkdir -p "$CHECKPOINT_DIR"


export TENSORBOARD_LOGGING_DIR="$OUTDIR/tensorboard"
mkdir -p "$TENSORBOARD_LOGGING_DIR"


echo "========================================"
echo "BERT SQuAD QUALITY CONTROL RUN"
echo "========================================"
echo "Host:             $(hostname)"
echo "Job:              $SLURM_JOB_ID"
echo "GPU visibility:   ${CUDA_VISIBLE_DEVICES:-NA}"
echo "Results:          $OUTDIR"
echo "Checkpoints:      $CHECKPOINT_DIR"
echo
echo "Train dataset:    FULL SQuAD"
echo "Original samples: 87599"
echo "Train batch:      12"
echo "Eval batch:       160"
echo "Epochs:           2"
echo "LR:               3e-5"
echo "Warmup:           1475 steps (~10%)"
echo "BF16:             ON"
echo "torch.compile:    OFF"
echo "drop_last:        OFF"
echo "max length:       384"
echo "doc stride:       128"
echo "========================================"


echo
echo "=== GPU / CPU topology ==="

nvidia-smi
nvidia-smi topo -m || true

lscpu | grep -E \
'CPU\(s\)|Core|Socket|NUMA|Thread' \
|| true


echo
echo "=== TRAINING ==="

srun \
    --ntasks=1 \
    --cpus-per-task=32 \
    --gres=gpu:a100:1 \
    --cpu-bind=cores \
    experiments/single_gpu/10_validation_earlystop_10epochs/profile_run.sh \
    "$OUTDIR" \
    --max-train-samples 87599 \
    --epochs 2 \
    --batch-size 12 \
    --eval-batch-size 160 \
    --learning-rate 3e-5 \
    --warmup-steps 1475 \
    --dataloader-num-workers 0 \
    --bf16 \
    --early-stopping-patience 1 \
    --early-stopping-threshold 0.001 \
    --seed 42 \
    --output-dir "$OUTDIR"


echo
echo "=== EM/F1 FOR EACH EPOCH CHECKPOINT ==="

srun \
    --ntasks=1 \
    --cpus-per-task=32 \
    --gres=gpu:a100:1 \
    python \
    experiments/single_gpu/11_quality_full_squad/evaluate_checkpoints.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "$OUTDIR" \
    --eval-batch-size 160


echo
echo "=== FINAL FILES ==="

du -sh "$OUTDIR"
find "$OUTDIR" -maxdepth 2 -type f \
    -printf '%p %k KB\n' \
    | sort


echo
echo "QUALITY RUN COMPLETED"
