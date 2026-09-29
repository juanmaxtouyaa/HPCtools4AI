#!/bin/bash
#SBATCH --job-name=bert-val-final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --array=0-3%2
#SBATCH --output=results/single_gpu/10_validation_earlystop_10epochs/slurm/slurm-%A_%a.out

set -euo pipefail

module load python/3.10.8

PROJECT_ROOT="$HOME/HPCTools/HPCtools4AI"
BASELINE_ROOT="$PROJECT_ROOT/BASELINE"

cd "$BASELINE_ROOT"

source "$PROJECT_ROOT/.venv/bin/activate"

export PYTHONUNBUFFERED=1


COMPILE=(0 0 1 1)
DROP_LAST=(0 1 0 1)

C="${COMPILE[$SLURM_ARRAY_TASK_ID]}"
D="${DROP_LAST[$SLURM_ARRAY_TASK_ID]}"


if [[ "$C" -eq 0 && "$D" -eq 0 ]]; then
    NAME="eager_keep"

elif [[ "$C" -eq 0 && "$D" -eq 1 ]]; then
    NAME="eager_drop"

elif [[ "$C" -eq 1 && "$D" -eq 0 ]]; then
    NAME="compile_keep"

else
    NAME="compile_drop"
fi


OUTDIR="results/single_gpu/10_validation_earlystop_10epochs/$NAME"


if [[ -e "$OUTDIR" ]]; then
    echo "ERROR: $OUTDIR already exists."
    exit 1
fi


mkdir -p "$OUTDIR"

JOB_TMP="${TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"

mkdir -p "$JOB_TMP"

export CHECKPOINT_DIR="${JOB_TMP}/bert_checkpoints_${SLURM_ARRAY_TASK_ID}"

rm -rf "$CHECKPOINT_DIR"
mkdir -p "$CHECKPOINT_DIR"

echo "TMPDIR:          ${TMPDIR:-<unset>}"
echo "Checkpoint dir:  $CHECKPOINT_DIR"

export TENSORBOARD_LOGGING_DIR="$OUTDIR/tensorboard"
mkdir -p "$TENSORBOARD_LOGGING_DIR"

export TENSORBOARD_LOGGING_DIR="$OUTDIR/tensorboard"
mkdir -p "$TENSORBOARD_LOGGING_DIR"

ARGS=(
    --max-train-samples 32768
    --epochs 10
    --batch-size 160
    --eval-batch-size 160
    --learning-rate 3e-5
    --dataloader-num-workers 0
    --bf16
    --early-stopping-patience 1
    --early-stopping-threshold 0.001
    --seed 42
    --output-dir "$OUTDIR"
)


if [[ "$D" -eq 1 ]]; then
    ARGS+=(--drop-last)
fi


if [[ "$C" -eq 1 ]]; then
    ARGS+=(--torch-compile)
    ARGS+=(--torch-compile-mode default)

    export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor_${NAME}"

    export TORCH_LOGS="recompiles"
fi


echo "========================================"
echo "FINAL VALIDATION EXPERIMENT"
echo "========================================"
echo "Job ID:                $SLURM_JOB_ID"
echo "Array task:            $SLURM_ARRAY_TASK_ID"
echo "Host:                  $(hostname)"
echo "Experiment:            $NAME"
echo "torch.compile:         $C"
echo "drop_last TRAIN:       $D"
echo "drop_last VALIDATION:  0"
echo "Maximum epochs:        10"
echo "Early-stop patience:   1"
echo "Early-stop threshold:  0.001"
echo "Train batch:           160"
echo "Eval batch:            160"
echo "Precision:             BF16"
echo "Train examples:        32768"
echo "Learning rate:         3e-5"
echo "Workers:               0"
echo "CPUs:                  $SLURM_CPUS_PER_TASK"
echo "CUDA visibility:       ${CUDA_VISIBLE_DEVICES:-NA}"
echo "========================================"

echo
echo "=== CPU topology ==="

lscpu | grep -E \
'CPU\(s\)|Core|Socket|NUMA|Thread' \
|| true

echo
echo "=== NUMA topology ==="

if command -v numactl >/dev/null 2>&1; then
    numactl --hardware || true
fi

echo
echo "=== GPU topology ==="

nvidia-smi topo -m || true

echo
echo "=== Starting single-GPU step ==="


srun \
    --ntasks=1 \
    --cpus-per-task=32 \
    --gres=gpu:a100:1 \
    --cpu-bind=cores \
    experiments/single_gpu/10_validation_earlystop_10epochs/profile_run.sh \
    "$OUTDIR" \
    "${ARGS[@]}"
