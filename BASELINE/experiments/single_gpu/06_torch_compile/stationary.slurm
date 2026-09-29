#!/bin/bash
#SBATCH --job-name=bert-compile-stationary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --array=0-1%2
#SBATCH --output=results/single_gpu/06_torch_compile/slurm/stationary-%A_%a.out

set -euo pipefail

module load python/3.10.8

PROJECT_ROOT="$HOME/HPCTools/HPCtools4AI"
BASELINE_ROOT="$PROJECT_ROOT/BASELINE"

cd "$BASELINE_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"

export PYTHONUNBUFFERED=1

MODES=(off on)
MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"

BATCH=160
WORKERS=0
SAMPLES=32768
EPOCHS=3

OUTDIR="results/single_gpu/06_torch_compile/stationary_${MODE}"

if [[ -e "$OUTDIR" ]]; then
    echo "ERROR: output directory already exists:"
    echo "  $OUTDIR"
    exit 1
fi

mkdir -p "$OUTDIR"

# Cold cache for this diagnostic job.
JOB_TMPDIR="${TMPDIR:-/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${JOB_TMPDIR}/torchinductor_stationary_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

EXTRA_ARGS=()

if [[ "$MODE" == "on" ]]; then
    EXTRA_ARGS+=(--torch-compile)

    # Ask TorchDynamo to report actual recompilations and their guard failures.
    export TORCH_LOGS="recompiles"
fi

echo "========================================"
echo "torch.compile stationary diagnostic"
echo "========================================"
echo "Job ID:                $SLURM_JOB_ID"
echo "Array task:            $SLURM_ARRAY_TASK_ID"
echo "Host:                  $(hostname)"
echo "torch.compile:         $MODE"
echo "Training examples:     $SAMPLES"
echo "Epochs:                $EPOCHS"
echo "Batch size:            $BATCH"
echo "DataLoader workers:    $WORKERS"
echo "BF16:                  true"
echo "TORCH_LOGS:            ${TORCH_LOGS:-disabled}"
echo "CUDA visibility:       ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Output directory:      $OUTDIR"
echo "========================================"
echo

srun \
    --ntasks=1 \
    --cpus-per-task=32 \
    --gres=gpu:a100:1 \
    python train.py \
        --max-train-samples "$SAMPLES" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH" \
        --learning-rate 3e-5 \
        --dataloader-num-workers "$WORKERS" \
        --bf16 \
        --skip-save \
        --output-dir "$OUTDIR" \
        "${EXTRA_ARGS[@]}"
