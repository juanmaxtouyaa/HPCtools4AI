#!/bin/bash
#SBATCH --job-name=bert-1gpu-compile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --array=0-5%2
#SBATCH --output=results/single_gpu/06_torch_compile/slurm/slurm-%A_%a.out

set -euo pipefail

module load python/3.10.8

PROJECT_ROOT="$HOME/HPCTools/HPCtools4AI"
BASELINE_ROOT="$PROJECT_ROOT/BASELINE"

cd "$BASELINE_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"

export PYTHONUNBUFFERED=1

MODES=(off on off on off on)
REPS=(1 1 2 2 3 3)

MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"
REP="${REPS[$SLURM_ARRAY_TASK_ID]}"

BATCH=160
WORKERS=0
SAMPLES=32768

OUTDIR="results/single_gpu/06_torch_compile/compile_${MODE}_r${REP}"

if [[ -e "$OUTDIR" ]]; then
    echo "ERROR: output directory already exists:"
    echo "  $OUTDIR"
    echo "Refusing to overwrite an existing experiment."
    exit 1
fi

mkdir -p "$OUTDIR"

# Separate cache per task so every compile=ON run measures a cold compile.
JOB_TMPDIR="${TMPDIR:-/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${JOB_TMPDIR}/torchinductor_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

EXTRA_ARGS=()

if [[ "$MODE" == "on" ]]; then
    EXTRA_ARGS+=(--torch-compile)
fi

echo "========================================"
echo "Single-GPU torch.compile comparison"
echo "========================================"
echo "Job ID:                $SLURM_JOB_ID"
echo "Array task:            $SLURM_ARRAY_TASK_ID"
echo "Host:                  $(hostname)"
echo "torch.compile:         $MODE"
echo "Repetition:            $REP"
echo "Training examples:     $SAMPLES"
echo "Batch size:            $BATCH"
echo "DataLoader workers:    $WORKERS"
echo "BF16:                  true"
echo "CPUs allocated:        $SLURM_CPUS_PER_TASK"
echo "Batch CUDA visibility: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Inductor cache:        $TORCHINDUCTOR_CACHE_DIR"
echo "Output directory:      $OUTDIR"
echo "========================================"
echo

srun \
    --ntasks=1 \
    --cpus-per-task=32 \
    --gres=gpu:a100:1 \
    python train.py \
        --max-train-samples "$SAMPLES" \
        --epochs 1 \
        --batch-size "$BATCH" \
        --learning-rate 3e-5 \
        --dataloader-num-workers "$WORKERS" \
        --bf16 \
        --skip-save \
        --output-dir "$OUTDIR" \
        "${EXTRA_ARGS[@]}"
