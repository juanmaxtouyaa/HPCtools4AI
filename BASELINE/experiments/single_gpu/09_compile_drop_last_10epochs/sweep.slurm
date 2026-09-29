#!/bin/bash
#SBATCH --job-name=bert-comp-drop10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --array=0-3%2
#SBATCH --output=results/single_gpu/09_compile_drop_last_10epochs/slurm/slurm-%A_%a.out

set -euo pipefail

module load python/3.10.8

PROJECT_ROOT="$HOME/HPCTools/HPCtools4AI"
BASELINE_ROOT="$PROJECT_ROOT/BASELINE"

cd "$BASELINE_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"

export PYTHONUNBUFFERED=1

# ============================================================
# 2 x 2 factorial experiment
#
# task 0: eager   + keep last batch
# task 1: eager   + drop last batch
# task 2: compile + keep last batch
# task 3: compile + drop last batch
# ============================================================

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

OUTDIR="results/single_gpu/09_compile_drop_last_10epochs/$NAME"

if [[ -e "$OUTDIR" ]]; then
    echo "ERROR: output directory already exists:"
    echo "  $OUTDIR"
    echo "Refusing to overwrite."
    exit 1
fi

mkdir -p "$OUTDIR"

ARGS=(
    --max-train-samples 32768
    --epochs 10
    --batch-size 160
    --learning-rate 3e-5
    --dataloader-num-workers 0
    --bf16
    --skip-save
    --seed 42
    --output-dir "$OUTDIR"
)

if [[ "$D" -eq 1 ]]; then
    ARGS+=(--drop-last)
fi

if [[ "$C" -eq 1 ]]; then
    ARGS+=(--torch-compile)
    ARGS+=(--torch-compile-mode default)

    # Unique cold Inductor cache for each compiled run.
    export TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/torchinductor_${NAME}"

    # Record graph recompilations.
    export TORCH_LOGS="recompiles"
fi

echo "========================================"
echo "10-EPOCH COMPILE x DROP_LAST EXPERIMENT"
echo "========================================"
echo "Job ID:                $SLURM_JOB_ID"
echo "Array task:            $SLURM_ARRAY_TASK_ID"
echo "Host:                  $(hostname)"
echo "Experiment:            $NAME"
echo "torch.compile:         $C"
echo "drop_last:             $D"
echo "Epochs:                10"
echo "Batch size:            160"
echo "Precision:             BF16"
echo "Samples requested:     32768"
echo "Learning rate:         3e-5"
echo "DataLoader workers:    0"
echo "CPUs allocated:        $SLURM_CPUS_PER_TASK"
echo "Batch CUDA visibility: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "========================================"
echo

srun \
    --ntasks=1 \
    --cpus-per-task=32 \
    --gres=gpu:a100:1 \
    python train_experiments.py "${ARGS[@]}"
