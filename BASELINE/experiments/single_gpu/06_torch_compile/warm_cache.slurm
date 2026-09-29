#!/bin/bash
#SBATCH --job-name=bert-compile-cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:10:00
#SBATCH --output=results/single_gpu/06_torch_compile/slurm/warm-cache-%j.out

set -euo pipefail

module load python/3.10.8

PROJECT_ROOT="$HOME/HPCTools/HPCtools4AI"
BASELINE_ROOT="$PROJECT_ROOT/BASELINE"

cd "$BASELINE_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"

export PYTHONUNBUFFERED=1

# One persistent cache shared by BOTH runs in this allocation.
JOB_TMPDIR="${TMPDIR:-/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${JOB_TMPDIR}/torchinductor_warm_test_${SLURM_JOB_ID}"

mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

COLD_OUT="results/single_gpu/06_torch_compile/cache_cold"
WARM_OUT="results/single_gpu/06_torch_compile/cache_warm"

if [[ -e "$COLD_OUT" || -e "$WARM_OUT" ]]; then
    echo "ERROR: cache diagnostic output already exists."
    exit 1
fi

echo "========================================"
echo "torch.compile cache diagnostic"
echo "========================================"
echo "Job ID:         $SLURM_JOB_ID"
echo "Host:           $(hostname)"
echo "Cache dir:      $TORCHINDUCTOR_CACHE_DIR"
echo "========================================"

echo
echo "========== COLD CACHE RUN =========="

srun \
    --ntasks=1 \
    --cpus-per-task=32 \
    --gres=gpu:a100:1 \
    python train.py \
        --max-train-samples 32768 \
        --epochs 1 \
        --batch-size 160 \
        --learning-rate 3e-5 \
        --dataloader-num-workers 0 \
        --bf16 \
        --torch-compile \
        --skip-save \
        --output-dir "$COLD_OUT"

echo
echo "========== WARM CACHE RUN =========="

srun \
    --ntasks=1 \
    --cpus-per-task=32 \
    --gres=gpu:a100:1 \
    python train.py \
        --max-train-samples 32768 \
        --epochs 1 \
        --batch-size 160 \
        --learning-rate 3e-5 \
        --dataloader-num-workers 0 \
        --bf16 \
        --torch-compile \
        --skip-save \
        --output-dir "$WARM_OUT"

echo
echo "========== CACHE TEST FINISHED =========="
