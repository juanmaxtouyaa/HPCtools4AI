#!/bin/bash
#SBATCH --job-name=bert-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:10:00
#SBATCH --output=results/single_gpu/07_profiling/slurm-%j.out

set -euo pipefail

module load python/3.10.8

PROJECT_ROOT="$HOME/HPCTools/HPCtools4AI"
BASELINE_ROOT="$PROJECT_ROOT/BASELINE"

cd "$BASELINE_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"

export PYTHONUNBUFFERED=1

OUTDIR="results/single_gpu/07_profiling/profile_run"

if [[ -e "$OUTDIR" ]]; then
    echo "ERROR: profiling output already exists:"
    echo "  $OUTDIR"
    exit 1
fi

echo "========================================"
echo "Single-GPU profiling run"
echo "========================================"
echo "Job ID:                $SLURM_JOB_ID"
echo "Host:                  $(hostname)"
echo "CPUs allocated:        $SLURM_CPUS_PER_TASK"
echo "Batch CUDA visibility: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Configuration:         BF16 / batch 160 / workers 0"
echo "========================================"
echo

srun \
    --ntasks=1 \
    --cpus-per-task=32 \
    --gres=gpu:a100:1 \
    experiments/single_gpu/07_profiling/profile_run.sh \
    "$OUTDIR"
