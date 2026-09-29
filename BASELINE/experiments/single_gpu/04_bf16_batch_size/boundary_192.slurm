#!/bin/bash
#SBATCH --job-name=bert-bf16-b192
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --output=results/single_gpu/04_bf16_batch_size/slurm/b192-%j.out

set -euo pipefail

module load python/3.10.8

PROJECT_ROOT="$HOME/HPCTools/HPCtools4AI"
BASELINE_ROOT="$PROJECT_ROOT/BASELINE"

cd "$BASELINE_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"

export PYTHONUNBUFFERED=1

BATCH=192
WORKERS=0
REP=1

OUTDIR="results/single_gpu/04_bf16_batch_size/b${BATCH}_r${REP}"

if [[ -e "$OUTDIR" ]]; then
    echo "ERROR: output directory already exists:"
    echo "  $OUTDIR"
    exit 1
fi

mkdir -p "$OUTDIR"

echo "========================================"
echo "Single-GPU BF16 batch boundary test"
echo "========================================"
echo "Job ID:                $SLURM_JOB_ID"
echo "Host:                  $(hostname)"
echo "Batch size:            $BATCH"
echo "DataLoader workers:    $WORKERS"
echo "Precision:             BF16"
echo "CPUs allocated:        $SLURM_CPUS_PER_TASK"
echo "Batch CUDA visibility: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Output directory:      $OUTDIR"
echo "========================================"
echo

srun \
    --ntasks=1 \
    --cpus-per-task=32 \
    --gres=gpu:a100:1 \
    python train.py \
        --max-train-samples 4096 \
        --epochs 1 \
        --batch-size "$BATCH" \
        --learning-rate 3e-5 \
        --dataloader-num-workers "$WORKERS" \
        --bf16 \
        --skip-save \
        --output-dir "$OUTDIR"
