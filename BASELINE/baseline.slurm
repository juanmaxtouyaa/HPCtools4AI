#!/bin/bash
#SBATCH --job-name=bert-baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-baseline-%j.out

set -euo pipefail

module load python/3.10.8

PROJECT_ROOT="$HOME/HPCTools/HPCtools4AI"
BASELINE_ROOT="$PROJECT_ROOT/BASELINE"

cd "$BASELINE_ROOT"

source "$PROJECT_ROOT/.venv/bin/activate"

export PYTHONUNBUFFERED=1

OUTDIR="$BASELINE_ROOT/results/final_baseline/job_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

echo "========================================"
echo "BERT SQuAD SINGLE-A100 BASELINE"
echo "========================================"
echo "Job ID:             $SLURM_JOB_ID"
echo "Host:               $(hostname)"
echo "CPUs:               $SLURM_CPUS_PER_TASK"
echo "CUDA visibility:    ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Output:             $OUTDIR"
echo
echo "Model:              google-bert/bert-base-uncased"
echo "Dataset:            full SQuAD v1.1 train split"
echo "Epochs:             1"
echo "Batch size:         160"
echo "Learning rate:      3e-5"
echo "Precision:          BF16"
echo "DataLoader workers: 0"
echo "torch.compile:      OFF"
echo "========================================"

srun \
    --ntasks=1 \
    --cpus-per-task=32 \
    --gres=gpu:a100:1 \
    python train.py \
        --epochs 1 \
        --batch-size 160 \
        --learning-rate 3e-5 \
        --dataloader-num-workers 0 \
        --bf16 \
        --skip-save \
        --output-dir "$OUTDIR"
