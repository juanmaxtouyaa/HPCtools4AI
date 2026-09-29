#!/bin/bash
#SBATCH --job-name=bert-bf16-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --array=0-5
#SBATCH --output=outputs/bf16_batch_sweep/slurm-%A_%a.out

module load python/3.10.8

PROJECT_ROOT="$HOME/HPCTools/HPCtools4AI"
cd "$PROJECT_ROOT/BASELINE" || exit 1

source "$PROJECT_ROOT/.venv/bin/activate"

export PYTHONUNBUFFERED=1

BATCHES=(64 96 128 160 192 256)
BATCH=${BATCHES[$SLURM_ARRAY_TASK_ID]}

OUTDIR="outputs/bf16_batch_sweep/b${BATCH}_r1"

echo "========================================"
echo "BF16 batch-size screening"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Host: $(hostname)"
echo "Batch size: $BATCH"
echo "DataLoader workers: 2"
echo "Precision: BF16"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo

python trainVersionBF16.py \
    --max-train-samples 4096 \
    --epochs 1 \
    --batch-size "$BATCH" \
    --learning-rate 3e-5 \
    --dataloader-num-workers 2 \
    --skip-save \
    --output-dir "$OUTDIR"
