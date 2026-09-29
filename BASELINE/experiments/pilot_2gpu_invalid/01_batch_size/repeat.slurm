#!/bin/bash
#SBATCH --job-name=bert-batch-rep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --array=0-5
#SBATCH --output=outputs/batch_sweep/repeat-%A_%a.out

module load python/3.10.8

PROJECT_ROOT="$HOME/HPCTools/HPCtools4AI"
cd "$PROJECT_ROOT/BASELINE" || exit 1

source "$PROJECT_ROOT/.venv/bin/activate"

export PYTHONUNBUFFERED=1

BATCHES=(48 96 128 48 96 128)
REPS=(2 2 2 3 3 3)

BATCH=${BATCHES[$SLURM_ARRAY_TASK_ID]}
REP=${REPS[$SLURM_ARRAY_TASK_ID]}

OUTDIR="outputs/batch_sweep/b${BATCH}_r${REP}"

echo "========================================"
echo "Batch-size repetition"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Host: $(hostname)"
echo "Batch size: $BATCH"
echo "Repetition: $REP"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo

python train.py \
    --max-train-samples 4096 \
    --epochs 1 \
    --batch-size "$BATCH" \
    --learning-rate 3e-5 \
    --dataloader-num-workers 4 \
    --skip-save \
    --output-dir "$OUTDIR"
