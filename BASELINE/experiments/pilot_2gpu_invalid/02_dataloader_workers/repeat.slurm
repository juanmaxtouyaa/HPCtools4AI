#!/bin/bash
#SBATCH --job-name=bert-workers-rep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --array=0-7
#SBATCH --output=outputs/worker_sweep/repeat-%A_%a.out

module load python/3.10.8

PROJECT_ROOT="$HOME/HPCTools/HPCtools4AI"
cd "$PROJECT_ROOT/BASELINE" || exit 1

source "$PROJECT_ROOT/.venv/bin/activate"

export PYTHONUNBUFFERED=1

WORKERS=(0 2 4 8 0 2 4 8)
REPS=(2 2 2 2 3 3 3 3)

NWORKERS=${WORKERS[$SLURM_ARRAY_TASK_ID]}
REP=${REPS[$SLURM_ARRAY_TASK_ID]}

OUTDIR="outputs/worker_sweep/w${NWORKERS}_r${REP}"

echo "========================================"
echo "DataLoader worker repetition"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Workers: $NWORKERS"
echo "Repetition: $REP"
echo "Batch size: 96"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo

python train.py \
    --max-train-samples 4096 \
    --epochs 1 \
    --batch-size 96 \
    --learning-rate 3e-5 \
    --dataloader-num-workers "$NWORKERS" \
    --skip-save \
    --output-dir "$OUTDIR"
