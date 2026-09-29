#!/bin/bash
#SBATCH --job-name=bert-precision
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --array=0-5
#SBATCH --output=outputs/precision_sweep/slurm-%A_%a.out

module load python/3.10.8

PROJECT_ROOT="$HOME/HPCTools/HPCtools4AI"
cd "$PROJECT_ROOT/BASELINE" || exit 1

source "$PROJECT_ROOT/.venv/bin/activate"

export PYTHONUNBUFFERED=1

MODES=(fp32 bf16 fp32 bf16 fp32 bf16)
REPS=(1 1 2 2 3 3)

MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
REP=${REPS[$SLURM_ARRAY_TASK_ID]}

OUTDIR="outputs/precision_sweep/${MODE}_r${REP}"

echo "========================================"
echo "Precision benchmark"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Host: $(hostname)"
echo "Mode: $MODE"
echo "Repetition: $REP"
echo "Batch size: 96"
echo "DataLoader workers: 2"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo

EXTRA_ARGS=()

if [ "$MODE" = "bf16" ]; then
    EXTRA_ARGS+=(--bf16)
fi

python train.py \
    --max-train-samples 4096 \
    --epochs 1 \
    --batch-size 96 \
    --learning-rate 3e-5 \
    --dataloader-num-workers 2 \
    --skip-save \
    --output-dir "$OUTDIR" \
    "${EXTRA_ARGS[@]}"
