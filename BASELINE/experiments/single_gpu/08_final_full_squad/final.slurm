#!/bin/bash
#SBATCH --job-name=bert-final-squad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --array=0-2%2
#SBATCH --output=results/single_gpu/08_final_full_squad/slurm/slurm-%A_%a.out

set -euo pipefail

module load python/3.10.8

PROJECT_ROOT="$HOME/HPCTools/HPCtools4AI"
BASELINE_ROOT="$PROJECT_ROOT/BASELINE"

cd "$BASELINE_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"

export PYTHONUNBUFFERED=1

REPS=(1 2 3)

REP="${REPS[$SLURM_ARRAY_TASK_ID]}"

BATCH=160
WORKERS=0
EPOCHS=1

OUTDIR="results/single_gpu/08_final_full_squad/rep${REP}"

if [[ -e "$OUTDIR" ]]; then
    echo "ERROR: output directory already exists:"
    echo "  $OUTDIR"
    echo "Refusing to overwrite an existing final benchmark."
    exit 1
fi

mkdir -p "$OUTDIR"

echo "========================================"
echo "FINAL BASELINE - full SQuAD"
echo "========================================"
echo "Job ID:                $SLURM_JOB_ID"
echo "Array task:            $SLURM_ARRAY_TASK_ID"
echo "Host:                  $(hostname)"
echo "Repetition:            $REP"
echo "Epochs:                $EPOCHS"
echo "Batch size:            $BATCH"
echo "DataLoader workers:    $WORKERS"
echo "Precision:             BF16"
echo "torch.compile:         OFF"
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
        --epochs "$EPOCHS" \
        --batch-size "$BATCH" \
        --learning-rate 3e-5 \
        --dataloader-num-workers "$WORKERS" \
        --bf16 \
        --skip-save \
        --output-dir "$OUTDIR"
