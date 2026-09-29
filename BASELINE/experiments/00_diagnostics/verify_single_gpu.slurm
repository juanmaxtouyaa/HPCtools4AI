#!/bin/bash
#SBATCH --job-name=verify-1gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=4G
#SBATCH --gres=gpu:a100:1
#SBATCH --exclusive
#SBATCH --time=00:03:00
#SBATCH --output=verify-single-gpu-%j.out

module load python/3.10.8

PROJECT_ROOT="$HOME/HPCTools/HPCtools4AI"
source "$PROJECT_ROOT/.venv/bin/activate"

echo "=== BATCH SHELL ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo
echo "=== SRUN STEP ==="

srun \
    --ntasks=1 \
    --cpus-per-task=32 \
    --gres=gpu:a100:1 \
    --exclusive \
    python - <<'PY'
import os
import torch

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count() =", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"GPU {i} =", torch.cuda.get_device_name(i))

assert torch.cuda.device_count() == 1, \
    "ERROR: expected exactly one visible GPU"

print("SUCCESS: exactly one A100 is visible.")
PY
