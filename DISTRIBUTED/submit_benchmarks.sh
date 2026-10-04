#!/bin/bash
# Submit the scaling campaign: REPS repetitions of each configuration.
#   ./submit_benchmarks.sh                      # 3 x {1n1g, 1n2g, 2n1g, 2n2g} = 12 jobs
#   REPS=1 CONFIGS="2n2g" ./submit_benchmarks.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REPS="${REPS:-3}"
CONFIGS="${CONFIGS:-1n1g 1n2g 2n1g 2n2g}"

# 1n1g and 2n1g use one GPU per node: --exclusive keeps other jobs off the node
# (they would share its PCIe host bridge and InfiniBand link)
declare -A SBATCH_OPTS=(
    [1n1g]="--exclusive --nodes=1 --ntasks-per-node=1 --gres=gpu:a100:1 --mem=32G"
    [1n2g]="--nodes=1 --ntasks-per-node=2 --gres=gpu:a100:2 --mem=64G"
    [2n1g]="--exclusive --nodes=2 --ntasks-per-node=1 --gres=gpu:a100:1 --mem=32G"
    [2n2g]="--nodes=2 --ntasks-per-node=2 --gres=gpu:a100:2 --mem=64G"
)

for cfg in $CONFIGS; do
    for rep in $(seq 1 "$REPS"); do
        # shellcheck disable=SC2086  # options are intentionally word-split
        RUN_TAG="bench_r${rep}" sbatch --job-name="ddp-${cfg}-r${rep}" ${SBATCH_OPTS[$cfg]} ddp.slurm
    done
done
squeue -u "$USER"
