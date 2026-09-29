#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

cd "$SCRIPT_DIR"

mkdir -p results/final_baseline

echo "Submitting single-A100 BERT/SQuAD baseline..."
sbatch baseline.slurm
