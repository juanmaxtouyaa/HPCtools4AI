# HPC Tools for AI — BERT/SQuAD Single-GPU Baseline

This directory contains the single-GPU baseline for the HPC Tools for AI project.

The objective is to fine-tune `google-bert/bert-base-uncased` on SQuAD using PyTorch and Hugging Face Transformers while characterizing training performance on exactly one NVIDIA A100 GPU.

## Baseline configuration

The final performance baseline uses:

- Model: `google-bert/bert-base-uncased`
- Dataset: SQuAD v1.1
- GPU: 1 × NVIDIA A100-PCIE-40GB
- Precision: BF16
- Epochs: 1
- Per-device batch size: 160
- Learning rate: `3e-5`
- DataLoader workers: 0
- Sequence length: 384
- Document stride: 128
- Optimizer: fused AdamW
- `torch.compile`: disabled
- TF32: default PyTorch behavior
- Gradient accumulation: 1

The full SQuAD training split contains 87,599 original examples and produces 88,492 tokenized training features with the adopted sliding-window preprocessing.

The final benchmark intentionally uses one epoch because a complete epoch already takes substantially more than one minute, satisfying the workload-duration requirement while keeping repeated benchmarking practical.

## Environment

The experiments were conducted with:

```text
Python       3.10.8
PyTorch      2.11.0+cu128
CUDA build   12.8
Transformers 5.17.0
Datasets     5.0.1
Accelerate   1.15.0
GPU          NVIDIA A100-PCIE-40GB
```

A Python virtual environment is used at:

```text
~/HPCTools/HPCtools4AI/.venv
```

The environment can be activated with:

```bash
module load python/3.10.8
source ~/HPCTools/HPCtools4AI/.venv/bin/activate
```

## Running the baseline

From the `BASELINE` directory:

```bash
./run_baseline.sh
```

or directly:

```bash
sbatch baseline.slurm
```

The SLURM script requests exactly one A100 and 32 CPU cores. The Python process is started through an explicit `srun` step requesting one A100:

```bash
srun \
    --ntasks=1 \
    --cpus-per-task=32 \
    --gres=gpu:a100:1 \
    python train.py ...
```

This is important on CESGA because earlier pilot jobs using node-level `--exclusive` exposed both A100 GPUs on a two-GPU node. Those pilot measurements were therefore invalid as single-GPU benchmarks and are excluded from the reported results.

`train.py` also verifies at runtime that exactly one CUDA device is visible and aborts otherwise.

## What is timed

The principal benchmark measures the wall-clock duration of `trainer.train()`.

CUDA synchronization is performed immediately before and after the timed section.

The reported training time therefore excludes:

- model loading;
- dataset download/loading;
- tokenization and preprocessing;
- final model serialization.

This isolates the actual fine-tuning phase.

## Final baseline result

Three valid full-SQuAD repetitions were obtained:

| Repetition | Training time |
|---|---:|
| 1 | 233.48 s |
| 2 | 235.41 s |
| 3 | 234.85 s |

Summary:

```text
Mean          234.58 s
Median        234.85 s
Sample std.     0.99 s
CV              0.42 %
```

The final single-A100 baseline is therefore approximately:

```text
234.6 seconds = 3.91 minutes per full SQuAD epoch
```

The low coefficient of variation indicates stable timing across the three valid repetitions.

## Performance optimization findings

The principal tuning results were:

- BF16 produced the largest performance improvement compared with the original precision configuration.
- Batch size 160 was selected as the BF16 throughput/memory knee.
- Increasing DataLoader workers from 0 to 2 or 4 produced no measurable training-speed improvement.
- Explicit TF32 substantially accelerated the FP32 path, although BF16 remained faster.
- `torch.compile` reduced CUDA memory usage but did not improve end-to-end time for the short baseline workload because compilation overhead dominated.
- Profiling showed sustained high GPU utilization, with no clear DataLoader or host-memory bottleneck.
- `drop_last=True` eliminates the irregular final batch and avoids one observed `torch.compile` recompile, but its benefit was insufficient to justify changing the official baseline.

Detailed measurements are documented in `EXPERIMENTS.md`.

## Quality-control experiment

The throughput-oriented baseline is not intended to be the best machine-learning hyperparameter configuration.

A separate quality-control run was therefore performed using the full SQuAD dataset with a configuration close to the classical BERT/SQuAD fine-tuning recipe:

```text
Train examples       87,599
Train features       88,492
Batch size           12
Epochs               2
Learning rate        3e-5
Warmup               1,475 steps (~10 %)
Precision            BF16
Optimizer            fused AdamW
Weight decay         0.01
torch.compile        disabled
drop_last            disabled
```

Validation loss:

```text
Epoch 1    0.9823
Epoch 2    0.9893
```

The best checkpoint was therefore epoch 1.

SQuAD validation quality for the best model:

```text
Exact Match    79.56
F1             87.50
```

The experiment demonstrates an important distinction between HPC throughput optimization and machine-learning quality: the large batch size used to maximize single-A100 throughput is not necessarily the best fine-tuning configuration for generalization.

The best quality-control model is stored outside the Git repository under CESGA `STORE`, while checkpoints and model weights are excluded from version control.

## Profiling

System profiling was sampled approximately once per second and included:

- GPU utilization;
- GPU memory usage;
- GPU memory-controller utilization;
- power draw;
- GPU temperature;
- SM and memory clocks;
- process-tree CPU usage;
- process RSS and virtual memory;
- process thread count;
- node load average;
- available system RAM.

For the representative BF16 batch-160 training profile, GPU utilization was at least 80% for approximately 97.6% of GPU samples, supporting the conclusion that the workload is primarily GPU-bound during the timed training phase.

## Repository contents

```text
BASELINE/
├── README.md
├── EXPERIMENTS.md
├── train.py
├── train_experiments.py
├── train_v3_validation.py
├── baseline.slurm
├── run_baseline.sh
├── experiments/
└── archive/
```

`train.py` is the official single-A100 baseline implementation.

`train_experiments.py` contains additional controls used for performance experiments such as `drop_last` and `torch.compile`.

`train_v3_validation.py` adds validation, early stopping, quality metrics, resource instrumentation, and best-checkpoint handling.

Generated results, TensorBoard files, checkpoints, virtual environments, and model weights are intentionally excluded from Git.

## Reproducibility notes

All valid performance results enforce exactly one visible CUDA GPU.

The random seed used for the experiments is 42.

Because GPU kernels and framework internals are not necessarily bitwise deterministic, small variations in loss or quality metrics may remain even with a fixed seed.

The CESGA compute nodes use a managed software and kernel environment. NUMA and GPU topology were recorded during profiling, but no claim is made that NUMA placement was manually optimized.

## Main conclusion

For this workload, the selected performance configuration is:

```text
BERT-base-uncased
full SQuAD
1 × A100-PCIE-40GB
BF16
batch size 160
1 epoch
~234.6 s training time
```

A separate quality-focused configuration using full SQuAD, batch size 12 and warmup obtained:

```text
EM 79.56
F1 87.50
```

These results provide both a reproducible single-GPU HPC baseline and a machine-learning quality sanity check for the later distributed-training experiments.
