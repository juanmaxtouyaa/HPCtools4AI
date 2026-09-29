# HPC Tools for AI — BERT/SQuAD Single-GPU Baseline

**Platform:** CESGA FinisTerrae III
**Model:** `google-bert/bert-base-uncased`
**Dataset:** SQuAD v1.1
**Framework:** PyTorch + Hugging Face Transformers
**Reference hardware:** 1 × NVIDIA A100-PCIE-40GB

---

## Abstract

This report establishes a reproducible **single-GPU reference baseline** for fine-tuning BERT on SQuAD before moving to distributed training. The study focuses on training throughput, GPU utilization, memory usage, precision modes, batch size, DataLoader configuration, `torch.compile`, and timing stability on exactly one NVIDIA A100 GPU.

The final performance baseline uses **BF16**, a **per-device batch size of 160**, and **one full SQuAD epoch**. Across three valid repetitions, the mean training time was **234.58 s** with a **0.42% coefficient of variation**, providing a stable reference point for later multi-GPU scaling experiments.

> **Reference baseline:** 1 × A100-PCIE-40GB · BF16 · batch 160 · 1 epoch · **234.58 s mean training time**

---

## 1. Objectives

The purpose of this baseline is to characterize BERT fine-tuning performance on a single accelerator before introducing distributed execution.

The baseline was designed to:

- fine-tune `google-bert/bert-base-uncased` on SQuAD v1.1;
- run on **exactly one NVIDIA A100 GPU**;
- measure the wall-clock duration of the training phase;
- use a workload long enough to produce meaningful timing measurements;
- identify relevant compute, memory, and input-pipeline bottlenecks;
- provide a reproducible reference for future distributed-training experiments.

An initial pilot series was discarded because the SLURM configuration exposed both GPUs of a two-A100 node to the training process. Those measurements are retained only for diagnostic purposes and are **not** included in the valid single-GPU results.

---

## 2. Experimental Environment

### 2.1 Software stack

| Component | Version |
|---|---|
| Python | 3.10.8 |
| PyTorch | 2.11.0+cu128 |
| CUDA build | 12.8 |
| Transformers | 5.17.0 |
| Datasets | 5.0.1 |
| Accelerate | 1.15.0 |
| GPU | NVIDIA A100-PCIE-40GB |

A dedicated virtual environment is stored at:

```text
../.venv
```

Activate it with:

```bash
module load python/3.10.8
source ../.venv/bin/activate
```

### 2.2 Dataset preprocessing

The full SQuAD training split contains **87,599 original examples**. With the adopted sliding-window tokenization strategy, these become **88,492 training features**.

Preprocessing parameters:

| Parameter | Value |
|---|---:|
| Maximum sequence length | 384 |
| Document stride | 128 |
| Padding | Fixed to maximum length |

---

## 3. Final Baseline Configuration

The selected single-A100 performance configuration is:

| Parameter | Value |
|---|---:|
| Model | `google-bert/bert-base-uncased` |
| Dataset | SQuAD v1.1 |
| GPU | 1 × NVIDIA A100-PCIE-40GB |
| Precision | BF16 |
| Epochs | 1 |
| Per-device batch size | 160 |
| Learning rate | `3e-5` |
| DataLoader workers | 0 |
| Gradient accumulation | 1 |
| Optimizer | Fused AdamW (`adamw_torch_fused`) |
| Scheduler | Linear |
| `torch.compile` | Disabled |
| TF32 | Default PyTorch behavior |

One complete epoch already takes substantially more than one minute, so a one-epoch benchmark satisfies the workload-duration requirement while keeping repeated measurements practical.

---

## 4. Single-GPU Enforcement on CESGA

The official SLURM configuration requests one A100 and 32 CPU cores. The Python process is launched through an explicit `srun` step:

```bash
srun \
    --ntasks=1 \
    --cpus-per-task=32 \
    --gres=gpu:a100:1 \
    python train.py ...
```

This detail is important on CESGA. Earlier jobs using node-level `--exclusive` exposed both A100 GPUs of a two-GPU node to the batch shell. As a result, Hugging Face Trainer could see two GPUs even though the job requested one accelerator.

The final implementation therefore uses two safeguards:

1. GPU isolation is enforced at the `srun` step.
2. `train.py` aborts unless **exactly one CUDA device** is visible at runtime.

This guarantees that all reported baseline timings correspond to one A100.

---

## 5. Timing Methodology

The principal benchmark measures only the wall-clock duration of:

```python
trainer.train()
```

CUDA synchronization is executed immediately before and after the timed region.

The reported training time therefore excludes:

- model loading;
- dataset download and loading;
- tokenization and preprocessing;
- final model serialization.

This isolates the fine-tuning phase and makes repeated measurements easier to compare.

---

## 6. Performance Optimization Study

### 6.1 Batch size

With the original precision configuration, increasing batch size from 48 to 96 and 128 did not materially improve throughput. Runtime remained close to 59 s on the controlled 4,096-example subset, while GPU memory usage increased significantly.

After enabling BF16, batch size was retuned. Batch size **160** was selected as the practical throughput/memory knee: batch 192 provided essentially no additional speedup while consuming more GPU memory.

### 6.2 DataLoader workers

Using 0, 2, or 4 workers produced nearly identical timings, around 59.2 s on the controlled subset.

**Conclusion:** the input pipeline was not a meaningful bottleneck, so `num_workers=0` was retained for simplicity and reproducibility.

### 6.3 BF16

BF16 produced the largest measured optimization.

| Precision | Mean training time | Peak allocated GPU memory |
|---|---:|---:|
| Default precision | ~59.64 s | ~12.46 GiB |
| BF16 | ~12.26 s | ~8.61 GiB |

This corresponds to approximately a **4.87× speedup** and a substantial reduction in memory usage.

### 6.4 TF32

TF32 substantially accelerated the default-precision path:

| Configuration | Mean training time |
|---|---:|
| TF32 disabled | ~59.63 s |
| TF32 enabled | ~19.55 s |

However, BF16 remained faster and more memory-efficient, so BF16 was selected for the official baseline.

### 6.5 `torch.compile`

`torch.compile` was tested with BF16 and batch size 160.

| Mode | Mean training time |
|---|---:|
| Eager | ~87.30 s |
| Compiled | ~151.31 s |

For the tested one-epoch workload, compilation overhead dominated and made the compiled path slower. A longer three-epoch experiment reduced the relative penalty, but compilation remained slower overall.

A recompile was also observed when the final batch had a different shape. Using `drop_last=True` removed that irregular batch, but the gain was not sufficient to justify changing the official baseline.

**Decision:** keep `torch.compile` disabled for the reference run.

---

## 7. Profiling Results

A representative BF16, batch-160 run was sampled approximately once per second.

Key observations:

| Metric | Representative value |
|---|---:|
| Mean GPU utilization | ~89.6% |
| GPU utilization ≥80% | ~97.6% of GPU samples |
| Peak GPU memory | ~26.8 GiB |
| Mean GPU power | ~238 W |

Additional profiling collected GPU memory usage, temperature, SM clock, power draw, and RSS memory for the training process.

The sustained GPU utilization and the absence of measurable DataLoader gains support the conclusion that the timed training phase is **primarily GPU-bound**.

---

## 8. Official Baseline Results

Three valid full-SQuAD repetitions were executed with the final configuration.

| Repetition | Training time |
|---|---:|
| 1 | 233.48 s |
| 2 | 235.41 s |
| 3 | 234.85 s |

### Summary statistics

| Statistic | Value |
|---|---:|
| Mean | **234.58 s** |
| Median | 234.85 s |
| Sample standard deviation | 0.99 s |
| Coefficient of variation | **0.42%** |

Therefore, the final single-A100 baseline is approximately:

> **234.6 seconds = 3.91 minutes per full SQuAD epoch**

The low coefficient of variation indicates good run-to-run timing stability.

---

## 9. Reproducibility

All valid performance runs enforce exactly one visible CUDA GPU.

The random seed used for the experiments is **42**. Small numerical differences may still occur because GPU kernels and framework internals are not guaranteed to be bitwise deterministic.

Generated results, checkpoints, TensorBoard files, virtual environments, and model weights are intentionally excluded from Git.

### Repository structure

```text
BASELINE/
├── README.md
├── README_ES.md
├── train.py
├── baseline.slurm
├── run_baseline.sh
├── profile.slurm
└── profile_run.sh
```

- `train.py` — official single-A100 baseline implementation.
- `baseline.slurm` — official SLURM job for the baseline.
- `run_baseline.sh` — convenience wrapper for submitting the baseline job.
- `profile.slurm` — SLURM job for the profiling run.
- `profile_run.sh` — resource-sampling wrapper used during profiling.

---

## 10. How to Run

From the `BASELINE` directory:

```bash
./run_baseline.sh
```

or directly:

```bash
sbatch baseline.slurm
```

---

## 11. Conclusion

The final reference configuration is:

```text
BERT-base-uncased
SQuAD v1.1
1 × NVIDIA A100-PCIE-40GB
BF16
batch size 160
1 epoch
~234.6 s training time
```

The main optimization was BF16, which delivered an approximately **4.87× speedup** on the controlled subset. DataLoader parallelism did not provide measurable benefit, and `torch.compile` was not advantageous for the short benchmark because compilation overhead dominated.

The final timing baseline is stable, with a **0.42% coefficient of variation**.

This baseline therefore provides both:

- a reproducible **single-GPU HPC performance reference**; and
