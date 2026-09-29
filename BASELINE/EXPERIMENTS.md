# Single-A100 Experiments

This document summarizes the experiments used to construct the final BERT/SQuAD single-GPU baseline.

Unless stated otherwise, experiments use `google-bert/bert-base-uncased`, SQuAD, sequence length 384, document stride 128 and exactly one NVIDIA A100-PCIE-40GB.

## Important validity note

Initial pilot experiments used a SLURM allocation containing:

```bash
#SBATCH --gres=gpu:a100:1
#SBATCH --exclusive
```

but launched Python directly from the batch shell.

On the tested CESGA nodes this caused the batch environment to expose:

```text
CUDA_VISIBLE_DEVICES=0,1
torch.cuda.device_count()=2
```

Consequently, Hugging Face Trainer could use both A100 GPUs.

Those measurements are archived as `pilot_2gpu_invalid` and are not used as single-GPU performance results.

All valid experiments launch the training process through an explicit single-GPU SLURM step:

```bash
srun \
    --ntasks=1 \
    --cpus-per-task=32 \
    --gres=gpu:a100:1 \
    python ...
```

The Python code additionally aborts unless exactly one CUDA GPU is visible.

---

## 1. Batch-size sweep

Configuration:

```text
Precision       default
Training data   first 4,096 SQuAD examples
Epochs          1
Workers         4
```

Results:

| Batch | Time | Throughput | Peak allocated VRAM |
|---:|---:|---:|---:|
| 48 | 58.89 s | 71.03 samples/s | 12.46 GiB |
| 96 | 58.36 s | 71.67 samples/s | 23.56 GiB |
| 128 | 58.81 s | 71.13 samples/s | 30.92 GiB |

Throughput had already plateaued around batch 48 in this precision regime. Larger batches mainly increased memory consumption.

Batch-size comparisons are not pure hardware comparisons because changing batch size also changes the number of optimizer updates.

---

## 2. DataLoader workers

Configuration:

```text
Batch           48
Training data   4,096 examples
Epochs          1
```

Results:

| Workers | Time |
|---:|---:|
| 0 | 59.24 s |
| 2 | 59.20 s |
| 4 | 59.23 s |

The total spread is approximately 0.07%.

No meaningful DataLoader bottleneck was observed.

Tokenization occurs before the timed training phase, so these workers only feed already-tokenized features to the GPU.

`dataloader_num_workers=0` was therefore retained for the baseline.

---

## 3. Precision

Configuration:

```text
Batch           48
Workers         0
Training data   4,096 examples
```

Three repetitions were performed per precision regime.

Default-precision times:

```text
59.37 s
59.82 s
59.74 s
mean ≈ 59.64 s
```

BF16 times:

```text
12.29 s
12.23 s
12.25 s
mean ≈ 12.26 s
```

BF16 therefore provided approximately:

```text
4.87× speedup
~79.5 % reduction in training time
```

Peak allocated CUDA memory also decreased from approximately 12.46 GiB to 8.61 GiB.

BF16 was retained for subsequent A100 experiments.

---

## 4. BF16 batch-size retuning

After enabling BF16, larger batches became possible.

| Batch | Time | Peak allocated | Peak reserved |
|---:|---:|---:|---:|
| 96 | 12.12 s | 15.69 GiB | 15.99 GiB |
| 128 | 11.98 s | 20.42 GiB | 20.79 GiB |
| 160 | 11.18 s | 25.20 GiB | 25.67 GiB |
| 192 | 11.18 s | 29.92 GiB | 30.49 GiB |

Batch 192 provided essentially no additional speedup over batch 160 while consuming approximately 4.8 GiB more VRAM.

Batch 160 was therefore selected as the throughput/memory knee.

---

## 5. TF32

The FP32 path was tested with TF32 explicitly disabled and enabled.

Configuration:

```text
Batch           48
Workers         0
Training data   4,096 examples
BF16            disabled
```

IEEE FP32:

```text
59.75 s
59.48 s
59.67 s
mean ≈ 59.63 s
```

TF32:

```text
19.37 s
19.67 s
19.60 s
mean ≈ 19.55 s
```

TF32 produced approximately a 3.05× speedup over the IEEE FP32 path.

BF16 nevertheless remained approximately 1.6× faster than TF32 for this workload and also used less memory.

The final baseline therefore uses BF16.

---

## 6. `torch.compile`

Configuration:

```text
Training subset    32,768 original examples
Tokenized features 33,071
Batch              160
Precision          BF16
Workers            0
Epochs             1
```

Eager execution, three repetitions:

```text
87.36 s
87.21 s
87.34 s
mean ≈ 87.30 s
```

`torch.compile`, cold cache:

```text
152.64 s
147.13 s
154.18 s
mean ≈ 151.31 s
```

For a one-epoch workload, compilation was approximately 73% slower end-to-end because compilation overhead dominated.

Memory usage decreased:

```text
Eager reserved      ~25.67 GiB
Compiled reserved   ~22.58 GiB
```

A longer three-epoch experiment gave:

```text
Eager       261.42 s
Compiled    297.00 s
```

The compiled model appeared faster after its initial compilation overhead, but the total three-epoch runtime was still slower.

### Recompilation finding

For 33,071 tokenized features at batch 160:

```text
206 full batches × 160
+ final batch of 111
= 207 batches/epoch
```

PyTorch logged a recompilation triggered by:

```text
input_ids batch dimension:
expected 160
actual   111
```

This established that the irregular final batch changes the input shape and can trigger an additional compiled graph.

---

## 7. System profiling

A representative run used:

```text
Training subset    32,768 original examples
Features           33,071
Batch              160
BF16               enabled
Workers            0
Epochs             1
```

Training time:

```text
87.59 s
```

Peak PyTorch memory:

```text
Allocated    ~25.20 GiB
Reserved     ~25.67 GiB
```

GPU samples showed:

```text
Mean GPU utilization        89.6 %
Median GPU utilization      89 %
Maximum GPU utilization     100 %

GPU utilization >= 50 %     98.8 % of samples
GPU utilization >= 80 %     97.6 % of samples
GPU utilization >= 90 %     47.6 % of samples
```

Mean power draw was approximately 238 W, with a recorded maximum around 329 W.

No clear CPU, DataLoader or system-memory bottleneck was observed during the timed training phase.

---

## 8. Final full-SQuAD performance benchmark

Selected configuration:

```text
Model               google-bert/bert-base-uncased
Dataset             full SQuAD
Original examples   87,599
Tokenized features  88,492
GPU                 1 × A100-PCIE-40GB
Epochs              1
Batch               160
Learning rate       3e-5
Workers             0
Precision           BF16
torch.compile       disabled
```

Three valid repetitions:

| Rep | Training time |
|---:|---:|
| 1 | 233.48 s |
| 2 | 235.41 s |
| 3 | 234.85 s |

Statistics:

```text
Mean              234.58 s
Median            234.85 s
Sample std.         0.99 s
Coefficient var.    0.42 %
```

This is the official single-A100 performance baseline.

---

## 9. Compile × `drop_last` diagnostic with validation

A 2×2 experiment tested:

```text
eager_keep
eager_drop
compile_keep
compile_drop
```

All runs used:

```text
Training subset     32,768 examples
Batch               160
BF16                enabled
Maximum epochs      10
Validation          each epoch
Early stopping      patience 1
```

All four runs selected epoch 3 as the best checkpoint according to validation loss and stopped after epoch 4.

| Run | Best eval loss | EM | F1 | Train call wall | Non-eval wall | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|
| eager_keep | 1.21579 | 74.32 | 83.35 | 405.34 s | 363.93 s | 25.67 GiB |
| eager_drop | 1.18549 | 74.36 | 83.18 | 383.83 s | 346.24 s | 25.67 GiB |
| compile_keep | 1.19879 | 74.25 | 83.10 | 420.06 s | 382.25 s | 22.58 GiB |
| compile_drop | 1.20061 | 74.91 | 83.80 | 419.67 s | 377.80 s | 22.58 GiB |

The compiled variants again used less CUDA memory but were not faster end-to-end.

`compile_keep` logged the same batch-shape recompilation:

```text
expected batch dimension: 160
actual final batch:       111
```

`drop_last=True` removes this irregular training batch, but it also discards 111 features per epoch for this subset.

The quality differences between the four configurations are small and are not interpreted as evidence that compilation or `drop_last` improves SQuAD accuracy.

---

## 10. Full-SQuAD quality-control run

The large batch used by the performance baseline was selected for A100 throughput, not specifically for machine-learning generalization.

A separate quality-oriented run therefore used the complete SQuAD training set and a configuration close to the classical BERT/SQuAD fine-tuning recipe:

```text
Train examples       87,599
Train features       88,492
Validation examples  10,570
Validation features  10,753

Batch size           12
Eval batch           160
Epochs               2
Learning rate        3e-5
Warmup               1,475 steps (~10 %)
Weight decay         0.01
Precision            BF16
Optimizer            fused AdamW
LR scheduler         linear
torch.compile        disabled
drop_last            disabled
```

Training runtime:

```text
trainer.train() wall time       666.74 s
validation during training       21.13 s
non-evaluation train-call time  645.62 s

Trainer throughput:
265.53 samples/s
22.13 steps/s
```

Validation loss:

```text
Epoch 1    0.98230
Epoch 2    0.9893
```

The best checkpoint according to validation loss was epoch 1.

Final SQuAD validation metrics for the best model:

```text
Exact Match    79.56
F1             87.50
```

Peak CUDA memory was much lower than the throughput baseline:

```text
Maximum allocated    3.30 GiB
Maximum reserved     5.26 GiB
```

This experiment confirms that optimizing the A100 for maximum throughput and optimizing the model for validation quality are distinct objectives.

---

## Final configuration choice

The official HPC baseline remains:

```text
1 × A100-PCIE-40GB
full SQuAD
BF16
batch 160
workers 0
1 epoch
torch.compile OFF
training time ≈ 234.6 s
```

The independent quality sanity check achieved:

```text
EM 79.56
F1 87.50
```

The distributed phase should therefore compare against the fixed single-A100 performance baseline rather than changing the machine-learning workload during scaling measurements.
