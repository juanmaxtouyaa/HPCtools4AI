Reference: official baseline 234.58 s (HF Trainer, 1 x A100); DDP script on 1 GPU 216.73 s.

| Config | GPUs | Global batch | Runs | Mean time (s) | Std (s) | CV | Samples/s | Speedup vs baseline | Eff. vs baseline | Speedup vs DDP-1GPU | Eff. vs DDP-1GPU | GPU util | IB TX/node (GB) | Mean loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1n1g | 1 | 160 | 3 | 216.73 | 1.17 | 0.54% | 408 | 1.08x | 108.2% | 1.00x | 100.0% | 99.7% | 0.0 | 1.7158 |
| 1n2g | 2 | 320 | 3 | 112.37 | 1.57 | 1.40% | 788 | 2.09x | 104.4% | 1.93x | 96.4% | 99.3% | 0.0 | 2.0394 |
| 2n1g | 2 | 320 | 3 | 111.79 | 0.42 | 0.37% | 792 | 2.10x | 104.9% | 1.94x | 96.9% | 99.4% | 121.4 | 2.0405 |
| 2n2g | 4 | 640 | 3 | 57.24 | 0.63 | 1.10% | 1546 | 4.10x | 102.5% | 3.79x | 94.7% | 98.1% | 91.6 | 2.5548 |

All-reduce of one full FP32 gradient:

| Config | GPUs | Size (MB) | Time (ms) | algbw (GB/s) | busbw (GB/s) |
|---|---:|---:|---:|---:|---:|
| 1n2g | 2 | 435.6 | 34.0 | 12.81 | 12.81 |
| 2n1g | 2 | 435.6 | 40.3 | 10.80 | 10.80 |
| 2n2g | 4 | 435.6 | 42.9 | 10.16 | 15.24 |
