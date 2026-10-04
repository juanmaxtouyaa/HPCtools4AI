#!/usr/bin/env python
"""
NCCL all-reduce micro-benchmark for the same SLURM layouts as train_ddp.py.

Times dist.all_reduce for several message sizes up to one full BERT-base QA
gradient (108,893,186 FP32 values = 435.6 MB) and reports algorithm bandwidth
(bytes / time) and bus bandwidth (algbw * 2(n-1)/n, the nccl-tests convention).
The full-gradient time is the communication cost DDP must hide behind the
backward pass at every optimizer step.
"""

import argparse
import json
import time

import torch
import torch.distributed as dist

from train_ddp import barrier, log0, nccl_version, setup_distributed

BERT_QA_PARAMS = 108_893_186


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes-mb", type=float, nargs="+",
                   default=[1, 4, 25, 100, BERT_QA_PARAMS * 4 / 1e6])
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    rank, _, world_size, device = setup_distributed()
    if world_size < 2:
        raise SystemExit("all-reduce benchmark needs at least 2 ranks")
    dtype = getattr(torch, args.dtype)
    elem = torch.tensor([], dtype=dtype).element_size()
    factor = 2 * (world_size - 1) / world_size

    log0(rank, f"NCCL {nccl_version()} | world size {world_size} | dtype {args.dtype}")
    log0(rank, f"{'size MB':>10} {'time ms':>10} {'algbw GB/s':>11} {'busbw GB/s':>11}")
    rows = []
    for size_mb in args.sizes_mb:
        numel = int(size_mb * 1e6 / elem)
        x = torch.zeros(numel, dtype=dtype, device=device)
        for _ in range(args.warmup):
            dist.all_reduce(x)
        torch.cuda.synchronize(device)
        barrier(device)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            dist.all_reduce(x)
        torch.cuda.synchronize(device)
        t = torch.tensor((time.perf_counter() - t0) / args.iters, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)  # slowest rank
        seconds = t.item()
        nbytes = numel * elem
        algbw = nbytes / seconds / 1e9
        rows.append({"size_mb": nbytes / 1e6, "time_ms": seconds * 1e3,
                     "algbw_gbs": algbw, "busbw_gbs": algbw * factor})
        log0(rank, f"{nbytes / 1e6:10.1f} {seconds * 1e3:10.2f} {algbw:11.2f} {algbw * factor:11.2f}")
        del x

    if rank == 0 and args.output:
        with open(args.output, "w") as f:
            json.dump({"world_size": world_size, "dtype": args.dtype,
                       "nccl_version": nccl_version(), "results": rows}, f, indent=4)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
