#!/usr/bin/env python
"""
BERT-base / SQuAD v1.1 fine-tuning with native PyTorch DistributedDataParallel.

HPC Tools (Master HPC, UDC) - Deliverable 2: distributed training on CESGA FT3.

Same model, preprocessing and hyper-parameters as BASELINE/train.py
(BF16, batch 160 per GPU, lr 3e-5, weight decay 0.01, fused AdamW,
linear schedule, grad-norm clipping 1.0, seed 42), but with an explicit
DDP training loop instead of the Hugging Face Trainer:

  * one process per GPU, launched by `srun` (SLURM_PROCID -> rank) or torchrun
  * NCCL backend (NVLink/PCIe/SHM inside a node, InfiniBand between nodes)
  * DistributedSampler: each rank processes 1/WORLD_SIZE of the data
  * DistributedDataParallel: gradients all-reduced (averaged) every step
  * rank 0 alone writes checkpoints and the results JSON
"""

import argparse
import datetime
import glob
import json
import os
import shutil
import socket
import time
from contextlib import contextmanager, nullcontext
from functools import partial

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import datasets
import transformers
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 384
DOC_STRIDE = 128
BASELINE_SECONDS = 234.58  # Deliverable 1: mean of 3 runs, 1 x A100, BF16, batch 160
EXPECTED_FEATURES = 88_492  # full SQuAD v1.1 train split after sliding-window tokenisation
MODEL_COLUMNS = [
    "input_ids",
    "token_type_ids",
    "attention_mask",
    "start_positions",
    "end_positions",
]


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="BERT/SQuAD fine-tuning with PyTorch DDP")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=160, help="per-GPU (per-rank) batch size")
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="0 disables clipping")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers per rank")
    p.add_argument("--bf16", action="store_true", help="BF16 autocast (same as baseline)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests only)")
    p.add_argument("--max-train-samples", type=int, default=None, help="subset of SQuAD (tests only)")
    p.add_argument(
        "--tokenized-dir",
        type=str,
        default="data/squad_bert-base-uncased_len384_stride128",
        help="pre-tokenised dataset (created by rank 0 if missing)",
    )
    p.add_argument("--prepare-data-only", action="store_true", help="tokenise, save, exit (no GPU needed)")
    p.add_argument("--skip-save", action="store_true", help="do not write the fine-tuned model")
    p.add_argument("--output-dir", type=str, default="results/manual")
    p.add_argument("--baseline-seconds", type=float, default=BASELINE_SECONDS)
    # Optional communication experiments
    p.add_argument("--bucket-cap-mb", type=int, default=25, help="DDP gradient bucket size (default 25)")
    p.add_argument("--bf16-grad-compress", action="store_true", help="all-reduce gradients in BF16")
    p.add_argument("--torch-profile", action="store_true", help="PyTorch profiler on steps 11-18")
    return p.parse_args()


# --------------------------------------------------------------------------
# Distributed helpers
# --------------------------------------------------------------------------
def setup_distributed():
    """Initialise torch.distributed from torchrun or SLURM environment variables.

    torchrun sets RANK / LOCAL_RANK / WORLD_SIZE; under plain `srun` we fall back
    to SLURM_PROCID / SLURM_LOCALID / SLURM_NTASKS. MASTER_ADDR and MASTER_PORT
    must be exported by the launcher (see ddp.slurm).
    """
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    # SLURM either exposes every GPU of the node to each task
    # (CUDA_VISIBLE_DEVICES=0,1 -> pick by local rank) or binds one GPU per
    # task (CUDA_VISIBLE_DEVICES=<one id> -> device 0). Support both.
    n_visible = torch.cuda.device_count()
    device_index = local_rank if n_visible > 1 else 0
    if device_index >= n_visible:
        raise RuntimeError(
            f"rank {rank}: local_rank={local_rank} but only {n_visible} GPU(s) visible "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})"
        )
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)

    if world_size == 1:  # allow `python train_ddp.py` without a launcher
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=30),
        device_id=device,  # binds this process to its GPU (eager NCCL init)
    )
    return rank, local_rank, world_size, device


def barrier(device):
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier(device_ids=[device.index])


@contextmanager
def rank0_first(rank, device):
    """Rank 0 runs the block first (downloads, tokenisation); the rest wait, then run it."""
    if rank != 0:
        barrier(device)
    yield
    if rank == 0:
        barrier(device)


def ib_counters():
    """Bytes sent/received by this node's InfiniBand ports (sysfs counts 4-byte words).

    Includes all traffic of the node (e.g. Lustre), so read it only around the
    timed loop. Returns (None, None) if the counters are not exposed.
    """
    tx = rx = 0
    found = False
    for d in glob.glob("/sys/class/infiniband/*/ports/*/counters"):
        try:
            with open(os.path.join(d, "port_xmit_data")) as f:
                tx += int(f.read()) * 4
            with open(os.path.join(d, "port_rcv_data")) as f:
                rx += int(f.read()) * 4
            found = True
        except (OSError, ValueError):
            continue
    return (tx, rx) if found else (None, None)


def nccl_version():
    v = torch.cuda.nccl.version()
    return ".".join(map(str, v)) if isinstance(v, tuple) else str(v)


def log0(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def describe_rank(rank, local_rank, world_size, device):
    props = torch.cuda.get_device_properties(device)
    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_index": device.index,
        "gpu_name": props.name,
        "gpu_uuid": str(getattr(props, "uuid", f"{os.environ.get('CUDA_VISIBLE_DEVICES')}:{device.index}")),
    }


def check_gpu_mapping(info, world_size):
    """Every rank must own a different physical GPU, otherwise NCCL hangs or errors."""
    everyone = [None] * world_size
    dist.all_gather_object(everyone, info)
    seen = {}
    for r in everyone:
        key = (r["host"], r["gpu_uuid"])
        if key in seen:
            raise RuntimeError(f"ranks {seen[key]} and {r['rank']} share GPU {key}")
        seen[key] = r["rank"]
    return everyone


# --------------------------------------------------------------------------
# Data (preprocessing identical to BASELINE/train.py)
# --------------------------------------------------------------------------
def preprocess_examples(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    tokenized = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    start_positions, end_positions = [], []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(i)
        answers = examples["answers"][sample_mapping[i]]

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        answer_start_char = answers["answer_start"][0]
        answer_end_char = answer_start_char + len(answers["text"][0])

        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if (offsets[token_start_index][0] > answer_start_char
                or offsets[token_end_index][1] < answer_end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        while (token_start_index < len(offsets)
               and offsets[token_start_index][0] <= answer_start_char):
            token_start_index += 1
        start_positions.append(token_start_index - 1)

        while token_end_index >= 0 and offsets[token_end_index][1] >= answer_end_char:
            token_end_index -= 1
        end_positions.append(token_end_index + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def tokenized_dir_for(args):
    suffix = f"_n{args.max_train_samples}" if args.max_train_samples else ""
    return os.path.abspath(args.tokenized_dir + suffix)


def prepare_tokenized_dataset(tokenizer, tok_dir, max_train_samples):
    """Tokenise SQuAD once and save it to disk (atomic rename, safe if two jobs race)."""
    if os.path.isdir(tok_dir):
        print(f"Tokenised dataset already present: {tok_dir}", flush=True)
        return
    print(f"Tokenising SQuAD -> {tok_dir}", flush=True)
    raw = load_dataset("rajpurkar/squad", split="train")
    if max_train_samples is not None:
        raw = raw.select(range(min(max_train_samples, len(raw))))
    tokenized = raw.map(
        partial(preprocess_examples, tokenizer=tokenizer),
        batched=True,
        remove_columns=raw.column_names,
        keep_in_memory=True,
    )
    os.makedirs(os.path.dirname(tok_dir), exist_ok=True)
    tmp_dir = f"{tok_dir}.tmp-{socket.gethostname()}-{os.getpid()}"
    tokenized.save_to_disk(tmp_dir)
    try:
        os.rename(tmp_dir, tok_dir)
    except OSError:
        if not os.path.isdir(tok_dir):
            raise
        shutil.rmtree(tmp_dir, ignore_errors=True)  # another job won the race
    print(f"Saved {len(tokenized)} features from {len(raw)} examples", flush=True)


# --------------------------------------------------------------------------
# Optimiser (mirrors HF Trainer defaults used in the baseline)
# --------------------------------------------------------------------------
def build_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Trainer does not decay biases or LayerNorm weights
        if "bias" in name or "layernorm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8, fused=True)


def make_profiler(enabled, rank, output_dir):
    if not enabled:
        return nullcontext(None)
    from torch.profiler import ProfilerActivity, profile, schedule

    trace_dir = os.path.join(output_dir, "torch_profile")
    os.makedirs(trace_dir, exist_ok=True)

    def on_trace_ready(prof):
        prof.export_chrome_trace(os.path.join(trace_dir, f"rank{rank}_step{prof.step_num}.json"))
        if rank == 0:
            for key in ("self_device_time_total", "self_cuda_time_total"):
                try:
                    print(prof.key_averages().table(sort_by=key, row_limit=25), flush=True)
                    break
                except Exception:  # sort key name differs between PyTorch versions
                    continue

    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=10, warmup=3, active=5, repeat=1),
        on_trace_ready=on_trace_ready,
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    args = parse_args()
    tok_dir = tokenized_dir_for(args)

    # Data preparation can run on a CPU node, without torch.distributed.
    if args.prepare_data_only:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        prepare_tokenized_dataset(tokenizer, tok_dir, args.max_train_samples)
        return

    rank, local_rank, world_size, device = setup_distributed()
    info = describe_rank(rank, local_rank, world_size, device)
    print(
        f"[rank {rank}/{world_size}] host={info['host']} local_rank={local_rank} "
        f"CUDA_VISIBLE_DEVICES={info['cuda_visible_devices']} device=cuda:{device.index} "
        f"({info['gpu_name']})",
        flush=True,
    )
    check_gpu_mapping(info, world_size)

    log0(rank, "\n=== Environment ===")
    log0(rank, f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | "
               f"NCCL {nccl_version()}")
    log0(rank, f"Transformers {transformers.__version__} | Datasets {datasets.__version__}")
    log0(rank, f"World size {world_size} | MASTER_ADDR={os.environ.get('MASTER_ADDR')} "
               f"MASTER_PORT={os.environ.get('MASTER_PORT')}")
    log0(rank, "\n=== Configuration ===")
    log0(rank, f"Per-GPU batch {args.batch_size} | global batch {args.batch_size * world_size} | "
               f"lr {args.learning_rate} | epochs {args.epochs} | BF16 {args.bf16} | "
               f"workers {args.num_workers} | bucket {args.bucket_cap_mb} MB | "
               f"bf16 grad compress {args.bf16_grad_compress}")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # Rank 0 downloads / tokenises first; the others then read from cache / disk.
    with rank0_first(rank, device):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
        if rank == 0:
            prepare_tokenized_dataset(tokenizer, tok_dir, args.max_train_samples)

    train_dataset = load_from_disk(tok_dir).with_format("torch", columns=MODEL_COLUMNS)
    n_features = len(train_dataset)
    log0(rank, f"Tokenised training features: {n_features}")
    if args.max_train_samples is None and n_features != EXPECTED_FEATURES:
        log0(rank, f"WARNING: expected {EXPECTED_FEATURES} features (baseline), got {n_features}")

    model.to(device)
    ddp_model = DDP(
        model,
        device_ids=[device.index],
        output_device=device.index,
        bucket_cap_mb=args.bucket_cap_mb,
        find_unused_parameters=False,  # BertForQuestionAnswering uses every parameter
    )
    if args.bf16_grad_compress:
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
        ddp_model.register_comm_hook(state=None, hook=default_hooks.bf16_compress_hook)

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: max(0.0, (total_steps - s) / max(1, total_steps))
    )
    log0(rank, f"Samples per rank {len(sampler)} | steps per epoch per rank {steps_per_epoch} | "
               f"total steps {total_steps}")

    # ------------------------------------------------------------------ train
    log0(rank, "\n=== Starting training ===")
    ddp_model.train()
    step = 0
    loss_sum = torch.zeros((), device=device)  # since last log line
    loss_total = torch.zeros((), device=device)  # whole run (epoch-average loss)
    loss_count = 0
    last_loss = float("nan")

    barrier(device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    ib_before = ib_counters() if local_rank == 0 else (None, None)
    t0 = time.perf_counter()

    with make_profiler(args.torch_profile, rank, args.output_dir) as prof:
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)  # different shuffle each epoch, same on every rank
            for batch in loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                    loss = ddp_model(**batch).loss
                loss.backward()  # DDP all-reduces gradient buckets during backward
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                step += 1
                loss_sum += loss.detach()  # stays on GPU: no host sync per step
                loss_total += loss.detach()
                loss_count += 1
                if prof is not None:
                    prof.step()

                if step % args.log_every == 0 or step == total_steps:
                    avg = loss_sum / loss_count
                    dist.all_reduce(avg, op=dist.ReduceOp.SUM)
                    last_loss = avg.item() / world_size
                    log0(rank, f"step {step:5d}/{total_steps} | loss {last_loss:.4f} | "
                               f"lr {scheduler.get_last_lr()[0]:.2e} | "
                               f"{time.perf_counter() - t0:7.1f} s")
                    loss_sum.zero_()
                    loss_count = 0
                if step >= total_steps:
                    break
            if step >= total_steps:
                break

    torch.cuda.synchronize(device)
    local_seconds = time.perf_counter() - t0  # this rank's own compute + communication
    barrier(device)
    wall_seconds = time.perf_counter() - t0  # slowest rank defines the job time
    ib_after = ib_counters() if local_rank == 0 else (None, None)

    dist.all_reduce(loss_total, op=dist.ReduceOp.SUM)
    mean_train_loss = loss_total.item() / (max(1, step) * world_size)  # comparable to Trainer's train_loss

    # ---------------------------------------------------------------- results
    info.update({
        "local_seconds": local_seconds,
        "wall_seconds": wall_seconds,
        "steps": step,
        "peak_mem_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_mem_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        # one value per node (local rank 0 only): InfiniBand bytes during the timed loop
        "ib_tx_bytes": None if ib_before[0] is None or ib_after[0] is None else ib_after[0] - ib_before[0],
        "ib_rx_bytes": None if ib_before[1] is None or ib_after[1] is None else ib_after[1] - ib_before[1],
    })
    everyone = [None] * world_size
    dist.all_gather_object(everyone, info)

    if not args.skip_save:
        if rank == 0:
            save_dir = os.path.join(args.output_dir, "model")
            ddp_model.module.save_pretrained(save_dir)  # unwrap: no "module." prefixes
            tokenizer.save_pretrained(save_dir)
            torch.save(
                {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "step": step, "args": vars(args)},
                os.path.join(save_dir, "training_state.pt"),
            )
            print(f"Model saved to {save_dir}", flush=True)
        barrier(device)  # nobody exits while rank 0 is still writing

    if rank == 0:
        job_seconds = max(r["wall_seconds"] for r in everyone)
        local = [r["local_seconds"] for r in everyone]
        full_run = args.max_steps is None or step == steps_per_epoch * args.epochs
        samples = n_features * args.epochs if full_run else step * args.batch_size * world_size
        hosts = sorted({r["host"] for r in everyone})
        speedup = args.baseline_seconds / job_seconds
        results = {
            "strategy": "PyTorch DistributedDataParallel (NCCL), srun-launched",
            "model": MODEL_NAME,
            "dataset": "rajpurkar/squad (v1.1) train",
            "nodes": len(hosts),
            "gpus_per_node": world_size // len(hosts),
            "world_size": world_size,
            "hosts": hosts,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "nccl_version": nccl_version(),
            "transformers_version": transformers.__version__,
            "datasets_version": datasets.__version__,
            "per_device_batch_size": args.batch_size,
            "global_batch_size": args.batch_size * world_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "max_train_samples": args.max_train_samples,
            "bf16": args.bf16,
            "seed": args.seed,
            "num_workers": args.num_workers,
            "bucket_cap_mb": args.bucket_cap_mb,
            "bf16_grad_compress": args.bf16_grad_compress,
            "torch_profile": args.torch_profile,
            "tokenized_train_features": n_features,
            "steps_per_epoch_per_rank": steps_per_epoch,
            "steps_run": step,
            "training_time_seconds": job_seconds,
            "rank_local_seconds_min": min(local),
            "rank_local_seconds_max": max(local),
            "rank_imbalance_seconds": max(local) - min(local),
            "samples_per_second": samples / job_seconds,
            "baseline_seconds": args.baseline_seconds,
            "speedup_vs_baseline": speedup,
            "efficiency_vs_baseline": speedup / world_size,
            "final_train_loss": last_loss,
            "mean_train_loss": mean_train_loss,
            "valid_benchmark": full_run and not args.torch_profile and args.max_train_samples is None,
            "env": {k: v for k, v in os.environ.items()
                    if k.startswith(("NCCL_", "MASTER_", "SLURM_JOB_NUM_NODES", "SLURM_NTASKS",
                                     "OMP_NUM_THREADS", "TORCH_"))},
            "ranks": everyone,
        }
        path = os.path.join(args.output_dir, "training_results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=4)

        print("\n=== Training finished ===")
        print(f"Training time (slowest rank): {job_seconds:.2f} s")
        print(f"Per-rank local time: min {min(local):.2f} s / max {max(local):.2f} s")
        print(f"Throughput: {samples / job_seconds:.1f} samples/s")
        print(f"Mean training loss over the run: {mean_train_loss:.4f} (last window {last_loss:.4f})")
        print(f"Speedup vs baseline {args.baseline_seconds:.2f} s: {speedup:.2f}x "
              f"(efficiency {100 * speedup / world_size:.1f}% on {world_size} GPUs)")
        print(f"Peak GPU memory allocated (max over ranks): "
              f"{max(r['peak_mem_allocated_gib'] for r in everyone):.2f} GiB")
        for r in everyone:
            if r["ib_tx_bytes"] is not None:
                print(f"InfiniBand during training on {r['host']}: TX {r['ib_tx_bytes'] / 1e9:.1f} GB, "
                      f"RX {r['ib_rx_bytes'] / 1e9:.1f} GB ({r['ib_tx_bytes'] / 1e6 / max(1, step):.0f} MB/step)")
        print(f"Results written to: {path}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
