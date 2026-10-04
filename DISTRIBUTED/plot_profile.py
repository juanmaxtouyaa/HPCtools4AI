#!/usr/bin/env python3
"""
Timeline of one DDP training step from a PyTorch profiler trace (train_ddp.py --torch-profile).

Draws, for one ProfilerStep of one rank, the GPU kernels on the compute stream and the
NCCL all-reduce kernels on the communication stream, plus the forward / backward /
optimizer phases (GPU-side annotations). Compute-stream idle time while an all-reduce
is running is highlighted: it is the communication DDP could not hide behind backward.

  python plot_profile.py results/runs/2n2g/profile_job<id>/torch_profile/rank0_step18.json \
         --step 15 --output results/profile_2n2g.png
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


def merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def intersect(a, b):
    """Intersection of two merged interval lists."""
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            out.append([s, e])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def complement(intervals, lo, hi):
    out, cur = [], lo
    for s, e in intervals:
        if s > cur:
            out.append([cur, s])
        cur = max(cur, e)
    if cur < hi:
        out.append([cur, hi])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace")
    p.add_argument("--step", type=int, default=15, help="ProfilerStep number to draw")
    p.add_argument("--output", default="results/profile_2n2g.png")
    p.add_argument("--title", default="2 nodes x 2 A100, rank 0")
    args = p.parse_args()

    with open(args.trace) as f:
        events = json.load(f)["traceEvents"]

    gpu_ann = [e for e in events if e.get("cat") == "gpu_user_annotation" and e.get("ph") == "X"]
    step = next(e for e in gpu_ann if e["name"] == f"ProfilerStep#{args.step}")
    t0, t1 = step["ts"], step["ts"] + step["dur"]

    def inside(e):
        return t0 <= e["ts"] < t1

    kernels = [e for e in events if e.get("cat") == "kernel" and inside(e)]
    nccl_all = [e for e in kernels if "nccl" in e["name"].lower()]
    nccl = [e for e in nccl_all if "allreduce" in e["name"].lower()]  # skip DDP's tiny buffer broadcast
    compute = [e for e in kernels if "nccl" not in e["name"].lower()]
    fwd = next(e for e in gpu_ann if e["name"] == "DistributedDataParallel.forward" and inside(e))
    opt = next(e for e in gpu_ann if e["name"].startswith("Optimizer.step") and inside(e))

    ms = lambda ts: (ts - t0) / 1e3  # noqa: E731  (trace timestamps are in microseconds)
    comp_iv = merge([[e["ts"], e["ts"] + e["dur"]] for e in compute])
    nccl_iv = merge([[e["ts"], e["ts"] + e["dur"]] for e in nccl])
    exposed = intersect(complement(comp_iv, t0, t1), nccl_iv)
    # Tail: exposed time after the last compute kernel of >= 1 ms that ends before the last
    # all-reduce finishes, i.e. the wait at the end of backward, before clipping/optimizer.
    last_ar_end = max(e for _, e in nccl_iv)
    long_comp = [e["ts"] + e["dur"] for e in compute
                 if e["dur"] >= 1000 and e["ts"] + e["dur"] <= last_ar_end]
    tail_start = max(long_comp) if long_comp else t0
    tail = intersect(exposed, [[tail_start, last_ar_end]])

    fwd_end, opt_start = fwd["ts"] + fwd["dur"], opt["ts"]
    nccl_total = sum(e - s for s, e in nccl_iv) / 1e3
    exposed_total = sum(e - s for s, e in exposed) / 1e3
    tail_total = sum(e - s for s, e in tail) / 1e3

    fig, ax = plt.subplots(figsize=(11, 3.6))
    phases = [
        (fwd["ts"], fwd_end, "forward", "#4C72B0"),
        (fwd_end, opt_start, "backward + grad clipping", "#55A868"),
        (opt_start, opt["ts"] + opt["dur"], "optimizer (fused AdamW)", "#8172B2"),
    ]
    for s, e, label, color in phases:
        ax.broken_barh([(ms(s), (e - s) / 1e3)], (2.1, 0.7), color=color, alpha=0.85)
        if (e - s) / 1e3 > 25:
            ax.text(ms(s) + (e - s) / 2e3, 2.45, label, ha="center", va="center", color="white", fontsize=9)

    ax.broken_barh([(ms(s), (e - s) / 1e3) for s, e in comp_iv], (1.1, 0.7), color="#55A868")
    ax.broken_barh([(ms(s), (e - s) / 1e3) for s, e in nccl_iv], (0.1, 0.7), color="#DD8452")
    for s, e in exposed:
        ax.broken_barh([(ms(s), (e - s) / 1e3)], (1.1, 0.7), color="#C44E52", alpha=0.6)
    ax.axvspan(ms(tail_start), ms(last_ar_end), color="#C44E52", alpha=0.18, lw=0)
    ax.annotate(f"exposed tail\n{tail_total:.1f} ms", xy=(ms(tail_start), 0.95),
                xytext=(ms(tail_start) - 45, 0.95), fontsize=8, va="center", ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.8))

    ax.set_yticks([0.45, 1.45, 2.45])
    ax.set_yticklabels(["NCCL stream\n(all-reduce)", "Compute stream\n(kernels)", "Phase"])
    ax.set_xlim(0, (t1 - t0) / 1e3)
    ax.set_ylim(0, 3)
    ax.set_xlabel("time within the training step (ms)")
    ax.set_title(
        f"BERT-base DDP step {args.step}, {args.title}: step {(t1 - t0) / 1e3:.0f} ms, "
        f"all-reduce busy {nccl_total:.0f} ms, exposed {exposed_total:.1f} ms (tail {tail_total:.1f} ms)",
        fontsize=10,
    )
    ax.legend(
        handles=[
            Patch(color="#55A868", label="compute kernels"),
            Patch(color="#DD8452", label="NCCL all-reduce kernels"),
            Patch(color="#C44E52", alpha=0.6, label="compute idle, waiting for all-reduce"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=8, frameon=False,
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)

    print(f"step {args.step}: {(t1 - t0) / 1e3:.1f} ms | forward {fwd['dur'] / 1e3:.1f} ms | "
          f"backward+clip {(opt_start - fwd_end) / 1e3:.1f} ms | optimizer {opt['dur'] / 1e3:.1f} ms")
    print(f"all-reduce kernels: {len(nccl)} (+{len(nccl_all) - len(nccl)} other NCCL), busy {nccl_total:.1f} ms, "
          f"first starts at {ms(nccl_iv[0][0]):.1f} ms, last ends at {ms(last_ar_end):.1f} ms "
          f"(optimizer starts at {ms(opt_start):.1f} ms)")
    print(f"exposed (compute idle while all-reduce runs): {exposed_total:.1f} ms "
          f"= {100 * exposed_total / nccl_total:.0f}% of all-reduce time; "
          f"hidden {100 - 100 * exposed_total / nccl_total:.0f}%")
    print(f"  of which tail before the optimizer: {tail_total:.1f} ms; "
          f"short waits during backward: {exposed_total - tail_total:.1f} ms")
    print(f"Figure written to {args.output}")


if __name__ == "__main__":
    main()
