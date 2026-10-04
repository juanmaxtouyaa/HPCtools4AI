#!/usr/bin/env python3
"""
Aggregate the benchmark runs into the scaling table used in the README.

Reads   results/runs/<config>/<tag>_job<id>/training_results.json  (+ gpu_<host>.csv)
        results/allreduce/<config>/job<id>/allreduce.json           (optional)
Writes  results/summary.csv, results/summary.md, results/scaling.png (if matplotlib)

  python summarize_results.py                 # valid full-epoch runs, any tag
  python summarize_results.py --tag bench     # only run directories starting with "bench"
"""

import argparse
import csv
import glob
import json
import os
import statistics

CONFIG_ORDER = ["1n1g", "1n2g", "2n1g", "2n2g"]


def load_runs(results_dir, tag, include_invalid):
    runs = []
    pattern = os.path.join(results_dir, "runs", "*", "*", "training_results.json")
    for path in sorted(glob.glob(pattern)):
        run_dir = os.path.dirname(path)
        if tag and not os.path.basename(run_dir).startswith(tag):
            continue
        with open(path) as f:
            r = json.load(f)
        if not include_invalid and not r.get("valid_benchmark", False):
            continue
        r["config"] = f"{r['nodes']}n{r['gpus_per_node']}g"
        r["run_dir"] = run_dir
        r.update(gpu_utilisation(run_dir))
        runs.append(r)
    return runs


def gpu_utilisation(run_dir):
    """Mean utilisation over 'active' samples (util >= 5 %) of every GPU sampled in the run."""
    active, high = [], 0
    for path in glob.glob(os.path.join(run_dir, "gpu_*.csv")):
        with open(path) as f:
            next(f, None)  # header
            for line in f:
                cols = [c.strip() for c in line.split(",")]
                try:
                    util = float(cols[2])
                except (IndexError, ValueError):
                    continue
                if util >= 5:
                    active.append(util)
                    high += util >= 80
    if not active:
        return {"gpu_util_mean": None, "gpu_util_ge80_pct": None}
    return {"gpu_util_mean": statistics.mean(active), "gpu_util_ge80_pct": 100 * high / len(active)}


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def summarise(runs, baseline_seconds):
    by_config = {}
    for r in runs:
        by_config.setdefault(r["config"], []).append(r)
    ordered = sorted(by_config, key=lambda c: (CONFIG_ORDER.index(c) if c in CONFIG_ORDER else 99, c))

    ref_1gpu = None
    if "1n1g" in by_config:
        ref_1gpu = statistics.mean(r["training_time_seconds"] for r in by_config["1n1g"])

    rows = []
    for cfg in ordered:
        rs = by_config[cfg]
        times = [r["training_time_seconds"] for r in rs]
        mean = statistics.mean(times)
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        world = rs[0]["world_size"]
        ib_tx = [rank["ib_tx_bytes"] for r in rs for rank in r["ranks"] if rank.get("ib_tx_bytes") is not None]
        steps = rs[0]["steps_run"]
        row = {
            "config": cfg,
            "nodes": rs[0]["nodes"],
            "gpus_per_node": rs[0]["gpus_per_node"],
            "world_size": world,
            "global_batch": rs[0]["global_batch_size"],
            "steps_per_rank": steps,
            "runs": len(rs),
            "mean_s": mean,
            "std_s": std,
            "cv_pct": 100 * std / mean,
            "min_s": min(times),
            "max_s": max(times),
            "samples_per_s": statistics.mean(r["samples_per_second"] for r in rs),
            "speedup_vs_baseline": baseline_seconds / mean,
            "efficiency_vs_baseline_pct": 100 * baseline_seconds / mean / world,
            "speedup_vs_ddp_1gpu": ref_1gpu / mean if ref_1gpu else None,
            "efficiency_vs_ddp_1gpu_pct": 100 * ref_1gpu / mean / world if ref_1gpu else None,
            "rank_imbalance_s": statistics.mean(r["rank_imbalance_seconds"] for r in rs),
            "peak_mem_gib": max(rank["peak_mem_allocated_gib"] for r in rs for rank in r["ranks"]),
            "final_loss": statistics.mean(r["final_train_loss"] for r in rs),
            "mean_loss": mean_or_none([r.get("mean_train_loss") for r in rs]),
            "gpu_util_mean_pct": mean_or_none([r["gpu_util_mean"] for r in rs]),
            "ib_tx_gb_per_node": statistics.mean(ib_tx) / 1e9 if ib_tx else None,
            "ib_tx_mb_per_step": statistics.mean(ib_tx) / 1e6 / steps if ib_tx else None,
        }
        rows.append(row)
    return rows, ref_1gpu


def fmt(v, spec, suffix=""):
    return "n/a" if v is None else format(v, spec) + suffix


def markdown(rows, baseline_seconds, ref_1gpu):
    lines = [
        f"Reference: official baseline {baseline_seconds:.2f} s (HF Trainer, 1 x A100); "
        f"DDP script on 1 GPU {fmt(ref_1gpu, '.2f')} s.",
        "",
        "| Config | GPUs | Global batch | Runs | Mean time (s) | Std (s) | CV | Samples/s "
        "| Speedup vs baseline | Eff. vs baseline | Speedup vs DDP-1GPU | Eff. vs DDP-1GPU "
        "| GPU util | IB TX/node (GB) | Mean loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['world_size']} | {r['global_batch']} | {r['runs']} "
            f"| {r['mean_s']:.2f} | {r['std_s']:.2f} | {r['cv_pct']:.2f}% | {r['samples_per_s']:.0f} "
            f"| {r['speedup_vs_baseline']:.2f}x | {r['efficiency_vs_baseline_pct']:.1f}% "
            f"| {fmt(r['speedup_vs_ddp_1gpu'], '.2f', 'x')} | {fmt(r['efficiency_vs_ddp_1gpu_pct'], '.1f', '%')} "
            f"| {fmt(r['gpu_util_mean_pct'], '.1f', '%')} | {fmt(r['ib_tx_gb_per_node'], '.1f')} "
            f"| {fmt(r['mean_loss'], '.4f')} |"
        )
    return "\n".join(lines)


def allreduce_table(results_dir):
    lines = []
    for path in sorted(glob.glob(os.path.join(results_dir, "allreduce", "*", "*", "allreduce.json"))):
        cfg = path.split(os.sep)[-3]
        with open(path) as f:
            data = json.load(f)
        biggest = max(data["results"], key=lambda x: x["size_mb"])
        lines.append(f"| {cfg} | {data['world_size']} | {biggest['size_mb']:.1f} | {biggest['time_ms']:.1f} "
                     f"| {biggest['algbw_gbs']:.2f} | {biggest['busbw_gbs']:.2f} |")
    if not lines:
        return ""
    head = ["", "All-reduce of one full FP32 gradient:", "",
            "| Config | GPUs | Size (MB) | Time (ms) | algbw (GB/s) | busbw (GB/s) |",
            "|---|---:|---:|---:|---:|---:|"]
    return "\n".join(head + lines)


def plot(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping plot (pip install matplotlib)")
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    gpus = [r["world_size"] for r in rows]
    top = max(gpus)
    ax.plot([1, top], [1, top], "--", color="grey", label="ideal (linear)")
    ax.plot(gpus, [r["speedup_vs_baseline"] for r in rows], "o", label="vs official baseline (Trainer)")
    if all(r["speedup_vs_ddp_1gpu"] for r in rows):
        ax.plot(gpus, [r["speedup_vs_ddp_1gpu"] for r in rows], "s", label="vs DDP script on 1 GPU")
    seen = {}
    for r in rows:  # 1n2g and 2n1g share x = 2: put their labels on opposite sides
        k = seen[r["world_size"]] = seen.get(r["world_size"], -1) + 1
        ax.annotate(r["config"], (r["world_size"], r["speedup_vs_baseline"]),
                    textcoords="offset points", xytext=(8, -12) if k % 2 else (-34, 4), fontsize=8)
    ax.set_xlabel("GPUs (A100-PCIE-40GB)")
    ax.set_ylabel("Speedup (time-to-epoch)")
    ax.set_title("BERT-base / SQuAD v1.1, 1 epoch, BF16, 160 per GPU")
    ax.set_xticks(sorted(set(gpus)))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Plot written to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--tag", default=None, help="only run directories starting with this prefix")
    p.add_argument("--baseline-seconds", type=float, default=234.58)
    p.add_argument("--include-invalid", action="store_true", help="also smoke/profiling runs")
    args = p.parse_args()

    runs = load_runs(args.results_dir, args.tag, args.include_invalid)
    if not runs:
        raise SystemExit(f"no runs found under {args.results_dir}/runs")
    rows, ref_1gpu = summarise(runs, args.baseline_seconds)

    csv_path = os.path.join(args.results_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md = markdown(rows, args.baseline_seconds, ref_1gpu) + "\n" + allreduce_table(args.results_dir)
    md_path = os.path.join(args.results_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write(md + "\n")
    print(md)
    print(f"\nWritten: {csv_path}, {md_path}")
    plot(rows, os.path.join(args.results_dir, "scaling.png"))


if __name__ == "__main__":
    main()
