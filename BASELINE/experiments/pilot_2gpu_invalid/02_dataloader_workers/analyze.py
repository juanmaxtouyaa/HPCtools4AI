import glob
import json
import statistics

WORKERS = [0, 2, 4, 8]

print()
print("DataLoader worker repetitions")
print("=" * 85)

for workers in WORKERS:
    paths = sorted(
        glob.glob(
            f"outputs/worker_sweep/w{workers}_r*/training_results.json"
        )
    )

    times = []
    throughputs = []
    vrams = []

    print(f"\nWorkers {workers}")
    print("-" * 85)

    for path in paths:
        with open(path) as f:
            data = json.load(f)

        rep = path.split("_r")[1].split("/")[0]

        training_time = data["training_time_seconds"]
        throughput = data["trainer_metrics"]["train_samples_per_second"]
        vram = data["peak_gpu_memory_reserved_gib"]

        times.append(training_time)
        throughputs.append(throughput)
        vrams.append(vram)

        print(
            f"r{rep}: "
            f"time={training_time:.2f} s, "
            f"throughput={throughput:.2f} samples/s, "
            f"VRAM={vram:.2f} GiB"
        )

    print()
    print(f"Mean time:         {statistics.mean(times):.2f} s")
    print(f"Median time:       {statistics.median(times):.2f} s")
    print(f"Std time:          {statistics.stdev(times):.2f} s")
    print(
        f"Mean throughput:   "
        f"{statistics.mean(throughputs):.2f} samples/s"
    )
    print(
        f"Median throughput: "
        f"{statistics.median(throughputs):.2f} samples/s"
    )
    print(f"Mean VRAM:         {statistics.mean(vrams):.2f} GiB")
