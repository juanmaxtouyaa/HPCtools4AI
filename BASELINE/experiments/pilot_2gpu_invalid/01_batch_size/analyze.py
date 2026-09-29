import glob
import json
import statistics

BATCHES = [48, 96, 128]

print()
print("Batch-size repetitions")
print("=" * 85)

for batch in BATCHES:
    paths = sorted(
        glob.glob(
            f"outputs/batch_sweep/b{batch}_r*/training_results.json"
        )
    )

    times = []
    throughputs = []
    allocated = []
    reserved = []

    print(f"\nBatch {batch}")
    print("-" * 85)

    for path in paths:
        with open(path) as f:
            data = json.load(f)

        rep = path.split("_r")[1].split("/")[0]

        training_time = data["training_time_seconds"]
        throughput = data["trainer_metrics"][
            "train_samples_per_second"
        ]
        vram_allocated = data[
            "peak_gpu_memory_allocated_gib"
        ]
        vram_reserved = data[
            "peak_gpu_memory_reserved_gib"
        ]

        times.append(training_time)
        throughputs.append(throughput)
        allocated.append(vram_allocated)
        reserved.append(vram_reserved)

        print(
            f"r{rep}: "
            f"time={training_time:.2f} s, "
            f"throughput={throughput:.2f} samples/s, "
            f"VRAM={vram_reserved:.2f} GiB"
        )

    print()
    print(
        f"Mean time:        {statistics.mean(times):.2f} s"
    )
    print(
        f"Median time:      {statistics.median(times):.2f} s"
    )
    print(
        f"Std time:         {statistics.stdev(times):.2f} s"
    )
    print(
        f"Mean throughput:  "
        f"{statistics.mean(throughputs):.2f} samples/s"
    )
    print(
        f"Median throughput:"
        f" {statistics.median(throughputs):.2f} samples/s"
    )
    print(
        f"Mean VRAM:        "
        f"{statistics.mean(reserved):.2f} GiB"
    )
