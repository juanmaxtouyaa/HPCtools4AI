import subprocess
import sys
import time

import psutil


command = sys.argv[1:]

process = subprocess.Popen(command)
root = psutil.Process(process.pid)

peak_rss = 0

while process.poll() is None:
    try:
        processes = [root] + root.children(recursive=True)

        rss = 0
        for p in processes:
            try:
                rss += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        peak_rss = max(peak_rss, rss)

    except psutil.NoSuchProcess:
        pass

    time.sleep(0.1)

return_code = process.wait()

print(
    f"\nPeak CPU RAM: "
    f"{peak_rss / 1024**3:.2f} GiB"
)

sys.exit(return_code)
