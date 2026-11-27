"""Shared helpers for the evaluation scripts."""

import csv
import datetime
import os
import pathlib
import platform
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"
# Run giglib from this checkout, rather than any installed version
sys.path.insert(0, str(HERE.parent))


def cpu_name():
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def provenance():
    """Columns identifying the code version and machine, to allow comparing runs."""
    import giglib

    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no", str(HERE.parent)], cwd=HERE, text=True
        ).strip()
        sha += "-dirty" if dirty else ""
    except Exception:
        sha = ""
    return {
        "giglib_version": giglib.__version__,
        "git_sha": sha,
        "cpu": cpu_name(),
        "cpu_cores": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def best_time(func, *, number=1, repeats=3):
    """
    The fastest of ``repeats`` timings of ``number`` calls to ``func``, per call
    (seconds), after an untimed call to warm up (e.g. numpy's first calls are slow)
    """
    func()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(number):
            func()
        times.append((time.perf_counter() - start) / number)
    return min(times)


def write_csv(rows, name):
    """
    Write a list of dicts to results/<name>, with a column for each key in any of them
    (left empty for rows without it)
    """
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
