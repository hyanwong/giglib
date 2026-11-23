"""
Compare two performance runs (e.g. before and after a change) case by case.

    python evaluation/compare.py evaluation/results/base.csv evaluation/results/new.csv

Prints the median value of each case in both runs, and the ratio new/base (below 1 is
an improvement). Differences within the run-to-run spread of the repeats are marked
"~" rather than as a change, so use --repeats >= 3 in performance.py for this to be
meaningful. See evaluation/README.md for the workflow.
"""

import argparse
import csv
import statistics

KEYS = ("case", "variant", "size", "unit")


def summarise(path):
    """Return {case key: (median, spread)} and the git SHAs and CPUs of a run"""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    values = {}
    for row in rows:
        values.setdefault(tuple(row[k] for k in KEYS), []).append(float(row["value"]))
    summary = {key: (statistics.median(v), max(v) - min(v)) for key, v in values.items()}
    shas = ",".join(sorted({row["git_sha"] for row in rows}))
    cpus = "; ".join(sorted({row["cpu"] for row in rows}))
    return summary, shas, cpus


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("base")
    p.add_argument("new")
    args = p.parse_args()
    base, base_sha, base_cpu = summarise(args.base)
    new, new_sha, new_cpu = summarise(args.new)
    print(f"base: {base_sha} ({base_cpu})")
    print(f"new:  {new_sha} ({new_cpu})")
    if base_cpu != new_cpu:
        print("Note: the runs are from different CPUs, so the ratios include hardware effects")
    print(f"\n{'case':12} {'variant':16} {'size':>6} {'base':>10} {'new':>10} {'new/base':>9}  unit")
    for key in sorted(base.keys() & new.keys()):
        case, variant, size, unit = key
        (b, b_spread), (n, n_spread) = base[key], new[key]
        ratio = n / b
        flag = "~" if abs(n - b) <= max(b_spread, n_spread) else ("faster" if ratio < 1 else "SLOWER")
        print(f"{case:12} {variant:16} {size:>6} {b:10.3g} {n:10.3g} {ratio:9.2f}  {unit}  {flag}")


if __name__ == "__main__":
    main()
