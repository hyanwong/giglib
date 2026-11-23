"""
Time some core giglib operations. Each case is timed several times (``--repeats``),
writing one row per case, size and repeat to results/performance.csv (or the file
given by ``--output``). Use compare.py to compare two runs.

Example:
  python evaluation/performance.py
  python evaluation/performance.py --cases haplotypes --repeats 5 --output new.csv
"""

import argparse

import common  # (also puts this checkout of giglib on the path)
import msprime

import giglib as gigl


def build_iedges(num_children, iedges_per_child, validate):
    """
    Add iedges for ``num_children`` children of one root, validating never ("none"),
    once at the end ("end"), after each child's iedges ("child"), or after every row
    ("row")
    """
    tables = gigl.Tables()
    tables.nodes.add_row(num_children + 1)
    for c in range(num_children):
        tables.nodes.add_row(num_children - c)  # in time order, as for a sorted table
    for child in range(1, num_children + 1):
        for k in range(iedges_per_child):
            tables.iedges.add_row(child, 0, child_left=10 * k, child_right=10 * (k + 1))
            if validate == "row":
                tables.validate()
        if validate == "child":
            tables.validate()
    if validate == "end":
        tables.validate()
    return tables


def case_add_iedges(size):
    """Microseconds per iedge to add iedges, with validation at different intervals"""
    num_children, per_child = size, 5
    for validate in ("none", "end", "child", "row"):
        seconds = common.best_time(lambda v=validate: build_iedges(num_children, per_child, v))
        yield {"variant": f"validate_{validate}", "size": size, "value": seconds / (num_children * per_child) * 1e6}


def case_haplotypes(size):
    """Milliseconds to decode the haplotype of each sample of a GIG from msprime"""
    ts = msprime.sim_ancestry(size, sequence_length=1e5, recombination_rate=1e-8, population_size=1e4, random_seed=1)
    ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=1)
    gig = gigl.Graph.from_tree_sequence(ts)
    samples = gig.sample_nodes()
    seconds = common.best_time(lambda: [gig.haplotype(u) for u in samples])
    yield {"variant": "haplotype", "size": size, "value": seconds / len(samples) * 1e3}


CASES = {
    # name: (function, unit, sizes, quick sizes)
    "add_iedges": (case_add_iedges, "us_per_iedge", [2000], [100]),
    "haplotypes": (case_haplotypes, "ms_per_haplotype", [50], [5]),
}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", help="Output CSV name in results/ (default performance.csv)")
    p.add_argument("--quick", action="store_true", help="Tiny smoke-test sizes, written to performance_quick.csv")
    args = p.parse_args()
    if args.repeats < 1:
        p.error("--repeats must be at least 1")
    if args.quick:
        args.repeats = 1
    output = args.output or ("performance_quick.csv" if args.quick else "performance.csv")

    prov = common.provenance()
    rows = []
    for name in args.cases:
        func, unit, sizes, quick_sizes = CASES[name]
        for size in quick_sizes if args.quick else sizes:
            for repeat in range(args.repeats):
                for row in func(size):
                    rows.append({"case": name, **row, "unit": unit, "repeat": repeat, **prov})
                    print(f"{name} {row['variant']} (size {size}): {row['value']:.3g} {unit}")
    print("Written to", common.write_csv(rows, output))


if __name__ == "__main__":
    main()
