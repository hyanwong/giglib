"""
Time some core giglib operations. Each case is timed several times (``--repeats``),
writing one row per case, size and repeat to results/performance.csv (or the file
given by ``--output``). Use compare.py to compare two runs.

Example:
  python evaluation/performance.py
  python evaluation/performance.py --cases haplotypes --repeats 5 --output new.csv
"""

import argparse
import time

import common  # (also puts this checkout of giglib on the path)
import msprime
import wright_fisher

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


def time_find_homology(tables, pairs, modes):
    """
    The mean time, in milliseconds per pair of genomes, of find_homology in each mode,
    and the mean number of records it returns
    """
    for mode in modes:
        seconds = common.best_time(lambda m=mode: [tables.find_homology(u, v, mode=m) for u, v in pairs])
        records = sum(len(tables.find_homology(u, v, mode=mode)) for u, v in pairs) / len(pairs)
        yield mode, seconds / len(pairs) * 1e3, records


def sample_pairs(gig, num_pairs=5):
    """Pairs of genomes in the last generation, from different individuals where possible"""
    samples = [int(u) for u in gig.sample_nodes(time=0).ids]
    return [(samples[2 * i], samples[2 * i + 3]) for i in range(num_pairs)]


def case_homology_microsatellites(size):
    """
    Milliseconds per pair of genomes for find_homology, in a forward simulation of 10
    diploids whose genomes (10 kb) differ only at 10 sparse microsatellites, after ``size``
    generations: the typical case, with few copies, but a deep graph
    """
    from giglib.tools.simulation.microsatellites_forward import DTWF_microsatellite_sim

    loci = [(x, 2, 10) for x in range(500, 10000, 1000)]
    sim = DTWF_microsatellite_sim(loci, mutation_rate=0.01, min_chain_length=1000, max_gap=50)
    gig = sim.run(num_diploids=10, seq_lens=[10000], gens=size, random_seed=1)
    for mode, ms, records in time_find_homology(
        gig.tables, sample_pairs(gig), ["first_mrca", "closest_homologues", "all"]
    ):
        yield {"variant": mode, "size": size, "value": ms, "records": records}


def case_microsatellites_population_size(size):
    """
    Milliseconds per generation to simulate ``size`` diploids for 20 generations, with the
    forward simulator of microsatellites (as in homology_microsatellites, which finds the
    homology between the parents at every meiosis), and milliseconds per pair of genomes
    for find_homology at the end
    """
    import time

    from giglib.tools.simulation.microsatellites_forward import DTWF_microsatellite_sim

    loci = [(x, 2, 10) for x in range(500, 10000, 1000)]
    generations = 20
    sim = DTWF_microsatellite_sim(loci, mutation_rate=0.01, min_chain_length=1000, max_gap=50)
    start = time.perf_counter()
    gig = sim.run(num_diploids=size, seq_lens=[10000], gens=generations, random_seed=1)
    yield {
        "variant": "simulation_per_generation",
        "size": size,
        "value": (time.perf_counter() - start) / generations * 1e3,
    }
    for mode, ms, records in time_find_homology(
        gig.tables, sample_pairs(gig), ["first_mrca", "closest_homologues", "all"]
    ):
        yield {"variant": f"find_homology_{mode}_per_pair", "size": size, "value": ms, "records": records}


def case_homology_single_origin(size):
    """
    Milliseconds per pair of genomes for find_homology, in a simulation of 20 genomes grown
    from a single base pair over ``size`` generations: the extreme case, in which everything
    is homologous, through many copies
    """
    from giglib.tools.simulation.single_origin import sim_single_origin

    gig = sim_single_origin(20, size, duplication_rate=0.05, deletion_rate=0.02, mean_length=10, random_seed=1)
    for mode, ms, records in time_find_homology(
        gig.tables, sample_pairs(gig), ["first_mrca", "closest_homologues", "all"]
    ):
        yield {"variant": mode, "size": size, "value": ms, "records": records}


def simplified_rows(gig, size):
    """
    Rows for simplifying a GIG with respect to its genomes in the last generation:
    milliseconds to simplify (with the number of nodes before and after), and then
    milliseconds per pair of genomes for find_homology in the simplified GIG, for the same
    pairs as in the unsimplified cases (so the two can be compared)
    """
    samples = [int(u) for u in gig.sample_nodes(time=0).ids]
    seconds = common.best_time(lambda: gig.simplify(samples))
    tables = gig.tables.copy()
    node_map = tables.simplify(samples)
    yield {
        "variant": "simplify",
        "size": size,
        "value": seconds * 1e3,
        "nodes_before": len(gig.nodes),
        "nodes_after": len(tables.nodes),
    }
    pairs = [(int(node_map[u]), int(node_map[v])) for u, v in sample_pairs(gig)]
    for mode, ms, records in time_find_homology(tables, pairs, ["first_mrca", "closest_homologues", "all"]):
        yield {"variant": f"find_homology_{mode}_per_pair", "size": size, "value": ms, "records": records}


def case_simplify_microsatellites(size):
    """
    As homology_microsatellites, but simplifying the GIG before finding homology
    """
    from giglib.tools.simulation.microsatellites_forward import DTWF_microsatellite_sim

    loci = [(x, 2, 10) for x in range(500, 10000, 1000)]
    sim = DTWF_microsatellite_sim(loci, mutation_rate=0.01, min_chain_length=1000, max_gap=50)
    gig = sim.run(num_diploids=10, seq_lens=[10000], gens=size, random_seed=1)
    yield from simplified_rows(gig, size)


def case_simplify_single_origin(size):
    """
    As homology_single_origin, but simplifying the GIG before finding homology
    """
    from giglib.tools.simulation.single_origin import sim_single_origin

    gig = sim_single_origin(20, size, duplication_rate=0.05, deletion_rate=0.02, mean_length=10, random_seed=1)
    yield from simplified_rows(gig, size)


def case_tskit_comparison(size):
    """
    Milliseconds per generation for a forward simulation of ``size`` haploid genomes of
    10 kb over 100 generations, with a crossover in half the children, and no structural
    variants, simplified every 10 generations: in tskit, in giglib with crossovers at
    random coordinates, and in giglib checking each crossover against the parents'
    homology (see wright_fisher.py). Each row also has the milliseconds per generation spent
    on the tables, on crossovers and on simplifying. All three must give the same genealogy.
    (Each simulation is timed once per repeat, as it is slow.)
    """
    generations = 100
    rows, results = [], {}
    for variant in wright_fisher.VARIANTS:
        start = time.perf_counter()
        ts, seconds = wright_fisher.simulate(
            variant, size, 10_000, generations, recombination_rate=0.5, simplify_interval=10, random_seed=1
        )
        total = time.perf_counter() - start
        results[variant] = ts
        rows.append(
            {
                "variant": variant,
                "size": size,
                "value": total / generations * 1e3,
                **{f"{part}_ms": value / generations * 1e3 for part, value in seconds.items()},
            }
        )
    for variant in wright_fisher.VARIANTS[1:]:
        if not wright_fisher.same_genealogy(results["tskit"], results[variant]):
            raise AssertionError(f"{variant} gives a different genealogy from tskit")
    yield from rows


CASES = {
    # name: (function, unit, sizes, quick sizes)
    "add_iedges": (case_add_iedges, "us_per_iedge", [2000], [100]),
    "haplotypes": (case_haplotypes, "ms_per_haplotype", [50], [5]),
    "homology_microsatellites": (case_homology_microsatellites, "ms_per_pair", [20, 40, 80], [10]),
    "microsatellites_population_size": (case_microsatellites_population_size, "ms", [10, 50, 100], [10]),
    "homology_single_origin": (case_homology_single_origin, "ms_per_pair", [100, 200, 400], [50]),
    "simplify_microsatellites": (case_simplify_microsatellites, "ms", [20, 40, 80], [10]),
    "simplify_single_origin": (case_simplify_single_origin, "ms", [100, 200, 400], [50]),
    "tskit_comparison": (case_tskit_comparison, "ms_per_generation", [20, 50, 100], [10]),
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
