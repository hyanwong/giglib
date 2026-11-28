"""
A minimal forward-time simulation with recombination and no structural variants, written
three ways, to compare giglib's costs with tskit's. Each child of a haploid Wright-Fisher
population has two parents (chosen at random), and with probability ``recombination_rate``
inherits the start of one and the end of the other, joined at a crossover; otherwise it
copies the first. The founders all descend from a single ancestor. The tables are
simplified every ``simplify_interval`` generations, and at the end. The three variants are:

- ``"tskit"``: crossovers at random coordinates, in tskit tables, simplified by tskit;
- ``"giglib_coordinates"``: the same, in giglib tables (possible as, with no structural
  variants, every genome shares the same coordinates), simplified by giglib;
- ``"giglib_homology"``: as ``"giglib_coordinates"``, but each crossover is checked against
  the homology between the parents (:func:`giglib.tools.recombination.allowed_crossovers`),
  as the simulators with structural variants must do.

All three make exactly the same random choices, so give the same genealogy: :func:`simulate`
returns the result as a tree sequence, to check this. So differences in time are due to the
tables and simplification (``"giglib_coordinates"`` against ``"tskit"``), and to finding
homology (``"giglib_homology"`` against ``"giglib_coordinates"``).
"""

import time

import numpy as np
import tskit

import giglib as gigl
from giglib.tools import recombination

VARIANTS = ("tskit", "giglib_coordinates", "giglib_homology")


def simulate(
    variant,
    population_size,
    sequence_length,
    generations,
    *,
    recombination_rate,
    simplify_interval,
    random_seed,
):
    """
    Run the simulation, and return the resulting tree sequence (simplified with respect to
    the last generation, which are nodes 0 to ``population_size - 1``), and the time spent
    in each part, in seconds: adding rows (and validating, for giglib), choosing crossovers,
    and simplifying.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant}")
    rng = np.random.default_rng(random_seed)
    seconds = {"tables": 0.0, "crossovers": 0.0, "simplify": 0.0}
    L = sequence_length
    if variant == "tskit":
        tables = tskit.TableCollection(sequence_length=L)

        def add_node(t):
            return tables.nodes.add_row(time=t)

        def add_edge(child, parent, left, right):
            tables.edges.add_row(left, right, parent, child)

    else:
        tables = gigl.Tables()

        def add_node(t):
            return tables.nodes.add_row(t)

        def add_edge(child, parent, left, right):
            tables.iedges.add_row(
                child, parent, child_left=left, child_right=right, parent_left=left, parent_right=right
            )

    start = time.perf_counter()
    # The founders all descend from one ancestor, so every pair of genomes shares ancestry
    # everywhere (as with coordinates), and has homology everywhere
    ancestor = add_node(generations + 1)
    population = []
    for _ in range(population_size):
        founder = add_node(generations)
        add_edge(founder, ancestor, 0, L)
        population.append(founder)
    if variant == "giglib_homology":
        tables.validate()
    seconds["tables"] += time.perf_counter() - start
    for t in range(generations - 1, -1, -1):
        children = []
        for _ in range(population_size):
            mum, dad = (population[i] for i in rng.integers(population_size, size=2))
            crossover = rng.random() < recombination_rate and mum != dad
            if crossover:
                x = int(rng.integers(1, L))
                if variant == "giglib_homology":
                    start = time.perf_counter()
                    allowed = recombination.allowed_crossovers(tables, mum, dad, min_chain_length=1, max_gap=0)
                    # Without structural variants, the whole genome is one allowed stretch
                    if not any(left < x < right and offset == 0 for left, right, offset in allowed):
                        raise AssertionError(f"A crossover at {x} is not allowed")
                    seconds["crossovers"] += time.perf_counter() - start
            start = time.perf_counter()
            child = add_node(t)
            if crossover:
                add_edge(child, mum, 0, x)
                add_edge(child, dad, x, L)
            else:
                add_edge(child, mum, 0, L)
            children.append(child)
            seconds["tables"] += time.perf_counter() - start
        population = children
        if variant == "giglib_homology":
            start = time.perf_counter()
            tables.validate()  # so that the next generation can find homology through this one
            seconds["tables"] += time.perf_counter() - start
        if (generations - t) % simplify_interval == 0 or t == 0:
            start = time.perf_counter()
            if variant == "tskit":
                tables.sort()
                node_map = tables.simplify(population, keep_input_roots=True)
            else:
                tables.validate()
                node_map = tables.simplify(population)
            population = [int(node_map[u]) for u in population]
            seconds["simplify"] += time.perf_counter() - start
    if variant == "tskit":
        ts = tables.tree_sequence()
    else:
        ts = tables.graph().to_tree_sequence(sequence_length=L)
    return ts, seconds


def same_genealogy(ts1, ts2):
    """
    Whether two tree sequences give the same genealogy of their samples (nodes 0 to n - 1,
    in the same order): the same branch-length divergence between every pair of samples,
    whatever the order of their other nodes
    """
    return np.allclose(ts1.divergence_matrix(mode="branch"), ts2.divergence_matrix(mode="branch"))
