"""
A forward-time simulator of a population of haploid genomes that grows from a single genome,
one base pair long, by clonal reproduction, with tandem duplications and deletions that
change the lengths of the genomes. Every position of every genome then descends from the
one base, so all positions are homologous to each other at some level. See
:ref:`sec_tools_simulation_single_origin`, and :issue:`221` for the plan.
"""

import numpy as np

import giglib as gigl


def sim_single_origin(
    population_size, generations, *, duplication_rate, deletion_rate, mean_length=20, random_seed=None
):
    """
    Simulate a population of haploid genomes, starting from a single genome of length 1.
    The population doubles in size each generation until it reaches ``population_size``,
    and then stays at that size. Each genome in a generation is the child of a genome
    chosen at random from the previous one (a Wright-Fisher model), and copies it, except
    that it has a tandem duplication with probability ``duplication_rate``, or else a
    deletion with probability ``deletion_rate``. A duplication copies a stretch of the
    parent's genome, and puts the copy straight after it. A deletion removes a stretch, but
    never the whole genome (so a genome of length 1 can't have one). The length of the
    stretch is drawn from a geometric distribution with mean ``mean_length``, but is no
    longer than the parent's genome (or one less, for a deletion).

    There is no recombination, and no selection. No mutations are added, so the genomes
    have no sequence, but their homology is recorded by the iedges.

    :param int population_size: The number of genomes in each generation, once the
        population has grown
    :param int generations: The number of generations. The single genome at the start is
        at this time, and the last generation, which are the samples, at time 0
    :param float duplication_rate: The probability that a child has a duplication
    :param float deletion_rate: The probability that a child has a deletion (if it has no
        duplication)
    :param float mean_length: The mean length of a duplication or deletion
    :param int random_seed: The random seed
    :return: A GIG
    :rtype: giglib.Graph
    """
    if population_size < 1 or generations < 1:
        raise ValueError("The population size and number of generations must be at least 1")
    if not (0 <= duplication_rate <= 1 and 0 <= deletion_rate <= 1):
        raise ValueError("The rates of duplication and deletion must be between 0 and 1")
    if duplication_rate + deletion_rate > 1:
        raise ValueError("The rates of duplication and deletion can't add up to more than 1")
    rng = np.random.default_rng(random_seed)
    tables = gigl.Tables()
    population = [tables.nodes.add_row(generations)]
    lengths = [1]  # the length of each genome in the population
    for time in range(generations - 1, -1, -1):
        flags = gigl.NODE_IS_SAMPLE if time == 0 else 0
        children, child_lengths = [], []
        for _ in range(min(2 * len(population), population_size)):
            i = rng.integers(len(population))
            parent, parent_length = population[i], lengths[i]
            child = tables.nodes.add_row(time, flags=flags)
            child_length = _add_child(
                tables, rng, child, parent, parent_length, duplication_rate, deletion_rate, mean_length
            )
            children.append(child)
            child_lengths.append(child_length)
        population, lengths = children, child_lengths
    tables.sort()
    return tables.graph()


def _add_child(tables, rng, child, parent, n, duplication_rate, deletion_rate, mean_length):
    """
    Add the iedges from a parent of length ``n`` to its child, and return the length of
    the child. Each piece is a (child_left, child_right, parent_left) tuple.
    """
    event = rng.random()
    if event < duplication_rate:
        # Repeat positions [s, s + L) of the parent: the copy follows the original
        L = min(rng.geometric(1 / mean_length), n)
        s = rng.integers(0, n - L + 1)
        pieces = [(0, s + L, 0), (s + L, s + 2 * L, s), (s + 2 * L, n + L, s + L)]
        new_length = n + L
    elif event < duplication_rate + deletion_rate and n > 1:
        # Remove positions [s, s + L) of the parent
        L = min(rng.geometric(1 / mean_length), n - 1)
        s = rng.integers(0, n - L + 1)
        pieces = [(0, s, 0), (s, n - L, s + L)]
        new_length = n - L
    else:
        pieces = [(0, n, 0)]
        new_length = n
    for child_left, child_right, parent_left in pieces:
        if child_right > child_left:
            tables.iedges.add_row(
                child,
                parent,
                child_left=int(child_left),
                child_right=int(child_right),
                parent_left=int(parent_left),
                parent_right=int(parent_left + child_right - child_left),
            )
    return int(new_length)
