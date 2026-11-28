"""
A forward-time simulator of a population of haploid genomes that grows from a single genome,
one base pair long, by clonal reproduction, with tandem duplications and deletions that
change the lengths of the genomes. Every position of every genome then descends from the
one base, so all positions are homologous to each other at some level. See
:ref:`sec_tools_simulation_single_origin`, and :issue:`221` for the plan.
"""

import numpy as np

import giglib as gigl
from giglib.tools import homology


def sim_single_origin(
    population_size,
    generations,
    *,
    duplication_rate,
    deletion_rate,
    mean_length=20,
    target_length=None,
    fitness_width=None,
    recombination_rate=0,
    min_chain_length=None,
    max_gap=None,
    min_chain_fraction=None,
    simplify_interval=None,
    random_seed=None,
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
    stretch is drawn from a geometric distribution with mean ``mean_length``, and redrawn
    if longer than the parent's genome (or one less, for a deletion). Capping it instead
    would make a deletion of most of a short genome common, and leave many genomes with a
    single position.

    If ``target_length`` is given, there is stabilising selection on genome length: a genome
    of length ``n`` has fitness ``exp(-(n - target_length)² / (2 fitness_width²))``, and
    parents are chosen in proportion to their fitness. Otherwise (the default) there is no
    selection, and every genome is equally likely to be a parent.

    With probability ``recombination_rate``, a child has two parents (chosen in the same
    way, and distinct), and its genome is the start of one and the end of the other, joined
    at a crossover. As in :class:`~giglib.tools.simulation.microsatellites_forward.DTWF_microsatellite_sim`,
    a crossover is only placed in a chain of homology between the parents at least
    ``min_chain_length`` positions long, whose blocks are separated by gaps of at most
    ``max_gap`` positions, between two positions of a block in the same orientation in both
    parents (chosen uniformly from all such places). If there is none, the child has the
    first parent only. With ``min_chain_fraction``, a chain is also long enough if it covers
    at least that fraction of the shorter parent's genome, so that short genomes (e.g. early
    on, without selection to make them grow) can recombine. This is for testing: as short
    chains are less likely to be allelic, it gives many unequal crossovers between short
    genomes, so it isn't meant for simulations with any biological realism. A recombinant
    child can also have a duplication or deletion, as above. A crossover in a chain between
    copies that aren't allelic (e.g. where a copy was lost in one parent) is unequal, and
    changes the number of copies.

    Finding homology between the parents at every crossover traces them back through the
    graph, which grows by ``population_size`` genomes every generation. So with
    ``simplify_interval``, the tables are simplified (see :meth:`giglib.Tables.simplify`)
    with respect to the current generation every ``simplify_interval`` generations, and
    with respect to the last generation at the end. As simplifying doesn't change the
    homology between the genomes that are kept, this gives the same genomes as not
    simplifying (and simplifying at the end), only faster.

    No mutations are added, so the genomes have no sequence, but their homology is recorded
    by the iedges.

    :param int population_size: The number of genomes in each generation, once the
        population has grown
    :param int generations: The number of generations. The single genome at the start is
        at this time, and the last generation, which are the samples, at time 0
    :param float duplication_rate: The probability that a child has a duplication
    :param float deletion_rate: The probability that a child has a deletion (if it has no
        duplication)
    :param float mean_length: The mean length of a duplication or deletion
    :param float target_length: The genome length favoured by selection, or None (default)
        for no selection
    :param float fitness_width: How fast fitness falls away from ``target_length``: the
        standard deviation of the Gaussian fitness function (needed if ``target_length`` is
        given)
    :param float recombination_rate: The probability that a child has two parents (default 0)
    :param int min_chain_length: The shortest chain of homology in which a crossover can be
        placed (needed if ``recombination_rate`` is not 0)
    :param int max_gap: The largest gap between consecutive blocks of a chain (needed if
        ``recombination_rate`` is not 0)
    :param float min_chain_fraction: If given, a chain is also long enough if it covers at
        least this fraction of the shorter parent's genome (for testing: see above)
    :param int simplify_interval: If given, simplify the tables every this many generations,
        and at the end (default None: don't simplify)
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
    if (target_length is None) != (fitness_width is None):
        raise ValueError("Selection needs both a target_length and a fitness_width")
    if fitness_width is not None and fitness_width <= 0:
        raise ValueError("The fitness_width must be positive")
    if not 0 <= recombination_rate <= 1:
        raise ValueError("The recombination_rate must be between 0 and 1")
    if recombination_rate > 0 and (min_chain_length is None or max_gap is None):
        raise ValueError("Recombination needs a min_chain_length and a max_gap")
    if min_chain_fraction is not None and not 0 < min_chain_fraction <= 1:
        raise ValueError("The min_chain_fraction must be greater than 0, and at most 1")
    if simplify_interval is not None and simplify_interval < 1:
        raise ValueError("The simplify_interval must be at least 1")
    rng = np.random.default_rng(random_seed)
    tables = gigl.Tables()
    population = [tables.nodes.add_row(generations)]
    lengths = [1]  # the length of each genome in the population
    for time in range(generations - 1, -1, -1):
        flags = gigl.NODE_IS_SAMPLE if time == 0 else 0
        children, child_lengths = [], []
        if target_length is not None:
            # The probability of each genome being a parent, in proportion to its fitness
            # (subtracting the largest log fitness first, so that exp() can't underflow to 0)
            log_fitness = -((np.array(lengths) - target_length) ** 2) / (2 * fitness_width**2)
            weights = np.exp(log_fitness - log_fitness.max())
            parent_probabilities = weights / weights.sum()
        else:
            parent_probabilities = None
        for _ in range(min(2 * len(population), population_size)):
            i = _choose_parent(rng, population, parent_probabilities)
            # The genome the child inherits, as segments [parent, parent_start, length]
            segments = [[population[i], 0, lengths[i]]]
            if recombination_rate > 0 and len(population) > 1 and rng.random() < recombination_rate:
                j = i
                while j == i:
                    j = _choose_parent(rng, population, parent_probabilities)
                shortest = min_chain_length
                if min_chain_fraction is not None:
                    shortest = min(shortest, min_chain_fraction * min(lengths[i], lengths[j]))
                breaks = _find_crossover(tables, rng, population[i], population[j], shortest, max_gap)
                if breaks is not None:
                    i_break, j_break = breaks
                    segments = [[population[i], 0, i_break], [population[j], j_break, lengths[j] - j_break]]
            child = tables.nodes.add_row(time, flags=flags)
            child_length = _add_child(tables, rng, child, segments, duplication_rate, deletion_rate, mean_length)
            children.append(child)
            child_lengths.append(child_length)
        population, lengths = children, child_lengths
        if recombination_rate > 0 or simplify_interval is not None:
            tables.validate()  # so that the next generation can trace through this one
        if simplify_interval is not None and (generations - time) % simplify_interval == 0:
            node_map = tables.simplify(population)
            population = [int(node_map[u]) for u in population]
    if simplify_interval is not None:
        tables.simplify(population)
    tables.sort()
    return tables.graph()


def _choose_parent(rng, population, parent_probabilities):
    """
    The index of a parent in the population: in proportion to the probabilities if given,
    otherwise uniformly
    """
    if parent_probabilities is None:
        return rng.integers(len(population))
    return rng.choice(len(population), p=parent_probabilities)


def _find_crossover(tables, rng, mum, dad, min_chain_length, max_gap):
    """
    Choose where to break ``mum`` and ``dad`` for a crossover, and return the two breaks
    (each between two positions), or None if there is nowhere allowed. A break is allowed
    between two positions of ``mum`` in the same block of a long enough chain of homology
    with ``dad`` (in the same orientation), so it is strictly inside both genomes. One is
    chosen uniformly from all the allowed breaks.
    """
    blocks = homology.homology_blocks(tables, mum, dad)
    allowed = [
        block
        for chain in homology.chains(blocks, max_gap=max_gap)
        if homology.chain_length(chain) >= min_chain_length
        for block in chain
        if not block.is_reversed
    ]
    # A block of n positions has n - 1 breaks between them
    num_breaks = np.array([block.u_right - block.u_left - 1 for block in allowed], dtype=float)
    if num_breaks.sum() == 0:
        return None
    block = allowed[rng.choice(len(allowed), p=num_breaks / num_breaks.sum())]
    mum_break = block.u_left + 1 + int(rng.integers(block.u_right - block.u_left - 1))
    return mum_break, mum_break + block.map.offset


def _stretch_length(rng, mean_length, max_length):
    """
    The length of a duplicated or deleted stretch: geometric with mean ``mean_length``,
    conditioned on being at most ``max_length`` (by redrawing). As a length of 1 has
    probability ``1 / mean_length``, this takes at most ``mean_length`` draws on average.
    """
    while True:
        length = rng.geometric(1 / mean_length)
        if length <= max_length:
            return int(length)


def _add_child(tables, rng, child, segments, duplication_rate, deletion_rate, mean_length):
    """
    Add the iedges from the parents to a child, and return the length of the child. The
    child inherits the genome made of ``segments`` (a list of ``[parent, parent_start,
    length]``), with a duplication or deletion. Each piece is a (child_left, child_right,
    inherited_left) tuple, in the coordinates of that inherited genome.
    """
    n = sum(length for _, _, length in segments)
    event = rng.random()
    if event < duplication_rate:
        # Repeat positions [s, s + L) of the parent: the copy follows the original
        L = _stretch_length(rng, mean_length, n)
        s = rng.integers(0, n - L + 1)
        pieces = [(0, s + L, 0), (s + L, s + 2 * L, s), (s + 2 * L, n + L, s + L)]
        new_length = n + L
    elif event < duplication_rate + deletion_rate and n > 1:
        # Remove positions [s, s + L) of the parent
        L = _stretch_length(rng, mean_length, n - 1)
        s = rng.integers(0, n - L + 1)
        pieces = [(0, s, 0), (s, n - L, s + L)]
        new_length = n - L
    else:
        pieces = [(0, n, 0)]
        new_length = n
    for child_left, child_right, inherited_left in pieces:
        # Cut the piece where it crosses from one segment (and parent) to the next
        segment_start = 0
        for parent, parent_start, length in segments:
            left = max(inherited_left, segment_start)
            right = min(inherited_left + child_right - child_left, segment_start + length)
            if left < right:
                tables.iedges.add_row(
                    child,
                    parent,
                    child_left=int(child_left + left - inherited_left),
                    child_right=int(child_left + right - inherited_left),
                    parent_left=int(parent_start + left - segment_start),
                    parent_right=int(parent_start + right - segment_start),
                )
            segment_start += length
    return int(new_length)
