"""
A backward-time simulator of microsatellites, which lays them on a genealogy simulated with
msprime.
"""

import numpy as np


def sim_microsatellites_backward(
    num_diploids,
    sequence_length,
    loci,
    *,
    max_units,
    population_size,
    recombination_rate,
    mutation_rate,
    model=None,
    random_seed=None,
):
    """
    Simulate microsatellites backward in time, and return the result as a GIG (see
    :issue:`202`). As changes in the length of microsatellites don't affect where
    crossovers can happen (see :ref:`sec_design_homology`), the genealogy is independent of
    them (:issue:`93`): so it is simulated with msprime, and the microsatellites are then
    laid on top, using msprime's stepwise mutation model (``msprime.SMM``), in which
    the alleles are numbers of repeats, between 1 and ``max_units``. Each locus is then
    given room for ``max_units`` repeats, and its alleles are written out as repeats and
    gaps (e.g. ``ATATAT----`` for 3 repeats of ``AT``, with room for 5), so that
    :func:`giglib.tools.indels.from_tree_sequence` converts them into changes in the
    coordinates of the genomes. Positions outside the microsatellites have unknown states.

    Unlike in
    :class:`~giglib.tools.simulation.microsatellites_forward.DTWF_microsatellite_sim`,
    gained repeats are inserted material of unknown origin, rather than copies of a
    neighbouring repeat.

    This needs msprime, which is not a requirement of giglib itself.

    :param int num_diploids: The number of diploid samples
    :param int sequence_length: The length of the sequence simulated by msprime, in which
        each microsatellite occupies a single position
    :param list loci: The microsatellites, as a list of ``(position, motif)`` tuples, e.g.
        ``(100, "AT")``, with positions in the sequence simulated by msprime
    :param int max_units: The largest number of repeats at any locus
    :param float population_size: The population size, passed to ``msprime.sim_ancestry``
    :param float recombination_rate: The recombination rate, passed to
        ``msprime.sim_ancestry``
    :param float mutation_rate: The rate at which a locus gains or loses a repeat
    :param model: The ancestry model, passed to ``msprime.sim_ancestry`` (default:
        the coalescent)
    :param int random_seed: The random seed
    :return: A GIG
    :rtype: giglib.Graph
    """
    import msprime

    from giglib.tools import indels

    loci = sorted(loci)
    positions = [position for position, _ in loci]
    if len(set(positions)) != len(positions) or any(not 0 <= p < sequence_length for p in positions):
        raise ValueError("Microsatellite positions must be distinct, and within the sequence")
    rng = np.random.default_rng(random_seed)
    ancestry_seed, mutation_seed = rng.integers(1, 2**31, size=2)
    ts = msprime.sim_ancestry(
        num_diploids,
        sequence_length=sequence_length,
        population_size=population_size,
        recombination_rate=recombination_rate,
        model=model,
        random_seed=ancestry_seed,
    )
    # Mutations only at the microsatellites
    breaks = sorted({0, sequence_length, *positions, *(p + 1 for p in positions)})
    rates = [mutation_rate if left in positions else 0 for left in breaks[:-1]]
    model = msprime.SMM(lo=1, hi=max_units)
    ts = msprime.sim_mutations(
        ts, rate=msprime.RateMap(position=breaks, rate=rates), model=model, random_seed=mutation_seed
    )

    # Give each locus room for max_units repeats, by shifting the positions after it
    motifs = dict(loci)
    span = {position: len(motif) * max_units for position, motif in loci}

    def new_position(x):
        return x + sum(span[p] - 1 for p in positions if p < x)

    def allele(position, num_units):
        repeats = motifs[position] * int(num_units)
        return repeats + "-" * (span[position] - len(repeats))

    tables = ts.dump_tables()
    tables.sequence_length = new_position(sequence_length)
    edges = tables.edges
    edges.set_columns(
        left=np.array([new_position(x) for x in edges.left]),
        right=np.array([new_position(x) for x in edges.right]),
        parent=edges.parent,
        child=edges.child,
    )
    tables.sites.clear()
    tables.mutations.clear()
    for site in ts.sites():
        position = int(site.position)
        tables.sites.add_row(position=new_position(position), ancestral_state=allele(position, site.ancestral_state))
        for m in site.mutations:
            tables.mutations.add_row(
                site=site.id, node=m.node, derived_state=allele(position, m.derived_state), parent=m.parent, time=m.time
            )
    # msprime only makes sites where there are mutations: a locus with none still needs
    # its (ancestral) number of repeats, from the model's root distribution
    mutated = {int(site.position) for site in ts.sites()}
    for position in positions:
        if position not in mutated:
            num_units = rng.choice(model.alleles, p=model.root_distribution)
            tables.sites.add_row(position=new_position(position), ancestral_state=allele(position, num_units))
    tables.sort()
    tables.build_index()
    return indels.from_tree_sequence(tables.tree_sequence())
