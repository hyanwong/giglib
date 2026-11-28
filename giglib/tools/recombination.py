"""
Rules for placing crossovers between two genomes, using their homology, shared by the
simulators in :mod:`giglib.tools.simulation`. These are *policy* (choices that depend on the
biology being modelled, see :ref:`sec_design_recombination`), built on the *mechanism* of
finding homology (:mod:`giglib.tools.homology`), so a simulator can use them, or choose its
own.

- :func:`allowed_crossovers`: where a crossover between two genomes is allowed: within long
  chains of homology (as a stand-in for allelic homology), in the same orientation in both
  genomes, and outside excluded regions (e.g. microsatellites, or centromeres).
- :func:`random_crossover`: choose one of the allowed places uniformly.
"""

import numpy as np

from giglib import intervals
from giglib.tools import homology


def allowed_crossovers(tables, u, v, *, min_chain_length, max_gap, u_excluded=(), v_excluded=()):
    """
    Where a crossover between a genome ``u`` and a genome ``v`` (on chromosome 0) is
    allowed, as stretches of ``u``, each with the offset to the corresponding positions of
    ``v``. A crossover can break ``u`` between any two adjacent positions of a stretch, and
    ``v`` between the corresponding two positions (adding the offset), so a break is always
    strictly inside both genomes. The rules are:

    - **Long chains of homology:** only in a chain of homology blocks between ``u`` and
      ``v`` (see :func:`~giglib.tools.homology.chains`) at least ``min_chain_length``
      positions long, whose blocks are separated by gaps of at most ``max_gap``. Long chains
      are a stand-in for allelic homology (see :ref:`sec_design_homology`).
    - **The same orientation:** not in a block that is reversed in ``v`` relative to ``u``
      (so crossovers are suppressed within heterozygous inversions).
    - **Outside excluded regions:** not in any of the intervals ``u_excluded`` of ``u``, or
      ``v_excluded`` of ``v`` (each a list of ``(left, right)`` tuples in that genome's
      coordinates), e.g. microsatellites or centromeres.

    :param giglib.Tables tables: The tables (which need the flags required by
        :meth:`~giglib.Tables.find_homology`)
    :param int u: The first genome (node ID)
    :param int v: The second genome (node ID)
    :param int min_chain_length: The shortest chain of homology in which a crossover can be
        placed
    :param int max_gap: The largest gap between consecutive blocks of a chain
    :param list u_excluded: Intervals of ``u`` in which crossovers are not allowed
    :param list v_excluded: Intervals of ``v`` in which crossovers are not allowed
    :return: A list of ``(left, right, offset)`` tuples, sorted by position in ``u``
    :rtype: list(tuple)
    """
    blocks = homology.homology_blocks(tables, u, v)
    allowed = []
    for chain in homology.chains(blocks, max_gap=max_gap):
        if homology.chain_length(chain) < min_chain_length:
            continue
        for block in chain:
            if block.is_reversed:
                continue
            offset = block.map.offset  # not reversed, so v's position is u's plus the offset
            # The excluded regions of v, in u's coordinates
            excluded = [*u_excluded, *[(left - offset, right - offset) for left, right in v_excluded]]
            for left, right in intervals.subtract([(block.u_left, block.u_right)], intervals.normalise(excluded)):
                allowed.append((left, right, offset))
    return allowed


def random_crossover(allowed, rng):
    """
    Choose a crossover uniformly from all the allowed breaks, as returned by
    :func:`allowed_crossovers` (a stretch of n positions has n - 1 breaks between them).

    :param list allowed: The allowed stretches, as ``(left, right, offset)`` tuples
    :param rng: A numpy random number generator, e.g. made by ``numpy.random.default_rng()``
    :return: The breaks in ``u`` and in ``v`` (each the position after the break), or None
        if no break is allowed
    :rtype: tuple(int, int)
    """
    num_breaks = np.array([right - left - 1 for left, right, _ in allowed], dtype=float)
    if num_breaks.sum() == 0:
        return None
    left, right, offset = allowed[rng.choice(len(allowed), p=num_breaks / num_breaks.sum())]
    u_break = left + 1 + int(rng.integers(right - left - 1))
    return u_break, u_break + offset
