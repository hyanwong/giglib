"""
Find homology between two genomes in a GIG, and join it into chains, as described in
:ref:`sec_design_homology`. Homology is defined by shared ancestry in the graph, without
any reference: two positions are homologous if they trace back to the same position in a
common ancestor.

It builds on :meth:`giglib.Tables.find_mrca_regions`, which finds the regions of two
genomes that share a common ancestor, and stops tracing each region at its most recent one
(as needed to place a crossover). This module adds:

- :func:`homology_blocks`: the homology from :meth:`~giglib.Tables.find_mrca_regions`, as
  blocks, each a maximal interval with a single map from one genome to the other
  (merging adjacent regions with different common ancestors but the same map);
- :func:`chains`: joining blocks into chains that are colinear, separated by small gaps
  (e.g. at microsatellites whose lengths differ), as a heuristic for allelic homology;
- :func:`find_homology`: *all* the homology between two genomes, with the most recent
  common ancestor of each pair of positions. Unlike
  :meth:`~giglib.Tables.find_mrca_regions`, this doesn't stop at the first common
  ancestor, so it also finds older homology, e.g. between copies of a region that was
  duplicated in a common ancestor of the two genomes.

This module is experimental: see :ref:`sec_tools`. It provides *mechanism* (finding
homology, and chaining it), not *policy* (e.g. where crossovers are allowed), which is up
to the simulators that use it. Discussion: :issue:`202`.
"""

import collections

from giglib import PositionMap


class Block(collections.namedtuple("Block", "u_left, u_right, u_chromosome, v_left, v_right, v_chromosome")):
    """
    A homology block: a maximal interval ``[u_left, u_right)`` of a chromosome of genome
    ``u`` that is homologous to the interval of a chromosome of genome ``v`` with ends
    ``(v_left, v_right)``, through a single :class:`~giglib.PositionMap`. If the block is
    reversed (e.g. through an inversion), ``v_left > v_right``, as for the parent interval
    of an inverted iedge.
    """

    __slots__ = ()

    @property
    def map(self):
        """
        The :class:`~giglib.PositionMap` from positions in ``u`` to homologous positions
        in ``v``
        """
        return PositionMap.from_intervals((self.u_left, self.u_right), (self.v_left, self.v_right))

    @property
    def length(self):
        return self.u_right - self.u_left

    @property
    def is_reversed(self):
        return self.v_left > self.v_right


def homology_blocks(tables, u, v, *, u_chromosome=0, v_chromosome=0):
    """
    Return the homology blocks between a chromosome of genome ``u`` and a chromosome of
    genome ``v``, sorted by their position in ``u``. Homology is found with
    :meth:`~giglib.Tables.find_mrca_regions`, and adjacent regions of ``u`` whose
    homology in ``v`` has the same map (e.g. through different common ancestors) are
    merged into a single block. A region of ``u`` can be in several blocks, if it is
    homologous to several regions of ``v`` (e.g. copies of a duplicated region).

    :param giglib.Tables tables: The tables (which need the flags required by
        :meth:`~giglib.Tables.find_mrca_regions`)
    :param int u: The first genome (node ID)
    :param int v: The second genome (node ID)
    :param int u_chromosome: The chromosome of ``u`` (default 0)
    :param int v_chromosome: The chromosome of ``v`` (default 0)
    :return: A list of :class:`Block` objects
    :rtype: list(Block)
    """
    mrcas = tables.find_mrca_regions(u, v, u_chromosomes=[u_chromosome], v_chromosomes=[v_chromosome])
    pieces = set()
    for mrca_intervals in mrcas.values():
        for u_segments, v_segments in mrca_intervals.values():
            for u_left, u_right, u_chr in u_segments:
                for v_left, v_right, v_chr in v_segments:
                    if u_chr != u_chromosome or v_chr != v_chromosome:
                        continue
                    # The ends of the MRCA interval map to (u_left, u_right) in u, and to
                    # (v_left, v_right) in v: so u_left corresponds to v_left
                    if u_left > u_right:  # put u in increasing order
                        u_left, u_right, v_left, v_right = u_right, u_left, v_right, v_left
                    pieces.add((u_left, u_right, v_left, v_right))
    blocks = []
    for u_left, u_right, v_left, v_right in sorted(pieces):
        block = Block(u_left, u_right, u_chromosome, v_left, v_right, v_chromosome)
        # Merge with a previous block that ends where this one starts, with the same map
        for i, previous in enumerate(blocks):
            if previous.u_right == u_left and previous.map == block.map:
                blocks[i] = previous._replace(u_right=u_right, v_right=v_right)
                break
        else:
            blocks.append(block)
    blocks.sort()
    return blocks


def _trim_start(block, trim):
    """
    Return the block without its first ``trim`` positions (in both genomes)
    """
    sign = -1 if block.is_reversed else 1
    return block._replace(u_left=block.u_left + trim, v_left=block.v_left + sign * trim)


def _start(block, length):
    """
    Return the first ``length`` positions of the block (in both genomes)
    """
    sign = -1 if block.is_reversed else 1
    return block._replace(u_right=block.u_left + length, v_right=block.v_left + sign * length)


def _trim_to_follow(before, after, max_gap):
    """
    If block ``after`` can follow block ``before`` in a chain, return the number of
    positions to trim from the start of ``after`` (where it overlaps ``before`` in either
    genome), otherwise None. ``after`` can follow ``before`` if they are on the same
    chromosomes, in the same orientation, and (once trimmed) ``after`` starts after
    ``before`` ends, by at most ``max_gap`` positions, in both genomes (in the direction of
    the blocks, in ``v``), with something left of ``after``.
    """
    if (before.u_chromosome, before.v_chromosome, before.is_reversed) != (
        after.u_chromosome,
        after.v_chromosome,
        after.is_reversed,
    ):
        return None
    u_gap = after.u_left - before.u_right
    v_gap = before.v_right - after.v_left if before.is_reversed else after.v_left - before.v_right
    trim = max(0, -u_gap, -v_gap)
    if trim >= after.length or u_gap + trim > max_gap or v_gap + trim > max_gap:
        return None
    return trim


def chains(blocks, *, max_gap):
    """
    Join homology blocks into chains: series of blocks that are in the same order in both
    genomes, in the same orientation, on the same chromosomes, and separated by gaps of at
    most ``max_gap`` positions in each genome (see :ref:`sec_design_homology`). Where a
    block overlaps the previous one in a chain (in either genome), its start is trimmed,
    as in the chaining of genome alignments: e.g. near a copy-number variant, where the
    homology of one copy continues into the region after it.

    Chains are found greedily: first the chain with the most homology (the total length
    of its blocks), then the chain with the most homology among the remaining homology,
    and so on, so that every position of every block is in exactly one chain (the parts
    trimmed from a block remain, to be used by other chains).

    :param list(Block) blocks: The blocks, e.g. from :func:`homology_blocks`
    :param int max_gap: The maximum gap between consecutive blocks in a chain
    :return: A list of chains, each a list of blocks in order, starting with the chain
        with the most homology
    :rtype: list(list(Block))
    """
    remaining = sorted(blocks)
    result = []
    while remaining:
        # For each block, the most homology in a chain ending with it, the block before it
        # in that chain, and how much of it was trimmed to follow that block
        best = [b.length for b in remaining]
        previous = [None] * len(remaining)
        trims = [0] * len(remaining)
        for i, after in enumerate(remaining):
            for j in range(i):
                trim = _trim_to_follow(remaining[j], after, max_gap)
                if trim is not None and best[j] + after.length - trim > best[i]:
                    best[i], previous[i], trims[i] = best[j] + after.length - trim, j, trim
        i = max(range(len(remaining)), key=lambda k: best[k])
        chain = []
        while i is not None:
            chain.append(i)
            i = previous[i]
        chain.reverse()
        result.append([_trim_start(remaining[k], trims[k]) for k in chain])
        # The parts trimmed from the starts of blocks remain, for other chains
        trimmed = [_start(remaining[k], trims[k]) for k in chain if trims[k] > 0]
        used = set(chain)
        remaining = sorted([b for k, b in enumerate(remaining) if k not in used] + trimmed)
    return result


def chain_length(chain):
    """
    The amount of homology in a chain: the total length of its blocks
    """
    return sum(block.length for block in chain)


class Homology(
    collections.namedtuple("Homology", "u_left, u_right, u_chromosome, v_left, v_right, v_chromosome, mrca, tmrca")
):
    """
    A segment of homology between genomes ``u`` and ``v``: the interval
    ``[u_left, u_right)`` of a chromosome of ``u`` is homologous to the interval of a
    chromosome of ``v`` with ends ``(v_left, v_right)`` (with ``v_left > v_right`` if it is
    reversed), through their most recent common ancestor ``mrca``, at time ``tmrca``.
    """

    __slots__ = ()


def find_homology(tables, u, v, *, u_chromosome=0, v_chromosome=0, time_cutoff=None):
    """
    Find all the homology between a chromosome of genome ``u`` and a chromosome of genome
    ``v``: every pair of positions, one in each, that trace back to the same position in a
    common ancestor, with the most recent such ancestor (the MRCA) of each pair.

    Unlike :meth:`~giglib.Tables.find_mrca_regions`, which stops tracing a position of
    ``u`` (or ``v``) once it has found a common ancestor, this carries on, so that it also
    finds older homology: e.g. between different copies of a region that was duplicated
    in a common ancestor of ``u`` and ``v`` (whose copies also each have more recent,
    allelic homology). So it can do much more work, which ``time_cutoff`` limits.

    .. note::
        This is written simply, using only the public API, and is not optimised. It may be
        more generally useful (e.g. for homology within a single genome, :issue:`135`, or
        for simplification), in which case it could be moved into the core of giglib,
        sharing its internals with :meth:`~giglib.Tables.find_mrca_regions` (see
        :issue:`207`).

    :param giglib.Tables tables: The tables (which need the iedges for each child to be
        adjacent and ordered by chromosome, as for
        :meth:`~giglib.tables.IEdgeTable.ids_for_child`)
    :param int u: The first genome (node ID)
    :param int v: The second genome (node ID)
    :param int u_chromosome: The chromosome of ``u`` (default 0)
    :param int v_chromosome: The chromosome of ``v`` (default 0)
    :param float time_cutoff: If given, only find homology whose MRCA is at most this old
    :return: A list of :class:`Homology` segments, sorted by their position in ``u``
    :rtype: list(Homology)
    """
    import heapq

    import portion as P

    if u == v:
        raise ValueError("u and v must be different genomes")
    if time_cutoff is None:
        time_cutoff = float("inf")
    node_time = tables.nodes.time
    # The material of u (side 0) and v (side 1) traced up to each node: for each node, a
    # dict mapping (side, chromosome of the node, map back to the original genome) to the
    # intervals of the node's genome (a portion interval set)
    material = {}
    heap = []

    def add(node, side, chromosome, to_original, intervals):
        if node not in material:
            material[node] = {}
            heapq.heappush(heap, (node_time[node], node))
        key = (side, chromosome, to_original)
        material[node][key] = material[node].get(key, P.empty()) | intervals

    identity = PositionMap(1, 0)
    for side, (genome, chromosome) in enumerate([(u, u_chromosome), (v, v_chromosome)]):
        add(genome, side, chromosome, identity, P.closedopen(0, P.inf))

    # The pairs of positions already reported: for each map from u to v (and the pair of
    # chromosomes), the positions of u. As nodes are visited from the most recent, a pair
    # is reported at its MRCA, and not again when it meets again in older ancestors.
    reported = {}
    result = []
    while heap:
        time, node = heapq.heappop(heap)
        if time > time_cutoff:
            break
        here = material.pop(node)
        for (side_u, chr_u, to_u), u_intervals in here.items():
            if side_u != 0:
                continue
            for (side_v, chr_v, to_v), v_intervals in here.items():
                if side_v != 1 or chr_v != chr_u:
                    continue
                for interval in u_intervals & v_intervals:
                    # The ends of the shared interval, in u and in v
                    u_ends = to_u.interval(interval.lower, interval.upper)
                    v_ends = to_v.interval(interval.lower, interval.upper)
                    if u_ends[0] > u_ends[1]:
                        u_ends, v_ends = u_ends[::-1], v_ends[::-1]
                    u_to_v = PositionMap.from_intervals(u_ends, v_ends)
                    key = (u_to_v, u_chromosome, v_chromosome)
                    new = P.closedopen(*u_ends) - reported.get(key, P.empty())
                    reported[key] = reported.get(key, P.empty()) | new
                    for piece in new:
                        if piece.empty:
                            continue
                        result.append(
                            Homology(
                                piece.lower,
                                piece.upper,
                                u_chromosome,
                                *u_to_v.interval(piece.lower, piece.upper),
                                v_chromosome,
                                node,
                                time,
                            )
                        )
        # Trace the material up to the parents
        for (side, chromosome, to_original), intervals in here.items():
            for ie_id in tables.iedges.ids_for_child(node, chromosome=chromosome):
                ie = tables.iedges[ie_id]
                overlap = intervals & P.closedopen(ie.child_left, ie.child_right)
                if overlap.empty or node_time[ie.parent] > time_cutoff:
                    continue
                to_parent = PositionMap.from_iedge(ie)
                parent_intervals = P.empty()
                for interval in overlap:
                    ends = to_parent.interval(interval.lower, interval.upper)
                    parent_intervals |= P.closedopen(min(ends), max(ends))
                add(ie.parent, side, ie.parent_chromosome, to_parent.inverse().then(to_original), parent_intervals)
    result.sort()
    return result
