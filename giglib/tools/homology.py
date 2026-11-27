"""
Tools for the homology between two genomes in a GIG, as described in
:ref:`sec_design_homology`. Homology is defined by shared ancestry in the graph, without
any reference: two positions are homologous if they trace back to the same position in a
common ancestor. The homology itself is found by :meth:`giglib.Tables.find_homology`
(see :ref:`sec_usage_homology`); this module adds:

- :func:`homology_blocks`: the homology, as blocks, each a maximal interval with a single
  map from one genome to the other (merging adjacent homology with different common
  ancestors but the same map);
- :func:`chains`: joining blocks into chains that are colinear, separated by small gaps
  (e.g. at microsatellites whose lengths differ), as a heuristic for allelic homology;
- :func:`random_homologous_positions`: a random pair of homologous positions, e.g. for
  placing a crossover;
- :func:`plot`: a dot plot of the homology between two genomes, coloured by the time to
  the most recent common ancestor.

This module is experimental: see :ref:`sec_tools`. It provides *mechanism* (finding
homology, and chaining it), not *policy* (e.g. where crossovers are allowed), which is up
to the simulators that use it. Discussion: :issue:`202`.
"""

import collections

from giglib import Position, PositionMap


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
    :meth:`~giglib.Tables.find_homology` (in its default mode), and adjacent regions of
    ``u`` whose homology in ``v`` has the same map (e.g. through different common
    ancestors) are merged into a single block. A region of ``u`` can be in several blocks, if it is
    homologous to several regions of ``v`` (e.g. copies of a duplicated region).

    :param giglib.Tables tables: The tables (which need the flags required by
        :meth:`~giglib.Tables.find_homology`)
    :param int u: The first genome (node ID)
    :param int v: The second genome (node ID)
    :param int u_chromosome: The chromosome of ``u`` (default 0)
    :param int v_chromosome: The chromosome of ``v`` (default 0)
    :return: A list of :class:`Block` objects
    :rtype: list(Block)
    """
    homology = tables.find_homology(u, v, u_chromosome=u_chromosome, v_chromosome=v_chromosome)
    pieces = {(h.u.left, h.u.right, h.v.left, h.v.right) for h in homology}
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
    # Blocks are ordered by where they end in u: a block can only follow one that ends
    # before it does, so every possible predecessor of a block comes before it
    remaining = sorted(blocks, key=_end_order)
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
        remaining = sorted([b for k, b in enumerate(remaining) if k not in used] + trimmed, key=_end_order)
    return result


def _end_order(block):
    return (block.u_chromosome, block.u_right, block.u_left, block.v_chromosome, block.v_left, block.v_right)


def chain_length(chain):
    """
    The amount of homology in a chain: the total length of its blocks
    """
    return sum(block.length for block in chain)


HomologousPositions = collections.namedtuple("HomologousPositions", "u, v, opposite_orientations")
HomologousPositions.__doc__ = """
A pair of homologous positions, as returned by :func:`random_homologous_positions`: ``u``
and ``v`` are :class:`~giglib.Position` tuples, and ``opposite_orientations`` is whether
one is reversed relative to the other (a property of the pair, not of either position).
"""


def random_homologous_positions(homology, rng):
    """
    Choose a pair of homologous positions at random, e.g. to place a crossover. A
    position of the common ancestors is chosen uniformly from the segments of homology
    (each segment of a common ancestor counting once, however many copies of it there are
    in ``u`` and ``v``), and then, if it has several copies in ``u`` or in ``v`` (e.g. through
    a duplication), one of each at random. How to choose the position is a policy of the
    simulator, so other simulators may choose differently.

    .. note::
        Adjacent segments are not merged, so if this is used to locate a breakpoint, the
        choice of whether to place the breakpoint to the left or right of the returned
        positions is left to the user.

    :param list(giglib.Homology) homology: The homology, as returned by
        :meth:`~giglib.Tables.find_homology` in its default mode, ``"first_mrca"``. (In
        that mode, every copy in ``u`` of a segment of a common ancestor is homologous,
        through it, to every copy in ``v``, and the segments of each common ancestor don't
        overlap. Neither is true in mode ``"all"``, in which some pairs of copies have a
        more recent common ancestor.)
    :param rng: A numpy random number generator, e.g. made by
        ``numpy.random.default_rng()``
    :raises ValueError: If ``homology`` isn't of this form
    :rtype: HomologousPositions
    """
    # Group the homology by segment of a common ancestor, in the order they first appear,
    # with the ends of the segments in u and v that correspond to the ends of the
    # ancestral segment, in order (so they may be reversed relative to it)
    groups = {}
    num_records = collections.Counter()
    for h in homology:
        mrca_ends, u_ends, v_ends = h.mrca_segment[:2], h.u[:2], h.v[:2]
        if mrca_ends[0] > mrca_ends[1]:
            mrca_ends, u_ends, v_ends = mrca_ends[::-1], u_ends[::-1], v_ends[::-1]
        key = (h.mrca, *mrca_ends, h.mrca_segment.chromosome)
        # The copies in u and in v, as dicts used as sets that keep their order
        u_copies, v_copies = groups.setdefault(key, ({}, {}))
        u_copies[(*u_ends, h.u.chromosome)] = None
        v_copies[(*v_ends, h.v.chromosome)] = None
        num_records[key] += 1
    # Check that each ancestral segment pairs every copy in u with every copy in v, and
    # that the segments of each common ancestor don't overlap
    previous_end = {}
    for mrca, left, right, chromosome in sorted(groups, key=lambda k: (k[0], k[3], k[1])):
        u_copies, v_copies = groups[(mrca, left, right, chromosome)]
        if num_records[(mrca, left, right, chromosome)] != len(u_copies) * len(v_copies):
            raise ValueError("The homology must be found with mode 'first_mrca'")
        if left < previous_end.get((mrca, chromosome), left):
            raise ValueError("The homology must be found with mode 'first_mrca'")
        previous_end[(mrca, chromosome)] = right
    total_length = sum(right - left for _, left, right, _ in groups)
    loc = rng.integers(total_length)
    for (_, left, right, _), (u_copies, v_copies) in groups.items():
        if loc < right - left:
            u_copies, v_copies = list(u_copies), list(v_copies)
            u_copy = u_copies[0] if len(u_copies) == 1 else u_copies[rng.integers(len(u_copies))]
            v_copy = v_copies[0] if len(v_copies) == 1 else v_copies[rng.integers(len(v_copies))]
            to_u = PositionMap.from_intervals((left, right), u_copy[:2])
            to_v = PositionMap.from_intervals((left, right), v_copy[:2])
            position = left + int(loc)
            return HomologousPositions(
                Position(to_u.position(position), u_copy[2]),
                Position(to_v.position(position), v_copy[2]),
                to_u.is_reversing != to_v.is_reversing,
            )
        loc -= right - left


def plot(
    tables, u, v, *, mode="all", u_chromosome=0, v_chromosome=0, time_cutoff=None, ax=None, cmap="viridis", linewidth=3
):
    """
    Plot the homology between a chromosome of genome ``u`` and a chromosome of genome
    ``v`` as a dot plot: each homologous segment is a line from its start to its end, at
    its position in ``u`` (horizontally) and in ``v`` (vertically), coloured by its time to
    the most recent common ancestor (TMRCA). By default, all the homology is shown (found
    with :meth:`~giglib.Tables.find_homology`, with ``mode="all"``), including that older
    than a more recent common ancestor of other positions. If everything is ultimately
    homologous (e.g. descends from a single ancestral position), that can be too much, and
    ``mode="closest_homologues"`` shows only the closest homology of each position.
    Allelic homology is a diagonal line, with a
    recent TMRCA; copies (e.g. of a TE, or of a copy-number variant) are lines off the
    diagonal, coloured by the age of the duplication; and reversed homology (e.g. an
    inversion) runs from top left to bottom right. With the default colour map, recent
    homology is dark blue and older homology lighter, up to yellow; places with no
    homology are left white, as if their TMRCA were infinite. See :ref:`sec_design_homology`.

    This needs matplotlib, which is not a requirement of giglib itself.

    :param giglib.Tables tables: The tables (which need the flags required by
        :meth:`~giglib.Tables.find_homology`)
    :param int u: The first genome (node ID)
    :param int v: The second genome (node ID)
    :param str mode: Which homology to show: the ``mode`` of
        :meth:`~giglib.Tables.find_homology` (default ``"all"``)
    :param int u_chromosome: The chromosome of ``u`` (default 0)
    :param int v_chromosome: The chromosome of ``v`` (default 0)
    :param float time_cutoff: If given, only show homology with a TMRCA at most this old
        (when everything is ultimately homologous, this hides ancient homology, and limits
        the work done)
    :param ax: The matplotlib axes to draw on (default: a new figure)
    :param str cmap: The matplotlib colour map for the TMRCA (default ``"viridis"``, so
        that more recent homology is darker)
    :param float linewidth: The width of the lines
    :return: The axes
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection

    homology = tables.find_homology(
        u, v, mode=mode, u_chromosome=u_chromosome, v_chromosome=v_chromosome, time_cutoff=time_cutoff
    )
    node_times = tables.nodes.time
    # Draw older homology first, so that more recent homology is drawn on top of it
    homology.sort(key=lambda h: -node_times[h.mrca])
    lines = [[(h.u.left, h.v.left), (h.u.right, h.v.right)] for h in homology]
    times = [node_times[h.mrca] for h in homology]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    collection = LineCollection(lines, array=np.array(times), cmap=cmap, linewidths=linewidth)
    ax.add_collection(collection)
    # The axes cover every plotted end (in v, both ends, as a segment may be reversed), and
    # the iedges above each genome (a root has none, so only the plotted ends count)
    u_ends = [x for h in homology for x in (h.u.left, h.u.right)]
    v_ends = [y for h in homology for y in (h.v.left, h.v.right)]
    ax.set_xlim(0, max([*u_ends, *_child_rights(tables, u, u_chromosome)], default=1))
    ax.set_ylim(0, max([*v_ends, *_child_rights(tables, v, v_chromosome)], default=1))
    ax.set_aspect("equal")
    ax.set_xlabel(f"position in genome {u}")
    ax.set_ylabel(f"position in genome {v}")
    ax.figure.colorbar(collection, ax=ax, label="TMRCA", shrink=0.8)
    return ax


def _child_rights(tables, u, chromosome):
    """
    The right ends of the iedges above a chromosome of ``u``
    """
    ids = tables.iedges.ids_for_child(u, chromosome=chromosome)
    return tables.iedges.child_right[ids].tolist()
