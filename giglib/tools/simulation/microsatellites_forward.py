"""
A forward-time simulator with microsatellites, which places crossovers using chains of
homology (see :mod:`giglib.tools.homology`).
"""

import numpy as np

from giglib.tools import homology

from .one_break_inversions import DTWF_one_break_no_rec_inversions_slow_sim


class DTWF_microsatellite_sim(DTWF_one_break_no_rec_inversions_slow_sim):
    """
    A simple simulator, mainly intended as a test harness for the homology algorithms in
    :mod:`giglib.tools.homology` (it generates genomes with many small differences in
    length, and many small tandem copies), rather than as a realistic model.

    A DTWF simulator with one crossover per chromosome per meiosis, and microsatellites:
    loci of tandem repeat units whose number changes by one with probability
    ``mutation_rate`` per locus per meiosis (see :issue:`202`). An expansion copies the
    last repeat unit of the locus from the parent (so the copy's homology is recorded,
    as for slippage), and a contraction deletes the last repeat unit.

    Crossovers are only placed in *chains* of homology between the two parental genomes
    at least ``min_chain_length`` positions long, whose blocks are separated by gaps of at
    most ``max_gap`` positions (see :ref:`sec_design_homology`), and never inside a
    microsatellite. So crossovers are allelic, however many repeat units the parents
    carry. The simulator records where the microsatellites are in each genome (in
    ``self.loci``), as it is what creates them.

    :param list loci: The microsatellites, as a list of ``(position, unit_length,
        num_units)`` tuples in the coordinates of the founding genomes (on chromosome 0),
        which must not overlap
    :param float mutation_rate: The probability that a locus gains or loses a repeat unit
        in each meiosis (with equal probability, but never losing its last unit)
    :param int min_chain_length: The shortest chain of homology in which a crossover can
        be placed
    :param int max_gap: The largest gap between consecutive blocks of a chain
    """

    def __init__(self, loci, *, mutation_rate, min_chain_length, max_gap, skip_validate=False, initial_sizes=None):
        super().__init__(skip_validate=skip_validate, initial_sizes=initial_sizes)
        self.founder_loci = sorted(loci)
        if self.founder_loci and self.founder_loci[0][0] < 0:
            raise ValueError("Microsatellite loci must be within the sequence")
        for (position, unit, num), next_locus in zip(self.founder_loci, self.founder_loci[1:]):
            if position + unit * num > next_locus[0]:
                raise ValueError("Microsatellite loci must not overlap")
        self.mutation_rate = mutation_rate
        self.min_chain_length = min_chain_length
        self.max_gap = max_gap
        self.loci = {}  # node: list of [locus_id, start, unit_length, num_units], in order

    def initialise_population(self, time, size, seq_lens, node_flags=None):
        if set(seq_lens) != {0}:
            raise ValueError("This simulator only supports a single chromosome (0)")
        if (
            self.founder_loci
            and self.founder_loci[-1][0] + self.founder_loci[-1][1] * self.founder_loci[-1][2] > (seq_lens[0])
        ):
            raise ValueError("Microsatellite loci must be within the sequence")
        pop = super().initialise_population(time, size, seq_lens, node_flags)
        self.loci = {}
        for nodes in pop.values():
            for u in nodes:
                self.loci[u] = [[i, *locus] for i, locus in enumerate(self.founder_loci)]
        return pop

    def _in_locus(self, u, x):
        return any(start <= x < start + unit * num for _, start, unit, num in self.loci[u])

    def find_crossover(self, mum, dad):
        """
        Return homologous positions in ``mum`` and ``dad``, either side of which a
        crossover can be placed: in the same orientation, in a long enough chain of
        homology, and not in a microsatellite
        """
        blocks = homology.homology_blocks(self.tables, mum, dad)
        allowed = [
            block
            for chain in homology.chains(blocks, max_gap=self.max_gap)
            if homology.chain_length(chain) >= self.min_chain_length
            for block in chain
            if not block.is_reversed
        ]
        if len(allowed) == 0:
            raise ValueError(f"No chain of homology between {mum} and {dad} is long enough for a crossover")
        weights = np.array([block.length for block in allowed], dtype=float)
        for _ in range(self.num_tries_for_breakpoint):
            block = allowed[self.rng.choice(len(allowed), p=weights / weights.sum())]
            x = block.u_left + int(self.rng.integers(block.length))
            y = block.map.position(x)
            if not self._in_locus(mum, x) and not self._in_locus(dad, y):
                return x, y
        raise ValueError(
            f"Could not find a crossover outside microsatellites after {self.num_tries_for_breakpoint} tries"
        )

    def add_inheritance_paths(self, parent_nodes, child, _):
        mum, dad = parent_nodes
        x, y = self.find_crossover(mum, dad)
        # Break just before or just after the homologous positions, and choose at random
        # which parent gives the left of the child's genome
        rnd = int(self.rng.integers(4))
        breaks = {mum: x + (rnd & 1), dad: y + (rnd & 1)}
        left_parent, right_parent = (mum, dad) if rnd & 2 else (dad, mum)
        left_break, right_break = breaks[left_parent], breaks[right_parent]
        right_length = self.tables.iedges.max_pos_as_child(right_parent, chromosome=0)
        # The child's genome, as a list of [parent, parent_start, length] segments in order
        segments = [
            segment
            for segment in ([left_parent, 0, left_break], [right_parent, right_break, right_length - right_break])
            if segment[2] > 0
        ]
        # Microsatellites to the left of the break come from the left parent, and to the
        # right of it from the right parent (shifted to the child's coordinates)
        loci = [list(locus) for locus in self.loci[left_parent] if locus[1] < left_break]
        loci += [
            [i, start - right_break + left_break, unit, num]
            for i, start, unit, num in self.loci[right_parent]
            if start >= right_break
        ]
        if [locus[0] for locus in loci] != list(range(len(self.founder_loci))):
            raise AssertionError("A crossover was not allelic: each microsatellite must be inherited once")
        # Mutations: gain or lose one repeat unit, at the end of the locus
        shift = 0
        for locus in loci:
            locus[1] += shift
            _, start, unit, num = locus
            if self.rng.random() < self.mutation_rate:
                end = start + unit * num
                if self.rng.random() < 0.5:  # expansion: copy the last unit from the parent
                    parent, parent_start = _segment_source(segments, end - unit)
                    i = _split_segments(segments, end)
                    segments.insert(i, [parent, parent_start, unit])
                    locus[3] += 1
                    shift += unit
                elif num > 1:  # contraction: delete the last unit
                    i = _split_segments(segments, end - unit)
                    j = _split_segments(segments, end)
                    del segments[i:j]
                    locus[3] -= 1
                    shift -= unit
        self.loci[child] = loci
        child_left = 0
        for parent, parent_start, length in _merge_segments(segments):
            self.add_iedge_row(
                child,
                parent,
                child_left=child_left,
                child_right=child_left + length,
                parent_left=parent_start,
                parent_right=parent_start + length,
            )
            child_left += length


def _segment_source(segments, x):
    """
    The (parent, parent position) of position ``x`` of a genome made of ``segments``
    """
    start = 0
    for parent, parent_start, length in segments:
        if x < start + length:
            return parent, parent_start + x - start
        start += length
    raise ValueError(f"Position {x} is beyond the end of the genome")


def _split_segments(segments, x):
    """
    Split the segments (in place) so that one starts at position ``x``, and return its
    index (``len(segments)`` if ``x`` is the end of the genome)
    """
    start = 0
    for i, (parent, parent_start, length) in enumerate(segments):
        if x == start:
            return i
        if x < start + length:
            segments[i : i + 1] = [
                [parent, parent_start, x - start],
                [parent, parent_start + x - start, start + length - x],
            ]
            return i + 1
        start += length
    if x == start:
        return len(segments)
    raise ValueError(f"Position {x} is beyond the end of the genome")


def _merge_segments(segments):
    """
    Merge consecutive segments that continue each other in the same parent
    """
    merged = []
    for parent, parent_start, length in segments:
        if merged and merged[-1][0] == parent and merged[-1][1] + merged[-1][2] == parent_start:
            merged[-1][2] += length
        else:
            merged.append([parent, parent_start, length])
    return merged
