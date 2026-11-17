"""
Define a generalised (genetic) inheritance graph object, which is
a validated set of GIG tables with some extra indexes etc for
efficient access.
"""

import json
import warnings
from collections import namedtuple

import numpy as np
import tskit

from .constants import Const, ValidFlags
from .tables import (
    IEdgeTableRow,
    IndividualTableRow,
    MutationTableRow,
    NodeTableRow,
    Tables,
    _IEdgeMethods,
    _MutationMethods,
    _NodeMethods,
)
from .util import as_id, complement_allele, is_unknown_time, warn_tskit_name


def _read_only(arr):
    """
    Return a read-only view of a numpy array (sharing the same memory)
    """
    if arr.flags.writeable:
        arr = arr.view()
        arr.flags.writeable = False
    return arr


class Graph:
    """
    This is similar to a _tskit_ :class:`~tskit:tskit.TreeSequence`.
    An instance can be created from a :class:`Tables` object by calling
    :meth:`Tables.graph`, which freezes the tables into a GIG.
    """

    def __init__(self, tables):
        """
        Create a :class:`Graph` from a set of Tables. This is for internal use only:
        the canonical way to do this is to use  {meth}`Tables.graph`
        """
        if not isinstance(tables, Tables):
            raise ValueError("tables must be a giglib.Tables object")
        self.tables = tables
        self._validate()
        self._validate_mutations()
        self._check_allele_complements()
        self.tables.freeze()
        # Cached arrays: read-only, so they can safely be shared between views
        self._iedge_map_sorted_by_parent = _read_only(self._iedge_map_sorted_by_parent)
        self._sample_ids = _read_only(self.tables.sample_nodes())

    @classmethod
    def from_tree_sequence(cls, ts):
        """
        Construct a GIG from a tree sequence.

        :param tskit.TreeSequence ts: A tree sequence object
        :return: A new GIG object
        :rtype: Graph
        """
        return cls(Tables.from_tree_sequence(ts))

    def _validate(self):
        assert hasattr(self, "tables")
        # Extra validation for a GIG here, e.g. edge transformations are
        # valid, etc.

        # Cached variables. We define these up top, as all validated GIGs
        # should have them, even if they have no edges

        # set up some aliases for use in the rest of the function.
        iedge_table = self.tables.iedges
        # These should all be references to the underlying arrays
        child = iedge_table.child
        parent = iedge_table.parent
        child_left = iedge_table.child_left
        child_right = iedge_table.child_right
        parent_left = iedge_table.parent_left
        parent_right = iedge_table.parent_right
        node_times = self.tables.nodes.time

        # the _id_range_for_parent dict mirrors the ._id_range_for_parent
        # dictionary, cached in the iedges table. However, this indexes into
        # a mapping which specifies iedges sorted primarily by parent ID.
        self._id_range_for_parent = {}
        self._iedge_map_sorted_by_parent = np.arange(
            len(iedge_table)
        )  # to overwrite later - initialised here in case we return early with 0 edges

        # TODO: should cache node spans here,
        # see https://github.com/hyanwong/giglib/issues/27

        if len(iedge_table) == 0:
            return

        # Check that all node IDs are non-negative and less than #nodes
        for nm, nodes in (("child", child), ("parent", parent)):
            if nodes.min() < 0:
                raise ValueError(f"Iedge {np.argmin(nodes)} contains negative {nm} ID.")
            if nodes.max() >= len(self.nodes):
                raise ValueError(f"Iedge {np.argmax(nodes)} contains {nm} ID >= num nodes")

        # check all parents are strictly older than their children
        # (NB: this also allows a time of np.inf for an "ultimate root" node)
        for i, ie in enumerate(iedge_table):
            if node_times[ie.parent] <= node_times[ie.child]:
                raise ValueError(f"Edge {i}: parent node {ie.parent} not older than child {ie.child}")

        # Check that all iedges have same absolute parent span as child span
        parent_spans = parent_right - parent_left
        child_spans = child_right - child_left
        span_equal = np.abs(parent_spans) == np.abs(child_spans)
        if not np.all(span_equal):
            bad = np.where(~span_equal)[0]
            raise ValueError(f"iedges {bad} have different parent and child spans")

        # Check that child left is always < child right (also checks nonzero span)
        if np.any(child_spans <= 0):
            raise ValueError(f"child_left >= child_right for iedges {np.where(child_spans < 0)[0]}")

        # Check iedges are sorted so that nodes are in decreasing order
        # of child time, with ties broken by child ID, increasing
        timediff = np.diff(node_times[child])
        chromodiff = np.diff(iedge_table.child_chromosome)
        node_id_diff = np.diff(child)

        if np.any(timediff > 0):
            raise ValueError("iedges are not sorted by child time, descending")

        if np.any(node_id_diff[timediff == 0] < 0):
            raise ValueError("iedges are not sorted by child time (desc) then ID (asc)")
        # must be adjacent
        iedge_table.set_bitflag(ValidFlags.IEDGES_FOR_CHILD_ADJACENT)

        # Check within a child, edges are sorted by chromosome ascending
        if np.any(np.diff(iedge_table.child_chromosome)[node_id_diff == 0] < 0):
            raise ValueError("iedges for a given child are not sorted by chromosome")
        iedge_table.set_bitflag(ValidFlags.IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC)

        # Check within a child & chromosome, edges are sorted by left coord ascending
        if np.any(np.diff(child_left)[np.logical_and(node_id_diff == 0, chromodiff == 0)] <= 0):
            # print(np.diff(self.tables.iedges.child_left))
            # print(np.logical_and(node_id_diff == 0, chromodiff==0))
            raise ValueError("iedges for a given child/chr are not sorted by left pos")
        iedge_table.set_bitflag(ValidFlags.IEDGES_FOR_CHILD_SECONDARY_ORDER_LEFT_ASC)

        # CREATE CACHED VARIABLES

        # Should probably cache more stuff here, e.g. list of samples

        # index to sort edges so they are sorted as in tskit: first by parent time
        # then grouped by parent id. Within each group with the same parent ID,
        # the edges are sorted by parent_chromosome, then child_left
        self._iedge_map_sorted_by_parent[:] = np.lexsort(
            (
                # NB: this is not totally symmetrical with the edge map sorted by child,
                # because a parent node can have multiple children at the same position
                child,
                # sort by max(parent_left, parent_right). This mirrors
                # the iedge row sort by max(child_left, child_right) == shild_left
                np.where(parent_right > parent_left, parent_right, parent_left),
                iedge_table.parent_chromosome,
                parent,
                node_times[parent],  # Primary key
            )
        )
        last_parent = -1
        for j, idx in enumerate(self._iedge_map_sorted_by_parent):
            ie = self.iedges[idx]
            if ie.parent != last_parent:
                # encountered new parent node
                self._id_range_for_parent[ie.parent] = {}
            if ie.parent_chromosome not in self._id_range_for_parent[ie.parent]:
                self._id_range_for_parent[ie.parent][ie.parent_chromosome] = [j, j + 1]
            else:
                self._id_range_for_parent[ie.parent][ie.parent_chromosome][1] = j + 1
            last_parent = ie.parent

        # Check every location has only one parent for each chromosome
        for u in range(len(self.nodes)):
            for chromosome in self.tables.iedges.chromosomes_as_child(u):
                prev_right = -np.inf
                for ie in self.iedges_for_child(u, chromosome=chromosome):
                    if ie.child_left < prev_right:
                        raise ValueError(f"Node {u} has multiple or duplicate parents at position" f" {ie.child_left}")
                    prev_right = ie.child_right
        self.tables.iedges.flags = ValidFlags.GIG

    def _iedge_id_above(self, u, chromosome, position):
        """
        Return the ID of the iedge above node ``u`` that contains ``position`` on the
        given chromosome of ``u`` (or None if ``u`` has no parent at that position)
        """
        ids = self.tables.iedges.ids_for_child(u, chromosome=chromosome)
        if len(ids) == 0:
            return None
        start, end = ids[0], ids[-1] + 1
        j = np.searchsorted(self.tables.iedges.child_left[start:end], position, side="right") - 1
        if j >= 0 and position < self.tables.iedges.child_right[start + j]:
            return start + j
        return None

    def _validate_mutations(self):
        """
        Check the mutations table (requires the iedges to have been validated)
        """
        tables = self.tables
        mutations = tables.mutations
        # Rebuild the node index before the tables are frozen, so that a Graph can never
        # use a stale index. Public column accessors are intended to be read-only (see
        # BaseTable._column), so columns should only change through methods that keep the
        # index up to date. But the index would still go stale if the underlying ``_data``
        # array were edited directly (as some library code does), so this check is
        # retained as a cheap safeguard: it costs little compared to the validation below.
        mutations._rebuild_index()
        n = len(mutations)
        if n == 0:
            return
        node = mutations.node
        chromosome = mutations.chromosome
        position = mutations.position
        time = mutations.time
        node_time = tables.nodes.time
        if node.min() < 0 or node.max() >= len(tables.nodes):
            bad = np.flatnonzero((node < 0) | (node >= len(tables.nodes)))
            raise ValueError(f"Mutations {bad} refer to nodes that are not in the node table")
        if position.min() < 0:
            raise ValueError(f"Mutations {np.flatnonzero(position < 0)} have negative positions")
        known = ~is_unknown_time(time)
        if not np.all(np.isfinite(time[known])):
            bad = np.flatnonzero(known & ~np.isfinite(time))
            raise ValueError(f"Mutations {bad} have times that are neither finite nor UNKNOWN_TIME")
        if not np.array_equal(tables._mutation_sort_order(), np.arange(n)):
            raise ValueError("Mutations are not in the required order: call tables.sort() first")

        too_young = known & (time < node_time[node])
        if np.any(too_young):
            i = np.argmax(too_young)
            raise ValueError(f"Mutation {i} has time {time[i]}, younger than its node ({node_time[node[i]]})")

        prev_key = None
        prev_time = np.inf
        for i in np.arange(n):
            key = (node[i], chromosome[i], position[i])
            if key != prev_key:
                prev_key = key
                prev_time = np.inf
            if known[i]:
                if time[i] > prev_time:
                    raise ValueError(
                        f"Mutation {i} has time {time[i]}, older than a previous mutation at the same "
                        "node and position: mutations at the same position must be in time order"
                    )
                prev_time = time[i]
                ie_id = self._iedge_id_above(node[i], chromosome[i], position[i])
                if ie_id is not None:
                    parent_time = node_time[tables.iedges.parent[ie_id]]
                    if time[i] >= parent_time:
                        raise ValueError(
                            f"Mutation {i} has time {time[i]}, which is not younger than the parent "
                            f"of node {node[i]} at that position (time {parent_time})"
                        )

    def _check_allele_complements(self):
        """
        Warn if there are inversions and multi-character alleles that cannot be
        complemented using the allele complements map
        """
        iedges = self.tables.iedges
        if len(self.tables.mutations) == 0 or not np.any(iedges.parent_right < iedges.parent_left):
            return
        complements = self.tables.allele_complements
        # Rule 3 of the complement rules: multi-character, not in the map, and with
        # some characters that are not in the map
        uncomplementable = sorted(
            {
                a
                for a in self.tables.mutations.derived_state
                if len(a) > 1 and a not in complements and not all(c in complements for c in a)
            }
        )
        if len(uncomplementable) > 0:
            warnings.warn(
                f"This GIG contains inversions, but the alleles {uncomplementable} cannot be "
                "complemented using tables.allele_complements, so will be treated as "
                "strand-symmetric. To silence this warning, add them to the map (mapping an "
                "allele to itself if it really is symmetric).",
                UserWarning,
                stacklevel=4,  # point to the caller of Tables.graph()
            )

    def _alleles(self, u, chromosome, left, right):
        """
        Return a list of the alleles at positions ``left`` to ``right - 1`` of node
        ``u``'s genome, with ``None`` for unknown (missing) alleles.
        See :meth:`_known_alleles`.
        """
        result = [None] * (right - left)
        for x, allele in self._known_alleles(u, chromosome, left, right).items():
            result[x - left] = allele
        return result

    def _known_alleles(self, u, chromosome, left, right):
        """
        Return a dict mapping positions (between ``left`` and ``right - 1``) of node
        ``u``'s genome to their alleles, omitting positions whose allele is unknown
        (missing). The time taken depends on the number of mutations and iedges
        traversed, not on the length of the interval.

        Each position is traced rootwards until it reaches a mutation (the most recent
        one at that node and position), or a node with no parent at that position.
        Each position has at most one parent, so tracing never branches. We track
        intervals in the coordinates of the current node, with an affine map
        ``position = m * x + k`` back to the position in ``u``, where ``m == -1`` if
        the interval has passed through an odd number of inversions.
        """
        result = {}
        complements = self.tables.allele_complements
        iedges = self.tables.iedges
        mutations = self.tables.mutations
        mut_position = mutations.position
        state_col = mutations._extra_data_cols["derived_state"]
        stack = [(u, chromosome, left, right, 1, 0)]
        while stack:
            v, c, a, b, m, k = stack.pop()
            # Resolve positions with mutations above v. The last mutation at a position
            # is the most recent, so determines the allele inherited from v.
            resolved = []
            start, end = mutations._range_for_node(v, c)
            if end > start:
                j = start + np.searchsorted(mut_position[start:end], a, side="left")
                hi = start + np.searchsorted(mut_position[start:end], b, side="left")
                while j < hi:
                    x = mut_position[j]
                    while j + 1 < hi and mut_position[j + 1] == x:
                        j += 1
                    state = mutations._extra_data[j][state_col]
                    result[m * x + k] = state if m == 1 else complement_allele(state, complements)
                    resolved.append(x)
                    j += 1
            # Trace the unresolved intervals up through the iedges above v
            ie_ids = iedges.ids_for_child(v, chromosome=c)
            if len(ie_ids) == 0:
                continue  # no parents: unresolved positions remain missing
            ie_start, ie_end = ie_ids[0], ie_ids[-1] + 1
            child_left = iedges.child_left[ie_start:ie_end]
            child_right = iedges.child_right[ie_start:ie_end]
            for p, q in zip([a, *(x + 1 for x in resolved)], [*resolved, b]):
                if p >= q:
                    continue
                i = np.searchsorted(child_right, p, side="right")
                while i < len(child_left) and child_left[i] < q:
                    ov_a, ov_b = max(p, child_left[i]), min(q, child_right[i])
                    if ov_a < ov_b:
                        ie = iedges[ie_start + i]
                        if ie.parent_left < ie.parent_right:
                            d = ie.parent_left - ie.child_left
                            stack.append((ie.parent, ie.parent_chromosome, ov_a + d, ov_b + d, m, k - m * d))
                        else:
                            # Inversion: parent position x' = s - x
                            s = ie.child_left + ie.parent_left - 1
                            stack.append((ie.parent, ie.parent_chromosome, s - ov_b + 1, s - ov_a + 1, -m, k + m * s))
                    i += 1
        return result

    def haplotype(self, u, *, chromosome=0, left=0, right=None, missing_data_character="N"):
        """
        Return the sequence of alleles in node ``u``'s genome, as a string. Each
        position takes the allele of the most recent mutation above it (traced up
        through the iedges, and complemented across inversions using
        ``tables.allele_complements``: alleles that cannot be complemented using the
        map, such as ``"0"`` or ``"1"``, are left unchanged). Positions with no mutation
        above them are missing data. See :ref:`sec_design_mutations`.

        For a GIG converted from a tree sequence ``ts`` without a reference sequence,
        ``haplotype(u, right=int(ts.sequence_length))`` for a sample ``u`` matches the
        corresponding sequence from :meth:`tskit:tskit.TreeSequence.alignments`. Note
        that ``right`` should be given explicitly, as the default is the length of
        ``u``'s genome implied by its iedges, which may be shorter.

        If any alleles have more or less than one character (e.g. representing small
        indels), the length of the returned string will differ from ``right - left``.

        :param int u: The node (or a :class:`~giglib.graph.Node` object)
        :param int chromosome: The chromosome in ``u``'s genome (default: 0)
        :param int left: The first position to return (default: 0)
        :param int right: The position after the last one to return. If ``None``
            (default), use the sequence length of ``u`` for this chromosome.
        :param str missing_data_character: The string to use for positions with
            unknown alleles (default: ``"N"``, as for :meth:`tskit:tskit.TreeSequence.alignments`)
        :return: The haplotype
        :rtype: str
        """
        u = as_id(u)
        if right is None:
            right = self.sequence_length(u, chromosome=chromosome)
        if left < 0 or right < left:
            raise ValueError(f"Invalid interval [{left}, {right})")
        alleles = self._alleles(u, chromosome, left, right)
        return "".join(missing_data_character if a is None else a for a in alleles)

    def allele_at(self, u, position, *, chromosome=0):
        """
        Return the allele at a position in node ``u``'s genome, given by the most recent
        mutation above that position (complemented across inversions using
        ``tables.allele_complements``, with alleles that cannot be complemented using
        the map left unchanged). See :ref:`sec_design_mutations`.

        :param int u: The node (or a :class:`~giglib.graph.Node` object)
        :param int position: The position, in ``u``'s coordinates
        :param int chromosome: The chromosome in ``u``'s genome (default: 0)
        :return: The allele, or ``None`` if it is unknown (missing data)
        :rtype: str
        """
        u = as_id(u)
        if position < 0:
            raise ValueError("position must be non-negative")
        return self._alleles(u, chromosome, position, position + 1)[0]

    @property
    def time_units(self):
        return self.tables.time_units

    @property
    def max_time(self):
        # Should also incorporate mutation times here
        return self.tables.nodes.time.max()

    @property
    def nodes(self):
        """
        A sequence-like view of all the :class:`~giglib.graph.Node` objects in this
        GIG, supporting ``len()``, indexing, and iteration. The node IDs are
        available as ``gig.nodes.ids``.
        """

        return GIGItemIterator(self.tables.nodes, Node)

    def filter_nodes(self, flags=None, time=None):
        """
        Return a sequence-like view of the :class:`~giglib.graph.Node` objects
        matching all the given criteria, supporting ``len()``, indexing, and
        iteration. The matching node IDs are available (as a read-only numpy array)
        via the ``.ids`` attribute of the returned view, e.g.
        ``gig.filter_nodes(time=0).ids``.

        :param int flags: If not ``None``, only return nodes for which *all* the
            bits in ``flags`` are set.
        :param time: If not ``None``, only return nodes at this time. This can be
            a single number, in which case node times must be exactly equal to this
            value, or a ``(start, end)`` pair, in which case node times must satisfy
            ``start <= time < end``.
        """
        if flags is None and time is None:
            return self.nodes
        candidates = None
        if flags is not None and (flags & Const.NODE_IS_SAMPLE):
            # Matching nodes must be a subset of the (cached) samples
            if flags == Const.NODE_IS_SAMPLE and time is None:
                return GIGItemIterator(self.tables.nodes, Node, ids=self._sample_ids)
            candidates = self._sample_ids
        ids = self.tables.nodes.filter(flags=flags, time=time, candidates=candidates)
        return GIGItemIterator(self.tables.nodes, Node, ids=ids)

    def sample_nodes(self, time=None):
        """
        Return a sequence-like view of the sample :class:`~giglib.graph.Node`
        objects. This is an alias for
        ``filter_nodes(flags=giglib.NODE_IS_SAMPLE, time=time)``. Sample node IDs
        are available via ``gig.sample_nodes().ids``: when ``time`` is ``None``
        this is a cached array, so is efficient to call repeatedly.
        """
        return self.filter_nodes(flags=Const.NODE_IS_SAMPLE, time=time)

    @property
    def individuals(self):
        """
        A sequence-like view of all the :class:`~giglib.graph.Individual` objects
        in this GIG, supporting ``len()``, indexing, and iteration.
        """
        return GIGItemIterator(self.tables.individuals, Individual)

    @property
    def iedges(self):
        """
        A sequence-like view of all the :class:`~giglib.graph.IEdge` objects
        in this GIG, supporting ``len()``, indexing, and iteration.
        """
        return GIGItemIterator(self.tables.iedges, IEdge)

    @property
    def mutations(self):
        """
        A sequence-like view of all the :class:`~giglib.graph.Mutation` objects
        in this GIG, supporting ``len()``, indexing, and iteration.
        """
        return GIGItemIterator(self.tables.mutations, Mutation)

    def mutations_for_node(self, u, *, chromosome=None):
        """
        Return a sequence-like view of the :class:`~giglib.graph.Mutation` objects
        above node ``u``, ordered by chromosome, then position, then the order in
        which they occurred. If ``chromosome`` is given, only return mutations on that
        chromosome of ``u``. To get mutation IDs instead, use
        :meth:`tables.MutationTable.ids_for_node`.
        """
        u = as_id(u)
        return GIGItemIterator(
            self.tables.mutations, Mutation, ids=self.tables.mutations.ids_for_node(u, chromosome=chromosome)
        )

    # tskit names which do not exist in giglib. These exist only to guide users
    # familiar with tskit towards the giglib equivalents, and emit a TskitNameWarning

    @property
    def num_nodes(self):
        warn_tskit_name("num_nodes", "len(gig.nodes)")
        return len(self.nodes)

    @property
    def num_edges(self):
        warn_tskit_name("num_edges", "len(gig.iedges)")
        return len(self.iedges)

    @property
    def num_individuals(self):
        warn_tskit_name("num_individuals", "len(gig.individuals)")
        return len(self.individuals)

    @property
    def num_mutations(self):
        warn_tskit_name("num_mutations", "len(gig.mutations)")
        return len(self.mutations)

    @property
    def num_samples(self):
        warn_tskit_name("num_samples", "len(gig.sample_nodes())")
        return len(self._sample_ids)

    def samples(self):
        """
        Not part of the giglib API: returns sample IDs, as in tskit, but emits a
        :class:`~giglib.TskitNameWarning`. Use :meth:`sample_nodes` to get sample
        :class:`~giglib.graph.Node` objects, or ``gig.sample_nodes().ids`` to get sample IDs.
        """
        warn_tskit_name("samples()", "gig.sample_nodes() for nodes, or gig.sample_nodes().ids for IDs")
        return self._sample_ids

    def node(self, u):
        """
        Not part of the giglib API (emits a :class:`~giglib.TskitNameWarning`): use ``gig.nodes[u]``
        """
        warn_tskit_name("node(u)", "gig.nodes[u]")
        return self.nodes[u]

    def mutation(self, i):
        """
        Not part of the giglib API (emits a :class:`~giglib.TskitNameWarning`): use ``gig.mutations[i]``
        """
        warn_tskit_name("mutation(i)", "gig.mutations[i]")
        return self.mutations[i]

    def individual(self, i):
        """
        Not part of the giglib API (emits a :class:`~giglib.TskitNameWarning`): use ``gig.individuals[i]``
        """
        warn_tskit_name("individual(i)", "gig.individuals[i]")
        return self.individuals[i]

    def edges(self):
        """
        Not part of the giglib API (emits a :class:`~giglib.TskitNameWarning`): use ``gig.iedges``
        """
        warn_tskit_name("edges()", "gig.iedges")
        return self.iedges

    def iedges_for_parent(self, u, *, chromosome=None):
        """
        Return a sequence-like view of the :class:`~giglib.graph.IEdge` objects with parent ``u``,
        ordered by parent chromosome then position. If ``chromosome`` is given, only
        return iedges for that chromosome in the parent. If ``u`` is not a parent
        (or has no iedges on that chromosome), the returned view is empty.
        """
        u = as_id(u)
        ranges = self._id_range_for_parent.get(u, {})
        if chromosome is None:
            if len(ranges) == 0:
                start = end = 0
            else:
                # chromosome ranges are adjacent and in increasing chromosome order
                start = next(iter(ranges.values()))[0]
                end = next(reversed(ranges.values()))[1]
        else:
            start, end = ranges.get(chromosome, (0, 0))
        return GIGItemIterator(self.tables.iedges, IEdge, ids=self._iedge_map_sorted_by_parent[start:end])

    def iedges_for_child(self, u, *, chromosome=None):
        """
        Return a sequence-like view of the :class:`~giglib.graph.IEdge` objects with child ``u``,
        ordered by child chromosome then left position. If ``chromosome`` is given,
        only return iedges for that chromosome in the child. To get iedge IDs instead,
        use :meth:`tables.IEdgeTable.ids_for_child`.
        """
        u = as_id(u)
        return GIGItemIterator(
            self.tables.iedges, IEdge, ids=self.tables.iedges.ids_for_child(u, chromosome=chromosome)
        )

    def chromosomes_as_child(self, u):
        """
        Return the chromosome IDs of node ``u`` that it inherits from its parents
        (i.e. those with iedges above ``u``), as a tuple in ascending order. See
        :meth:`tables.IEdgeTable.chromosomes_as_child`.
        """
        u = as_id(u)
        return self.tables.iedges.chromosomes_as_child(u)

    def chromosomes_as_parent(self, u):
        """
        Return the chromosome IDs of node ``u`` that it passes on to its children
        (i.e. those with iedges below ``u``), as a tuple in ascending order. This is
        needed e.g. for roots, which have no iedges above them.
        """
        u = as_id(u)
        return tuple(sorted(self._id_range_for_parent.get(u, {})))

    def chromosomes(self, u):
        """
        Return the chromosome IDs of node ``u`` that are traced by any iedge above
        or below it, as a tuple in ascending order.
        """
        u = as_id(u)
        return tuple(sorted(set(self.chromosomes_as_child(u)) | set(self.chromosomes_as_parent(u))))

    def max_pos_as_child(self, u, *, chromosome=None):
        u = as_id(u)
        return self.tables.iedges.max_pos_as_child(u, chromosome=chromosome)

    def max_pos_as_parent(self, u, *, chromosome=None):
        u = as_id(u)
        if u not in self._id_range_for_parent:
            return None
        if chromosome is None:
            return max(self.max_pos_as_parent(u, chromosome=c) for c in self._id_range_for_parent[u].keys())
        else:
            if chromosome not in self._id_range_for_parent[u]:
                return None
            idx = self._id_range_for_parent[u][chromosome][1] - 1  # index of last iedge
            return self.tables.iedges[self._iedge_map_sorted_by_parent[idx]].parent_max

    def max_pos(self, u, *, chromosome=None):
        """
        Return the maximum position in the edges above or below this node.
        This defines the known sequence length of the node u for a given
        chromosome (or the maximum position in any chromosome if not specified)
        """
        u = as_id(u)
        max_child = self.max_pos_as_child(u, chromosome=chromosome)
        max_parent = self.max_pos_as_parent(u, chromosome=chromosome)
        if max_parent is None and max_child is None:
            return None
        if max_parent is None:
            return max_child
        if max_child is None:
            return max_parent
        return max(max_child, max_parent)

    def min_pos_as_child(self, u, *, chromosome=None):
        u = as_id(u)
        return self.tables.iedges.min_pos_as_child(u, chromosome=chromosome)

    # We don't yet define min_pos_as_parent or therefore min_pos, because
    # _iedge_map_sorted_by_parent is not sorted by lowest position, so
    # we would have to iterate through all the edges for this parent

    def sequence_length(self, u, *, chromosome=0):
        """
        Return the known sequence length of the given chromosome (default 0) of
        node u: equivalent to ``max_pos(u, chromosome=chromosome)`` but returns 0
        rather than None if this is an isolated node (with no associated edges).
        See :meth:`total_sequence_length` for the length over all chromosomes.
        """
        u = as_id(u)
        return self.max_pos(u, chromosome=chromosome) or 0

    def total_sequence_length(self, u):
        """
        Return the sum of the known sequence lengths of node ``u`` over all
        its chromosomes (0 if the node is isolated)
        """
        u = as_id(u)
        return sum(self.sequence_length(u, chromosome=c) for c in self.chromosomes(u))

    def to_tree_sequence(self, sequence_length=None):
        """
        Convert this GIG to a tree sequence. This can only be done if
        each iedge has the same child_left as parent_left and the same
        child_right as parent_right, and all iedges and mutations are on the
        same chromosome. This is mainly intended for testing.

        Mutations are grouped into sites by position. Since a GIG stores ancestral
        states as mutations above roots (see :ref:`sec_design_mutations`), the first
        mutation above each root of the local tree (excluding isolated nodes) at a
        site gives that site's ancestral state. This can only be done if all the roots
        at the site have such a mutation, and they all agree. All other mutations
        become mutations in the tree sequence, and the mutations at a site must
        either all have known times, or all have unknown times.

        :param int sequence_length: The sequence length in the resulting
            tree sequence (if ``None``, take the maximum position of any edge).
        :return: A tree sequence
        :rtype: tskit.TreeSequence
        :raises ValueError: If the GIG cannot be represented as a tree sequence

        """
        for ie in self.iedges:
            if ie.child_left != ie.parent_left or ie.child_right != ie.parent_right:
                raise ValueError(
                    f"Cannot convert to tree sequence: iedge {ie.id}: child "
                    f"and parent intervals are not the same in all iedges ({ie})"
                )
        iedges = self.tables.iedges
        chromosomes = np.unique(
            np.concatenate((iedges.child_chromosome, iedges.parent_chromosome, self.tables.mutations.chromosome))
        )
        if len(chromosomes) > 1:
            raise ValueError("Cannot convert to tree sequence: iedges and mutations must all be on one chromosome")
        if sequence_length is None:
            sequence_length = self.tables.iedges.child_right.max()
        tables = tskit.TableCollection(sequence_length)
        tables.time_units = self.time_units
        int32_info = np.iinfo(np.int32)
        int32_min = int32_info.min
        int32_max = int32_info.max
        for node in self.nodes:
            if node.individual < int32_min or node.individual > int32_max:
                raise ValueError(f"Cannot store individual IDs > {int32_max} in tskit")
            tables.nodes.add_row(
                time=node.time,
                flags=node.flags,
                individual=np.int32(node.individual),  # can safely cast
            )
        for ie in self.iedges:
            tables.edges.add_row(
                left=ie.child_left,
                right=ie.child_right,
                child=ie.child,
                parent=ie.parent,
            )
        for individual in self.individuals:
            for p in individual.parents:
                if p < int32_min or p > int32_max:
                    raise ValueError(f"Cannot store individual IDs > {int32_max} in tskit")
            tables.individuals.add_row(parents=np.int32(individual.parents))  # can cast

        tables.provenances.add_row(record=json.dumps({"parameters": {"command": "gig.to_tree_sequence"}}))
        tables.sort()
        if len(self.mutations) > 0:
            self._add_tskit_mutations(tables)
        return tables.tree_sequence()

    def _add_tskit_mutations(self, tables):
        """
        Add sites and mutations to a tskit TableCollection, which already contains
        the nodes and edges of this GIG (see :meth:`to_tree_sequence`)
        """
        mutations = self.tables.mutations
        if mutations.position.max() >= tables.sequence_length:
            raise ValueError("Mutation positions must be less than the sequence length")
        by_position = {}
        for m in self.mutations:  # in order of occurrence, for any one node and position
            by_position.setdefault(int(m.position), []).append(m)
        ts = tables.tree_sequence()
        tree = ts.first()
        for position in sorted(by_position):
            tree.seek(position)
            roots = {u for u in tree.roots if tree.num_children(u) > 0}
            ancestral_state = None
            seen_roots = set()
            other_mutations = []
            for m in by_position[position]:
                if m.node in roots and m.node not in seen_roots:
                    seen_roots.add(m.node)
                    if ancestral_state is None:
                        ancestral_state = m.derived_state
                    elif m.derived_state != ancestral_state:
                        raise ValueError(
                            f"Cannot convert to a tree sequence: the roots at position {position} "
                            "have different ancestral states"
                        )
                else:
                    other_mutations.append(m)
            if seen_roots != roots:
                raise ValueError(
                    f"Cannot convert to a tree sequence: at position {position}, not all roots "
                    "have a mutation giving an ancestral state"
                )
            known = [m.is_time_known() for m in other_mutations]
            if any(known) and not all(known):
                raise ValueError(
                    f"Cannot convert to a tree sequence: the mutations at position {position} "
                    "have a mixture of known and unknown times"
                )
            site = tables.sites.add_row(position=position, ancestral_state=ancestral_state)
            # tskit requires older mutations first at a site
            other_mutations.sort(key=lambda m: -m.time if m.is_time_known() else 0)
            for m in other_mutations:
                tables.mutations.add_row(
                    site=site,
                    node=m.node,
                    derived_state=m.derived_state,
                    time=m.time if m.is_time_known() else tskit.UNKNOWN_TIME,
                    metadata=m.metadata if m.metadata is not None else b"",
                )
        tables.sort()
        tables.build_index()
        tables.compute_mutation_parents()

    def decapitate(self, time, *, discard_old_mutations=False):
        """
        Return a new GIG with all nodes at least as old as ``time`` removed. See
        :meth:`Tables.decapitate` for details, including how mutations are treated.

        :param float time: The cutoff time
        :param bool discard_old_mutations: If False (default), record the alleles
            produced by old mutations using new mutations, so that the haplotypes of the
            remaining nodes are unchanged. If True, simply discard old mutations. See
            :meth:`Tables.decapitate`.
        :return: A new GIG object
        :rtype: Graph
        """
        tables = self.tables.copy()
        tables.decapitate(time, discard_old_mutations=discard_old_mutations)
        return self.__class__(tables)

    def sample_resolve(self):
        """
        Sample resolve a GIG, keeping only those edge regions which
        transmit information to the current samples. This is rather
        like running the Hudson algorithm on a fixed graph, but without
        counting up the number of samples under each node. This is
        identical to the :meth:`Tables.sample_resolve` method, but returns
        a new GIG instead.

        :return: A new GIG object
        :rtype: Graph
        """
        new_tables = self.tables.copy()
        new_tables.sample_resolve()
        return new_tables.graph()

    def find_mrca_regions(self, u, v, **kwargs):
        """
        A wrapper around :meth:`Tables.find_mrca_regions`. Unlike the tables
        method, ``u`` and ``v`` can be either node IDs or
        :class:`~giglib.graph.Node` objects.
        """
        return self.tables.find_mrca_regions(as_id(u), as_id(v), **kwargs)


class GIGItemIterator:
    """
    A sequence-like view of items in a GIG that are represented as table rows,
    such as nodes, iedges, and individuals, optionally restricted to a subset of
    IDs. Supports ``len()``, indexing, and iteration, returning objects (rather than
    IDs). Since it has a ``__len__`` method, it plays nicely with progressbar
    output from tqdm (e.g. ``for ie in tqdm.tqdm(gig.iedges): ...``)
    """

    def __init__(self, table, cls, ids=None):
        # If ids is None, the view covers all rows in the table
        self._ids = None if ids is None else _read_only(np.asarray(ids))
        self.table = table
        self.cls = cls

    @property
    def ids(self):
        """
        The IDs of the items in this view, as a read-only numpy array
        """
        if self._ids is None:
            self._ids = _read_only(np.arange(len(self.table), dtype=np.int64))
        return self._ids

    def __getitem__(self, index):
        if self._ids is None:
            # The index is the ID, so must be normalised and bounds-checked here:
            # otherwise numpy would wrap a too-negative index round to the end.
            n = len(self.table)
            i = index + n if index < 0 else index
            if not 0 <= i < n:
                raise IndexError(f"index {index} is out of range for {n} items")
            i = int(i)  # e.g. if a numpy integer was used as the index
            return self.cls(*self.table[i], i)
        else:
            true_id = int(self._ids[index])
            return self.cls(*self.table[true_id], true_id)

    def __len__(self):
        if self._ids is None:
            return len(self.table)
        else:
            return len(self._ids)

    def __iter__(self):
        if self._ids is None:
            for i in range(len(self.table)):
                yield self.cls(*self.table[i], i)
        else:
            for i in self._ids.tolist():  # as Python ints
                yield self.cls(*self.table[i], i)


class IEdge(_IEdgeMethods, namedtuple("IEdge", (*IEdgeTableRow._fields, "id"))):
    """
    A single interval edge in a Graph. This has the same fields and properties as an
    :class:`~giglib.tables.IEdgeTableRow`, and also has an ID. Since the iedges in a
    graph are guaranteed to be valid (e.g. the child and parent intervals have the
    same length), it also has properties and methods that rely on validity, such as
    :attr:`span` and :meth:`transform_position`, which are not available on table rows.
    """

    @property
    def span(self):
        """
        The length of the inherited interval, which is the same in the child and the
        parent (i.e. ``child_right - child_left``, or ``abs(parent_span)``)
        """
        return self.child_right - self.child_left

    def transform_position(self, x, direction):
        """
        Transform a position from child coordinates to parent coordinates
        (``direction=Const.ROOTWARDS``), or from parent to child coordinates
        (``direction=Const.LEAFWARDS``). If this is an inversion then an edge such as
        [0, 10) -> (10, 0] means that position 9 gets transformed to 0, and position
        10 is not transformed at all. This means subtracting 1 from the returned
        (transformed) position.
        """
        if direction == Const.ROOTWARDS:
            if x < self.child_left or x >= self.child_right:
                raise ValueError(f"Position {x} not in child interval for {self}")
            if self.is_inversion():
                return self.child_left - x + self.parent_left - 1
            else:
                return x - self.child_left + self.parent_left

        elif direction == Const.LEAFWARDS:
            if self.is_inversion():
                if x < self.parent_right or x >= self.parent_left:
                    raise ValueError(f"Position {x} not in parent interval for {self}")
                return self.child_left - x + self.parent_left - 1
            else:
                if x < self.parent_left or x >= self.parent_right:
                    raise ValueError(f"Position {x} not in parent interval for {self}")
                return x - self.parent_left + self.child_left
        raise ValueError(f"Direction must be Const.ROOTWARDS or Const.LEAFWARDS, not {direction}")


class Node(_NodeMethods, namedtuple("Node", (*NodeTableRow._fields, "id"))):
    """
    An object representing a single node in a :class:`Graph`. This acts like an
    :class:`~giglib.tables.NodeTableRow` but also has an ID.
    """


class Mutation(_MutationMethods, namedtuple("Mutation", (*MutationTableRow._fields, "id"))):
    """
    An object representing a single mutation in a :class:`Graph`. This acts like a
    :class:`~giglib.tables.MutationTableRow` but also has an ID.
    """


class Individual(namedtuple("Individual", (*IndividualTableRow._fields, "id"))):
    """
    An object representing a single individual in a :class:`Graph`. This acts like an
    :class:`~giglib.tables.IndividualTableRow` but also has an ID.
    """

    pass
