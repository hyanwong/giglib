"""
Define a generalised (genetic) inheritance graph object, which is
a validated set of GIG tables with some extra indexes etc for
efficient access.
"""

import json
from collections import namedtuple

import numpy as np
import tskit

from .constants import Const, ValidFlags
from .tables import IEdgeTableRow, IndividualTableRow, NodeTableRow, Tables
from .util import as_id, warn_tskit_name


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
                for ie in self.iedges_for_child(u, chromosome):
                    if ie.child_left < prev_right:
                        raise ValueError(f"Node {u} has multiple or duplicate parents at position" f" {ie.child_left}")
                    prev_right = ie.child_right
        self.tables.iedges.flags = ValidFlags.GIG

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

    def individual(self, i):
        """
        Not part of the giglib API (emits a :class:`~giglib.TskitNameWarning`): use ``gig.individuals[i]``
        """
        warn_tskit_name("individual(i)", "gig.individuals[i]")
        return self.individuals[i]

    def iedges_for_parent(self, u, chromosome=None):
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

    def iedges_for_child(self, u, chromosome=None):
        """
        Return a sequence-like view of the :class:`~giglib.graph.IEdge` objects with child ``u``,
        ordered by child chromosome then left position. If ``chromosome`` is given,
        only return iedges for that chromosome in the child. To get iedge IDs instead,
        use :meth:`tables.IEdgeTable.ids_for_child`.
        """
        u = as_id(u)
        return GIGItemIterator(self.tables.iedges, IEdge, ids=self.tables.iedges.ids_for_child(u, chromosome))

    def chromosomes_as_child(self, u):
        u = as_id(u)
        return self.tables.iedges.chromosomes_as_child(u)

    def chromosomes_as_parent(self, u):
        u = as_id(u)
        return self._id_range_for_parent.get(u, {}).keys()

    def chromosomes(self, u):
        u = as_id(u)
        return self.chromosomes_as_child(u) | self.chromosomes_as_parent(u)

    def max_pos_as_child(self, u, chromosome=None):
        u = as_id(u)
        return self.tables.iedges.max_pos_as_child(u, chromosome=chromosome)

    def max_pos_as_parent(self, u, chromosome=None):
        u = as_id(u)
        if u not in self._id_range_for_parent:
            return None
        if chromosome is None:
            return max(self.max_pos_as_parent(u, c) for c in self._id_range_for_parent[u].keys())
        else:
            if chromosome not in self._id_range_for_parent[u]:
                return None
            idx = self._id_range_for_parent[u][chromosome][1] - 1  # index of last iedge
            return self.tables.iedges[self._iedge_map_sorted_by_parent[idx]].parent_max

    def max_pos(self, u, chromosome=None):
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

    def min_pos_as_child(self, u, chromosome=None):
        u = as_id(u)
        return self.tables.iedges.min_pos_as_child(u, chromosome=chromosome)

    # We don't yet define min_pos_as_parent or therefore min_pos, because
    # _iedge_map_sorted_by_parent is not sorted by lowest position, so
    # we would have to iterate through all the edges for this parent

    def sequence_length(self, u, chromosome):
        """
        Return the known sequence length of the node u: equivalent to
        max_position(u) but returns 0 rather than None if this is an
        isolated node (with no associated edges).
        """
        u = as_id(u)
        return self.max_pos(u, chromosome) or 0

    def total_sequence_length(self, u):
        """
        Return the sum of the known sequence lengths of node ``u`` over all
        its chromosomes (0 if the node is isolated)
        """
        u = as_id(u)
        return sum(self.sequence_length(u, c) for c in self.chromosomes(u))

    def to_tree_sequence(self, sequence_length=None):
        """
        Convert this GIG to a tree sequence. This can only be done if
        each iedge has the same child_left as parent_left and the same
        child_right as parent_right.

        :param int sequence_length: The sequence length in the resulting
            tree sequence (if ``None``, take the maximum position of any edge).
        :return: A tree sequence
        :rtype: tskit.TreeSequence

        """
        for ie in self.iedges:
            if ie.child_left != ie.parent_left or ie.child_right != ie.parent_right:
                raise ValueError(
                    f"Cannot convert to tree sequence: iedge {ie.id}: child "
                    f"and parent intervals are not the same in all iedges ({ie})"
                )
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
        return tables.tree_sequence()

    def decapitate(self, time):
        """
        Return a new GIG with all nodes older than time removed.

        :param float time: The cutoff time
        :return: A new GIG object
        :rtype: Graph
        """
        tables = self.tables.copy()  # placeholder for making these editable
        tables.decapitate(time)
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
            if index < 0:
                index += len(self.table)
            return self.cls(**self.table[index]._asdict(), id=index)
        else:
            true_id = self._ids[index]
            return self.cls(**self.table[true_id]._asdict(), id=true_id)

    def __len__(self):
        if self._ids is None:
            return len(self.table)
        else:
            return len(self._ids)

    def __iter__(self):
        if self._ids is None:
            for i in range(len(self.table)):
                yield self.cls(**self.table[i]._asdict(), id=i)
        else:
            for i in self._ids:
                yield self.cls(**self.table[i]._asdict(), id=i)


class IEdge(namedtuple("IEdge", (*IEdgeTableRow._fields, "id"))):
    """
    A single interval edge in a Graph. Similar to an edge table row
    but with an ID and various other useful methods that rely on
    the graph being a consistent GIG (e.g. that abs(parent_span) == abs(child_span))
    """

    @property
    def span(self):
        return self.child_right - self.child_left

    @property
    def parent_max(self):
        """
        The highest parent position on this edge (inversions can have left > right)
        """
        return max(self.parent_right, self.parent_left)

    @property
    def parent_min(self):
        """
        The lowest parent position on this edge (inversions can have left > right)
        """
        return min(self.parent_right, self.parent_left)

    @property
    def child_max(self):
        """
        The highest child position on this edge
        """
        return self.child_right

    @property
    def child_min(self):
        """
        The lowest child position on this edge
        """
        return self.child_left

    def is_inversion(self):
        return self.parent_right < self.parent_left

    def is_simple_inversion(self):
        """
        Is this an in-situ inversion, where parent and child coords are simply swapped
        """
        return self.parent_left == self.child_right and self.parent_right == self.child_left

    def transform_position(self, x, direction):
        """
        Transform a position from child coordinates to parent coordinates.
        If this is an inversion then an edge such as [0, 10) -> (10, 0]
        means that position 9 gets transformed to 0, and position 10 is
        not transformed at all. This means subtracting 1 from the returned
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


class Node(namedtuple("Node", (*NodeTableRow._fields, "id"))):
    """
    An object representing a single node in a :class:`Graph`. This acts like an
    :class:`~giglib.tables.NodeTableRow` but also has an ID.
    """

    def is_sample(self):
        return bool(self.flags & Const.NODE_IS_SAMPLE)


class Individual(namedtuple("Individual", (*IndividualTableRow._fields, "id"))):
    """
    An object representing a single individual in a :class:`Graph`. This acts like an
    :class:`~giglib.tables.IndividualTableRow` but also has an ID.
    """

    pass
