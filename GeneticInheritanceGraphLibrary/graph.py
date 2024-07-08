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
            raise ValueError("tables must be a GeneticInheritanceGraphLibrary.Tables object")
        self.tables = tables
        self._validate()
        self.tables.freeze()

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
        # Cache
        self.sample_ids = self.tables.sample_ids

        # TODO: should cache node spans here,
        # see https://github.com/hyanwong/GeneticInheritanceGraphLibrary/issues/27

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
    def num_nodes(self):
        # Deprecated
        return len(self.tables.nodes)

    @property
    def num_iedges(self):
        # Deprecated
        return len(self.tables.iedges)

    @property
    def num_samples(self):
        # Deprecated
        return len(self.sample_ids)

    @property
    def nodes(self):
        """
        Return an object used to iterate over the nodes in this GIG
        """

        return GIGItemIterator(self.tables.nodes, Node)

    def samples(self):
        """
        Iterate over the sample nodes, returning
        """
        return GIGItemIterator(self.tables.nodes, Node, ids=self.sample_ids)

    @property
    def individuals(self):
        """
        Return an object used to iterate over the individuals in this GIG
        """
        return GIGItemIterator(self.tables.individuals, Individual)

    @property
    def iedges(self):
        """
        Return an object used to iterate over the iedges in this GIG
        """
        return GIGItemIterator(self.tables.iedges, IEdge)

    def iedges_for_parent(self, u, chromosome=None):
        """
        Iterate over all iedges with parent u
        """
        if chromosome is None:
            start = next(self._id_range_for_parent[u].keys())[0]
            end = next(reversed(self._id_range_for_parent[u].keys()))[1]
        else:
            start, end = self._id_range_for_parent[u][chromosome]
        for i in range(start, end):
            yield self.iedges[self._iedge_map_sorted_by_parent[i]]

    def iedges_for_child(self, u, chromosome=None):
        """
        Iterate over all iedges with child u
        """
        for i in self.tables.iedges.ids_for_child(u, chromosome):
            yield self.iedges[i]

    def chromosomes_as_child(self, u):
        return self.tables.iedges.chromosomes_as_child(u)

    def chromosomes_as_parent(self, u):
        return self._id_range_for_parent.get(u, {}).keys()

    def chromosomes(self, u):
        return self.chromosomes_as_child(u) | self.chromosomes_as_parent(u)

    def max_pos_as_child(self, u, chromosome=None):
        return self.tables.iedges.max_pos_as_child(u, chromosome=chromosome)

    def max_pos_as_parent(self, u, chromosome=None):
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
        return self.max_pos(u, chromosome) or 0

    def total_sequence_length(self, u):
        """
        Return the sum of all the sequence lengths in all the chromosomes
        """
        iedges = self.tables.iedges
        chroms = self._id_range_for_parent.get(u, {}).keys() | iedges._id_range_for_child.get(u, {}).keys()
        if len(chroms) > 0:
            return max(self.sequence_length(u, c) for c in chroms)
        return 0

    def to_tree_sequence(self, sequence_length=None):
        """
        Convert this GIG to a tree sequence. This can only be done if
        each iedge has the same child_left as parent_left and the same
        child_right as parent_right.

        If sequence_length is not None, it will be used as the sequence length,
        otherwise the sequence length will be the maximum position of any edge.
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
        identical to the :meht:`Tables.sample_resolve` method, but returns
        a new GIG instead.
        """
        new_tables = self.tables.copy()
        new_tables.sample_resolve()
        return new_tables.graph()

    def find_mrca_regions(self, *args, **kwargs):
        """
        A wrapper around the find_mrca_regions method in the tables object
        """
        return self.tables.find_mrca_regions(*args, **kwargs)


class GIGItemIterator:
    """
    Class to help iterate over items in a GIG. In paticular this is focussed on
    items that are represented as table rows, such as nodes, iedges, and . Since it has a
    __len__ method, it should play nicely with showing progressbar
    output with tqdm (e.g. for i in tqdm.tqdm(gig.iedges): ...)
    """

    def __init__(self, table, cls, ids=None):
        self.ids = ids
        self.table = table
        self.cls = cls

    def __getitem__(self, index):
        if self.ids is None:
            return self.cls(**self.table[index]._asdict(), id=index)
        else:
            true_id = self.ids[index]
            return self.cls(**self.table[true_id]._asdict(), id=true_id)

    def __len__(self):
        if self.ids is None:
            return len(self.table)
        else:
            return len(self.ids)

    def __iter__(self):
        if self.ids is None:
            for i in range(len(self.table)):
                yield self.cls(**self.table[i]._asdict(), id=i)
        else:
            for i in self.ids:
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
    :class:`~tables.NodeTableRow` but also has an ID.
    """

    def is_sample(self):
        return self.flags & Const.NODE_IS_SAMPLE


class Individual(namedtuple("Individual", (*IndividualTableRow._fields, "id"))):
    """
    An object representing a single individual in a :class:`Graph`. This acts like an
    :class:`~tables.IndividualTableRow` but also has an ID.
    """

    pass
