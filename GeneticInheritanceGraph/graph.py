"""
Define a generalised (genetic) inheritance graph object, which is
a validated set of GIG tables with some extra indexes etc for
efficient access.
"""
import dataclasses

import numpy as np
import portion as P

from .tables import IEdgeTableRow
from .tables import NodeTableRow
from .tables import Tables


def from_tree_sequence(ts):
    """
    Construct a GIG from a tree sequence
    """
    return Graph(Tables.from_tree_sequence(ts))


class Graph:
    def __init__(self, tables):
        """
        Create a gig.Graph from a set of Tables. This is for internal use only:
        the canonical way to do this is to use tables.Graph()
        """
        if not isinstance(tables, Tables):
            raise ValueError("tables must be a GeneticInheritanceGraph.Tables object")
        self.tables = tables
        self._validate()

    def _validate(self):
        assert hasattr(self, "tables")
        # Extra validation for a GIG here, e.g. edge transformations are
        # valid, etc.

        # Cached variables. We define these up top, as all validated GIGs
        # should have them, even if they have no edges
        self.parent_range = -np.ones((self.num_nodes, 2), dtype=np.int32)
        self.child_range = -np.ones((self.num_nodes, 2), dtype=np.int32)
        self.iedge_map_sorted_by_child = np.arange(
            self.num_iedges
        )  # to overwrite later

        if len(self.tables.iedges) == 0:
            return

        iedges_child = self.tables.iedges.child
        iedges_parent = self.tables.iedges.parent
        # Check that all node IDs are non-negative and less than num_nodes
        for nm, nodes in (("child", iedges_child), ("parent", iedges_parent)):
            if nodes.min() < 0:
                raise ValueError(f"Iedge {np.argmin(nodes)} contains negative {nm} ID.")
            if nodes.max() >= self.num_nodes:
                raise ValueError(
                    f"Iedge {np.argmax(nodes)} contains {nm} ID >= num nodes"
                )

        # check all parents are strictly older than their children
        node_times = self.tables.nodes.time
        for i, ie in enumerate(self.iedges):
            if node_times[ie.parent] <= node_times[ie.child]:
                raise ValueError(
                    f"Edge {i}: parent node {ie.parent} not older than child {ie.child}"
                )

        # Check that all iedges have same absolute parent span as child span
        parent_spans = self.tables.iedges.parent_right - self.tables.iedges.parent_left
        child_spans = self.tables.iedges.child_right - self.tables.iedges.child_left
        span_equal = np.abs(parent_spans) == np.abs(child_spans)
        if not np.all(span_equal):
            bad = np.where(~span_equal)[0]
            raise ValueError(f"iedges {bad} have different parent and child spans")

        # Check that child left is always < child right
        if np.any(child_spans <= 0):
            raise ValueError(
                f"child_left >= child_right for iedges {np.where(child_spans < 0)[0]}"
            )

        # Check iedges are sorted so that all parent IDs are adjacent
        iedges_parent = self.tables.iedges.parent
        tot_parents = len(np.unique(iedges_parent))
        if np.sum(np.diff(iedges_parent) != 0) != tot_parents - 1:
            raise ValueError("iedges are not sorted by parent ID")

        # Check iedges are also sorted by parent time
        if np.any(np.diff(self.tables.nodes.time[iedges_parent]) < 0):
            raise ValueError("iedges are not sorted by parent time")

        # TODO: Check within a parent, edges are sorted by child ID

        # CREATE CACHED VARIABLES

        # Should probably cache more stuff here, e.g. list of samples

        # index to sort edges so they are sorted by child time
        # and grouped by child id. Within each group with the same child ID,
        # the edges are sorted by child_left
        # See https://github.com/tskit-dev/tskit/discussions/2869
        self.iedge_map_sorted_by_child[:] = np.lexsort(
            (
                self.tables.iedges.parent,
                self.tables.iedges.child_right,
                self.tables.iedges.child_left,
                self.tables.iedges.child,
                self.tables.nodes.time[self.tables.iedges.child],  # Primary key
            )
        )
        last_child = -1
        for j, idx in enumerate(self.iedge_map_sorted_by_child):
            ie = self.iedges[idx]
            if ie.child != last_child:
                self.parent_range[ie.child, 0] = j
            if last_child != -1:
                self.parent_range[last_child, 1] = j
            last_child = ie.child
        if last_child != -1:
            self.parent_range[last_child, 1] = self.num_iedges

        last_parent = -1
        for ie in self.iedges:
            if ie.parent != last_parent:
                self.child_range[ie.parent, 0] = ie.id
            if last_parent != -1:
                self.child_range[last_parent, 1] = ie.id
            last_parent = ie.parent
        if last_parent != -1:
            self.child_range[last_parent, 1] = self.num_iedges

        # Check every location has only one parent
        for u in range(self.num_nodes):
            prev_right = -np.inf
            for ie in sorted(self.iedges_for_child(u), key=lambda x: x.child_left):
                if ie.child_left < prev_right:
                    raise ValueError(
                        f"Node {u} has multiple or duplicate parents at position"
                        f" {ie.child_left}"
                    )
                prev_right = ie.child_right

    @property
    def num_nodes(self):
        return len(self.tables.nodes)

    @property
    def num_iedges(self):
        return len(self.tables.iedges)

    def samples(self):
        # TODO - we should really cache this
        return self.tables.samples()

    @property
    def num_samples(self):
        # TODO - we should really cache this
        return len(self.tables.samples())

    @property
    def nodes(self):
        return Items(self.tables.nodes, Node)

    @property
    def iedges(self):
        return Items(self.tables.iedges, IEdge)

    def iedges_for_child(self, u):
        """
        Iterate over all iedges with child u
        """
        for i in range(*self.parent_range[u, :]):
            yield self.iedges[self.iedge_map_sorted_by_child[i]]

    def iedges_for_parent(self, u):
        """
        Iterate over all iedges with parent u
        """
        for i in range(*self.child_range[u, :]):
            yield self.iedges[i]

    def sequence_length(self, u):
        """
        Return the known sequence length of the node u (found from the
        maximum position in the edges above or below this node)
        """
        maxpos = [ie.parent_max for ie in self.iedges_for_parent(u)]
        maxpos += [ie.child_max for ie in self.iedges_for_child(u)]
        if len(maxpos) > 0:
            return max(maxpos)
        return 0

    def sample_resolve(self):
        """
        Sample resolve a GIG, keeping only those edge regions which
        transmit information to the current samples. This is rather
        like running the Hudson algorithm on a fixed graph, but without
        counting up the number of samples under each node.

        The algorithm is implemented by using a stack that contains intervals
        for each node, ordered by node time (oldest first). When considering a
        node, we pop the (youngest) node off the end of the stack, which
        ensures that we have collect all the intervals with that node as a
        parent, before passing inheritance information upwards
        """

        # NB - for simplicity we put all nodes onto the stack at the start,
        # and fill out their intervals as we go. An alternative
        # would be to create a dynamic stack ordered by node time, e.g.
        # stack = SortedDict(lambda k: -self.tables.nodes.time[k])
        stack = {u: P.empty() for u in np.argsort(-self.tables.nodes.time)}
        new_tables = self.tables.copy()
        new_tables.iedges.clear()
        iedges = self.iedges
        while len(stack) > 0:
            u, intervals = stack.popitem()
            if self.nodes[u].is_sample():
                intervals = P.closedopen(0, np.inf)

            # NOTE: if we had an index here sorted by left coord
            # we could binary search to first match, and could
            # break once e_right > left (I think?)
            for j in range(*self.parent_range[u]):
                ie = iedges[self.iedge_map_sorted_by_child[j]]
                assert ie.child == u
                for i in intervals:
                    if ie.child_right > i.lower and i.upper > ie.child_left:
                        inter_left = max(ie.child_left, i.lower)
                        inter_right = min(ie.child_right, i.upper)
                        parent_left = ie.transform_to_parent(
                            inter_left, is_interval=True
                        )
                        parent_right = ie.transform_to_parent(
                            inter_right, is_interval=True
                        )
                        if ie.is_inversion():
                            stack[ie.parent] |= P.closedopen(parent_right, parent_left)
                        else:
                            stack[ie.parent] |= P.closedopen(parent_left, parent_right)
                        new_tables.iedges.add_row(
                            parent=ie.parent,
                            child=ie.child,
                            child_left=inter_left,
                            child_right=inter_right,
                            parent_left=parent_left,
                            parent_right=parent_right,
                        )
        new_tables.sort()
        return self.__class__(new_tables)


class Items:
    """
    Class to wrap all the items in a table. Since it has a
    __len__ method, it should play nicely with showing progressbar
    output with tqdm (e.g. for i in tqdm.tqdm(gig.iedges): ...)
    """

    def __init__(self, table, cls):
        self.table = table
        self.cls = cls

    def __getitem__(self, index):
        return self.cls(**self.table[index].asdict(), id=index)

    def __len__(self):
        return len(self.table)

    def __iter__(self):
        for i in range(len(self.table)):
            yield self.cls(**self.table[i].asdict(), id=i)


@dataclasses.dataclass(frozen=True, kw_only=True)
class IEdge(IEdgeTableRow):
    """
    A single interval edge in a Graph. Similar to an edge table row
    but with an ID and various other useful methods that rely on
    the graph being a consistent GIG (e.g. that abs(parent_span) == abs(child_span))
    """

    id: int  # NOQA: A003

    @property
    def span(self):
        return self.child_span if self.child_span >= 0 else self.parent_span

    @property
    def parent_max(self):
        """
        The highest parent position on this edge (inversions can have left > right)
        """
        return max(self.parent_right, self.parent_left)

    @property
    def child_max(self):
        """
        The highest child position on this edge
        """
        return self.child_right

    def is_inversion(self):
        return self.child_span < 0 or self.parent_span < 0

    def transform_to_parent(self, child_position, is_interval=False):
        """
        Transform a position from child coordinates to parent coordinates.
        If this is an inversion and we are transforming a point position,
        then an edge such as [0, 10) -> (10, 0] means that position 9 gets
        transformed to 0, and position 10 is not transformed at all. This
        means subtracting 1 from the returned (transformed) position.

        If, on the other hand, we are transforming an interval, then we
        don't want to subtract the one.
        """
        if child_position < self.child_left or child_position > self.child_right:
            raise ValueError(
                f"Position {child_position} is not in the child interval for {self}"
            )
        if self.is_inversion():
            if is_interval:
                return (self.child_right - child_position + self.child_left) + (
                    self.child_right - self.parent_left
                )
            else:
                return (self.child_right - child_position + self.child_left - 1) + (
                    self.child_right - self.parent_left
                )
        else:
            return child_position - self.child_left + self.parent_left


@dataclasses.dataclass(frozen=True, kw_only=True)
class Node(NodeTableRow):
    """
    A single node in a Graph. Similar to an node table row but with an ID.
    """

    id: int  # NOQA: A003
