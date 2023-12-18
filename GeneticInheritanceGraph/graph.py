"""
Define a generalised (genetic) inheritance graph object, which is
a validated set of GIG tables with some extra indexes etc for
efficient access.
"""
import dataclasses

import numpy as np
import portion as P
from sortedcontainers import SortedDict

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

        # Cached variables
        self.parent_range = -np.ones((self.num_nodes, 2), dtype=np.int32)
        self.child_range = -np.ones((self.num_nodes, 2), dtype=np.int32)

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

        # Check that parent left is always < parent right
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
        self.iedge_map_sorted_by_child = np.lexsort(
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

    def find_mrca_regions(self, u, v, time_cutoff=None):
        """
        Find all regions between nodes u and v that share a most recent
        common ancestor in the GIG which is more recent than time_cutoff.

        Returns a dict of intervals, each of which is a region between u and v

        Implementation-wise, this is similar to the sample_resolve algorithm, but

        1. Instead of following the ancestry of *all* samples upwards, we
           follow just two of them. This means we are unlikely to visit all
           nodes in the graph.
        2. We do not edit/change the edge intervals: instead we return
           the intervals shared between the two samples.
        3. We keep a separate stack for the ancestral intervals for each
           of these samples, We check which of the stacks has the youngest
           node at the end, and pop only that one off. With the additional
           constraint that nodes of equal time are listed on the stack by
           node ID (i.e. ID is used to break ties), this ensures that we
           can detect when the same node is at the top of the two stacks,
           indicating a common ancestor (in this case we pop from both stacks)
        4. We do not add older regions to the stack if they have coalesced.
        5. We have a time cutoff, so that we don't have to go all the way
           back to the "origin of life"
        """
        # This is quite like the sample_resolve algorithm, only we can stop
        # once we have found a match between two regions, and we don't
        # need to go all the way back to the root, if a time_cutoff is
        # specified. The complexity comes from having to track the original
        # coordinates for each interval. For this we use a portion.IntervalDict.
        shared_intervals = {}
        if u == v:
            L = self.sequence_length(u)
            shared_intervals[u] = [
                P.IntervalDict({P.closedopen(0, L): OriginalInterval(0)})
            ] * 2
            return shared_intervals

        iedges = self.iedges
        stack = BinaryIntervalStack(self, u, v)
        node_times = self.tables.nodes.time
        for idx, node_id, ivls in stack.reduce():
            if time_cutoff is not None and node_times[node_id] > time_cutoff:
                break
            if idx == BinaryIntervalStack.SHARED:
                # common ancestor found: we can save the shared intervals
                shared = ivls[0].domain() & ivls[1].domain()
                shared_intervals[node_id] = []
                track_ancestry = {}
                for i in BinaryIntervalStack.BOTH:
                    mrca = ivls[i].copy()
                    del ivls[i][shared]
                    del mrca[~shared]
                    shared_intervals[node_id].append(mrca)
                    track_ancestry[i] = ivls[i]
            else:
                track_ancestry = {idx: ivls}
            for stackindex, intervals in track_ancestry.items():
                for j in range(*self.parent_range[node_id]):
                    ie = iedges[self.iedge_map_sorted_by_child[j]]
                    for i, orig in intervals.items():
                        if ie.child_right > i.lower and i.upper > ie.child_left:
                            inter_left = max(ie.child_left, i.lower)
                            inter_right = min(ie.child_right, i.upper)
                            parent_left = ie.transform_to_parent(
                                inter_left, is_interval=True
                            )
                            parent_right = ie.transform_to_parent(
                                inter_right, is_interval=True
                            )
                            origin = OriginalInterval(
                                ie.transform_to_parent(orig.map_left_to_original),
                                not orig.inverted
                                if ie.is_inversion()
                                else orig.inverted,
                            )
                            interval = P.closedopen(parent_left, parent_right)
                            stack.add(stackindex, ie.parent, interval, origin)
        return shared_intervals


@dataclasses.dataclass
class OriginalInterval:
    """
    A class to track coordinates to map back to the original interval
    map_left_to_original maps from the current left position back to
    equivalent position in the sample's original coordinates.
    """

    map_left_to_original: int
    inverted: bool = False


class BinaryIntervalStack:
    """
    A class encapsulating a pair of SortedDict stacks. Each item is keyed by node ID
    with values containing an IntervalDict from the portion library. The IntervalDict
    associates each interval with an OriginalInterval object, to track the
    original coordinates from which it has been transformed.

    Each stack is sorted by node time (youngest last) with ties broken by node ID.
    Popping an item off the end of this class will grab the interval
    corresponding to the youngest node on either stack.
    """

    A = 0  # constant, to be used as an alias
    B = 1  # constant, to be used as an alias
    BOTH = [A, B]
    SHARED = 2  # just an indicator variable, to show this node is from both stacks

    def cmp(self, k):
        """
        The order to sort the stacks by: place the lowest times at the end
        (so we can pop them off easily), and make sure ties are broken by node ID
        (so we will simultaneously pop off identical node IDs if they are on both stacks)
        """
        return (-self.nodes_time[k], -k)

    def __init__(self, gig, a, b):
        self.gig = gig
        self.nodes_time = gig.tables.nodes.time
        self.stack = [
            SortedDict(
                self.cmp,
                {u: P.IntervalDict({P.closedopen(0, np.inf): OriginalInterval(0)})},
            )
            for u in (a, b)
        ]
        self.stackA = self.stack[self.A]  # handy alias
        self.stackB = self.stack[self.B]  # handy alias

    def __len__(self):
        return len(self.stackA) + len(self.stackB)

    def reduce(self):
        """
        Return a generator that, when iterated over, reduces
        the size of the stacks by popping off the item with the
        smallest node time in either stack, breaking ties using the
        node ID. Each iteration returns a tuple of stack index
        (A=0 or B=1, or SHARED), node ID, and the relevant
        interval(s).

        If the top of both stacks points to the same node,
        then this could be a common ancestor node, SHARED is
        returned as the index, and a pair of IntervalDicts is
        returned, one for each stack.
        """
        popA = popB = True
        while True:
            try:
                if popA:
                    lastA, intervalsA = self.stackA.popitem()
                if popB:
                    lastB, intervalsB = self.stackB.popitem()
            except KeyError:
                break
            popA = popB = True
            if lastA == lastB:
                yield self.SHARED, lastA, (intervalsA, intervalsB)
            elif self.nodes_time[lastA] < self.nodes_time[lastB]:
                popB = False
                yield self.A, lastA, intervalsA
            elif self.nodes_time[lastA] > self.nodes_time[lastB]:
                popA = False
                yield self.B, lastB, intervalsB
            elif lastA < lastB:
                popB = False
                yield self.A, lastA, intervalsA
            else:
                popA = False
                yield self.B, lastB, intervalsB

    def add(self, stackindex, node, interval, origin):
        stack = self.stack[stackindex]
        if node not in stack:
            stack[node] = P.IntervalDict()
        stack[node][interval] = origin


class Items:
    """
    Class to wrap all the items in a table
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
