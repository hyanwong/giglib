"""
Define a generalised (genetic) inheritance graph object, which is
a validated set of GIG tables with some extra indexes etc for
efficient access.
"""
import dataclasses
import json

import numpy as np
import portion as P
import tskit

from .constants import IEDGES_CHILD_IDS_ADJACENT
from .constants import LEAFWARDS
from .constants import ROOTWARDS
from .constants import VALID_GIG
from .tables import IEdgeTableRow
from .tables import IndividualTableRow
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
        the canonical way to do this is to use tables.graph()
        """
        if not isinstance(tables, Tables):
            raise ValueError(
                "tables must be a GeneticInheritanceGraphLibrary.Tables object"
            )
        self.tables = tables
        self._validate()
        self.tables.freeze()

    def _validate(self):
        assert hasattr(self, "tables")
        # Extra validation for a GIG here, e.g. edge transformations are
        # valid, etc.

        # Cached variables. We define these up top, as all validated GIGs
        # should have them, even if they have no edges
        self.child_range = -np.ones((len(self.nodes), 2), dtype=np.int32)
        self.iedge_map_sorted_by_parent = np.arange(
            len(self.iedges)
        )  # to overwrite later - initialised here in case we return early with 0 edges

        # Cache
        self.samples = self.tables.samples()

        # TODO: should cache node spans here,
        # see https://github.com/hyanwong/GeneticInheritanceGraphLibrary/issues/27

        if len(self.tables.iedges) == 0:
            return

        iedges_child = self.tables.iedges.child
        iedges_parent = self.tables.iedges.parent
        # Check that all node IDs are non-negative and less than #nodes
        for nm, nodes in (("child", iedges_child), ("parent", iedges_parent)):
            if nodes.min() < 0:
                raise ValueError(f"Iedge {np.argmin(nodes)} contains negative {nm} ID.")
            if nodes.max() >= len(self.nodes):
                raise ValueError(
                    f"Iedge {np.argmax(nodes)} contains {nm} ID >= num nodes"
                )

        # check all parents are strictly older than their children
        # (NB: this also allows a time of np.inf for an "ultimate root" node)
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

        # Check that child left is always < child right (also checks nonzero span)
        if np.any(child_spans <= 0):
            raise ValueError(
                f"child_left >= child_right for iedges {np.where(child_spans < 0)[0]}"
            )

        # Check iedges are sorted so that nodes are in decreasing order
        # of child time, with ties broken by child ID, increasing
        timediff = np.diff(self.tables.nodes.time[iedges_child])
        node_id_diff = np.diff(iedges_child)

        if np.any(timediff > 0):
            raise ValueError("iedges are not sorted by child time, descending")

        if np.any(node_id_diff[timediff == 0] < 0):
            raise ValueError("iedges are not sorted by child time (desc) then ID (asc)")
        # must be adjacent
        self.tables.iedges.set_bitflag(IEDGES_CHILD_IDS_ADJACENT)

        # Check within a child, edges are sorted by left coord
        if np.any(np.diff(self.tables.iedges.child_left)[node_id_diff == 0] <= 0):
            raise ValueError("iedges for a given child are not sorted by left coord")

        # CREATE CACHED VARIABLES

        # Should probably cache more stuff here, e.g. list of samples

        # index to sort edges so they are sorted as in tskit: first by parent time
        # then grouped by parent id. Within each group with the same parent ID,
        # the edges are sorted by child_is and then child_left
        self.iedge_map_sorted_by_parent[:] = np.lexsort(
            (
                self.tables.iedges.child_left,
                self.tables.iedges.child,
                self.tables.iedges.parent,
                self.tables.nodes.time[self.tables.iedges.parent],  # Primary key
            )
        )
        last_parent = -1
        for j, idx in enumerate(self.iedge_map_sorted_by_parent):
            ie = self.iedges[idx]
            if ie.parent != last_parent:
                self.child_range[ie.parent, 0] = j
            if last_parent != -1:
                self.child_range[last_parent, 1] = j
            last_parent = ie.parent
        if last_parent != -1:
            self.child_range[last_parent, 1] = len(self.iedges)

        # Check every location has only one parent
        for u in range(len(self.nodes)):
            prev_right = -np.inf
            for ie in self.iedges_for_child(u):
                if ie.child_left < prev_right:
                    raise ValueError(
                        f"Node {u} has multiple or duplicate parents at position"
                        f" {ie.child_left}"
                    )
                prev_right = ie.child_right
        self.tables.iedges.flags = VALID_GIG

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
        return len(self.samples)

    @property
    def nodes(self):
        return Items(self.tables.nodes, Node)

    @property
    def individuals(self):
        return Items(self.tables.individuals, Individual)

    @property
    def iedges(self):
        return Items(self.tables.iedges, IEdge)

    def iedges_for_parent(self, u):
        """
        Iterate over all iedges with child u
        """
        for i in range(*self.child_range[u, :]):
            yield self.iedges[self.iedge_map_sorted_by_parent[i]]

    def iedges_for_child(self, u):
        """
        Iterate over all iedges with parent u
        """
        for i in self.tables.iedges.ids_for_child(u):
            yield self.iedges[i]

    def max_position(self, u):
        """
        Return the maximum position in the edges above or below this node.
        This defines the known sequence length of the node u.
        """
        maxpos = [ie.parent_max for ie in self.iedges_for_parent(u)]
        maxpos += [ie.child_max for ie in self.iedges_for_child(u)]
        if len(maxpos) > 0:
            return max(maxpos)
        return None

    def min_position(self, u):
        """
        Return the minimum position in the edges above or below this node.
        Genetic data for positions before this is treated as unknown.
        """
        minpos = [ie.parent_min for ie in self.iedges_for_parent(u)]
        minpos += [ie.child_min for ie in self.iedges_for_child(u)]
        if len(minpos) > 0:
            return min(minpos)
        return None

    def sequence_length(self, u):
        """
        Return the known sequence length of the node u: equivalent to
        max_position(u) but returns 0 rather than None if this is an
        isolated node (with no associated edges).
        """
        return self.max_position(u) or 0

    def to_tree_sequence(self, sequence_length=None):
        """
        convert this GIG to a tree sequence. This can only be done if
        each iedge has the same child_left as parent_left and the same
        child_right as parent_right.

        If sequence_length is not None, it will be used as the sequence length,
        otherwise the sequence length will be the maximum position of any edge.
        """
        if np.any(
            self.tables.iedges.child_left != self.tables.iedges.parent_left
        ) or np.any(self.tables.iedges.child_right != self.tables.iedges.parent_right):
            raise ValueError(
                "Cannot convert to tree sequence: "
                "child and parent intervals are not the same in all iedges"
            )
        if sequence_length is None:
            sequence_length = self.tables.iedges.child_right.max()
        tables = tskit.TableCollection(sequence_length)
        tables.time_units = self.time_units
        for node in self.nodes:
            tables.nodes.add_row(
                time=node.time, flags=node.flags, individual=node.individual
            )
        for ie in self.iedges:
            tables.edges.add_row(
                left=ie.child_left,
                right=ie.child_right,
                child=ie.child,
                parent=ie.parent,
            )
        for individual in self.individuals:
            tables.individuals.add_row(parents=individual.parents)

        tables.provenances.add_row(
            record=json.dumps({"parameters": {"command": "gig.to_tree_sequence"}})
        )
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
        counting up the number of samples under each node.

        The algorithm is implemented by using a stack that contains intervals
        for each node, ordered by node time (oldest first). When considering a
        node, we pop the (youngest) node off the end of the stack, which
        ensures that we have collected all the intervals with that node as a
        parent, before passing inheritance information upwards
        """
        # == Implementation details ==
        # A. Rather than representing the intervals for a node as a list of
        #    (left, right) tuples, the "portion" library is used, allowing multiple
        #    intervals to be combined into a single object and automatically merged
        #    together when they adjoin or overlap (coalesce). This reduces the
        #    number of intervals and their fragmentation as we go up the stack.
        # B. For simplicity we put all nodes onto the stack at the start, in time
        #    order (relying on the fact that python dicts now preserve order). We
        #    then fill out their intervals as we go, and gradually remove the items
        #    from the stack as we work through it. Since sample_resolve() will touch
        #    almost all nodes, declaring and filling the stack up-front is efficient.
        #    An alternative if only a few nodes are likely to be visited would
        #    be to create a dynamic stack continuously sorted by node time, e.g.
        #        stack = sortedcontainers.SortedDict(lambda k: -nodes_time[k])
        # C. This differs slightly from the similar tskit method
        #    `.simplify(filter_nodes=False, keep_unary=True)`, because it does not
        #    squash adjacent edges together. It should therefore preserve
        #    "recombination diamonds". This could be useful for some applications,
        stack = {c: P.empty() for c in np.argsort(-self.tables.nodes.time)}
        new_tables = self.tables.copy(omit_iedges=True)
        while len(stack) > 0:
            child, intervals = stack.popitem()
            if self.nodes[child].is_sample():
                intervals = P.closedopen(0, np.inf)
            for ie_id in self.tables.iedges.ids_for_child(child):
                ie = self.tables.iedges[ie_id]
                parent = ie.parent
                child_ivl = P.closedopen(ie.child_left, ie.child_right)
                is_inversion = ie.is_inversion()
                # JEROME'S NOTE (from previous algorithm): if we had an index here
                # sorted by left coord we could binary search to first match, and
                # could break once c_right > left (I think?), rather than having
                # to intersect all the intervals with the current c_rgt/c_lft
                for interval in intervals:
                    if c := child_ivl & interval:  # intersect
                        p = self.tables.iedges.transform_interval(
                            ie_id, (c.lower, c.upper), ROOTWARDS
                        )
                        if is_inversion:
                            # For passing upwards, interval orientation doesn't matter
                            # so put the lowest position first, as `portion` requires
                            stack[parent] |= P.closedopen(p[1], p[0])
                        else:
                            stack[parent] |= P.closedopen(*p)
                        # Add the trimmed interval to the new edge table
                        new_tables.iedges.add_row(
                            c.lower, c.upper, *p, child=child, parent=parent
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
        return self.parent_span < 0

    def is_simple_inversion(self):
        """
        Is this an in-situ inversion, where parent and child coords are simply swapped
        """
        return (
            self.parent_left == self.child_right
            and self.parent_right == self.child_left
        )

    def transform_position(self, x, direction):
        """
        Transform a position from child coordinates to parent coordinates.
        If this is an inversion then an edge such as [0, 10) -> (10, 0]
        means that position 9 gets transformed to 0, and position 10 is
        not transformed at all. This means subtracting 1 from the returned
        (transformed) position.
        """

        if direction == ROOTWARDS:
            if x < self.child_left or x >= self.child_right:
                raise ValueError(f"Position {x} not in child interval for {self}")
            if self.is_inversion():
                return self.child_left - x + self.parent_left - 1
            else:
                return x - self.child_left + self.parent_left

        elif direction == LEAFWARDS:
            if self.is_inversion():
                if x < self.parent_right or x >= self.parent_left:
                    raise ValueError(f"Position {x} not in parent interval for {self}")
                return self.child_left - x + self.parent_left - 1
            else:
                if x < self.parent_left or x >= self.parent_right:
                    raise ValueError(f"Position {x} not in parent interval for {self}")
                return x - self.parent_left + self.child_left
        raise ValueError(f"Direction must be ROOTWARDS or LEAFWARDS, not {direction}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Node(NodeTableRow):
    """
    A single node in a Graph. Similar to an node table row but with an ID.
    """

    id: int  # NOQA: A003


@dataclasses.dataclass(frozen=True, kw_only=True)
class Individual(IndividualTableRow):
    """
    A single individual in a Graph. Similar to an individual table row but with an ID.
    """

    id: int  # NOQA: A003
