"""
Define a generalised (genetic) inheritance graph object, which is
a validated set of GIG tables with some extra indexes etc for
efficient access.
"""
import collections
import dataclasses

import numpy as np
import portion as P
import sortedcontainers

from .constants import LEAFWARDS
from .constants import ROOTWARDS
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
        the canonical way to do this is to use tables.graph()
        """
        if not isinstance(tables, Tables):
            raise ValueError(
                "tables must be a GeneticInheritanceGraphLibrary.Tables object"
            )
        self.tables = tables
        self._validate()

    def _validate(self):
        assert hasattr(self, "tables")
        # Extra validation for a GIG here, e.g. edge transformations are
        # valid, etc.

        # Cached variables. We define these up top, as all validated GIGs
        # should have them, even if they have no edges
        self.parent_range = -np.ones((len(self.nodes), 2), dtype=np.int32)
        self.child_range = -np.ones((len(self.nodes), 2), dtype=np.int32)
        self.iedge_map_sorted_by_child = np.arange(
            len(self.iedges)
        )  # to overwrite later

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
            self.parent_range[last_child, 1] = len(self.iedges)

        last_parent = -1
        for ie in self.iedges:
            if ie.parent != last_parent:
                self.child_range[ie.parent, 0] = ie.id
            if last_parent != -1:
                self.child_range[last_parent, 1] = ie.id
            last_parent = ie.parent
        if last_parent != -1:
            self.child_range[last_parent, 1] = len(self.iedges)

        # Check every location has only one parent
        for u in range(len(self.nodes)):
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

    @staticmethod  # undocumented: internal use
    def intersect(lft_1, rgt_1, lft_2, rgt_2):
        # Return the intersection of two intervals, or None if they do not overlap
        if rgt_1 > lft_2 and rgt_2 > lft_1:
            return max(lft_1, lft_2), min(rgt_1, rgt_2)
        return None

    @staticmethod  # undocumented: internal use
    def general_intersect(a, b, c, d):
        # Return the intersection of two intervals, allowing for
        # a > b and/or c > d
        if a < b:
            return Graph.intersect(a, b, c, d) if c < d else Graph.intersect(a, b, d, c)
        else:
            return Graph.intersect(b, a, c, d) if c < d else Graph.intersect(b, a, d, c)

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
        tables = self.tables.copy()
        tables.iedges.clear()  # clear old iedges: trimmed ones will be added back
        while len(stack) > 0:
            c, intervals = stack.popitem()  # the node `c` is treated as the child
            if self.nodes[c].is_sample():
                intervals = P.closedopen(0, np.inf)
            for ie in self.iedges_for_child(c):
                p, c_lft, c_rgt = ie.parent, ie.child_left, ie.child_right  # cache
                # JEROME'S NOTE (from previous algorithm): if we had an index here
                # sorted by left coord we could binary search to first match, and
                # could break once c_right > left (I think?), rather than having
                # to intersect all the intervals with the current c_rgt/c_lft
                for i in intervals:
                    if child_ivl := self.intersect(c_lft, c_rgt, i.lower, i.upper):
                        parnt_ivl = ie.transform_interval(child_ivl, ROOTWARDS)
                        if ie.is_inversion():
                            # For passing upwards, interval orientation doesn't matter
                            # so put the lowest position first, as `portion` requires
                            stack[p] |= P.closedopen(parnt_ivl[1], parnt_ivl[0])
                        else:
                            stack[p] |= P.closedopen(*parnt_ivl)
                        # Add the trimmed interval to the new edge table
                        tables.iedges.add_row(*child_ivl, *parnt_ivl, parent=p, child=c)
        tables.sort()
        return self.__class__(tables)

    def find_mrca_regions(self, u, v, time_cutoff=None, as_interval_dict=False):
        """
        Find all regions between nodes u and v that share a most recent
        common ancestor in the GIG which is more recent than time_cutoff.
        If as_interval_dict is True, returns a dict of the following form:
            {
                MRCA_node_idA : portion.IntervalDict,
                MRCA_node_idB : portion.IntervalDict,
                ...
            }
        Where the IntervalDicts are keyed by interval k = portion.closedopen(kA, kB)
        covering the MRCA node, and whose values contain
        a tuple of two lists of intervals stored as python tuples e.g. (uA, uB).
        The first list contains the interval(s) in u that correspond to k, and
        the second contains the interval(s) in v that
        correspond to k. If there has been no duplication of the interval since
        the MRCA, then the lists will only contain a single interval each.

        If as_interval_dict is False, the same structure is returned but the
        portion.IntervalDict objects are replaced by normal dictionaries,
        so that as well as the intervals within intervals corresponding to
        u and v being tuples, the key k is a standard (kA, kB) python tuple.


        Implementation-wise, this is similar to the sample_resolve algorithm, but
        1. Instead of following the ancestry of *all* samples upwards, we
           follow just two of them. This means we are unlikely to visit all
           nodes in the graph, and so we create a dynamic stack (using the
           sortedcontainers.SortedDict class, which is kept ordered by node time.
        2. We do not edit/change the edge intervals as we go. Instead we simply
           return the intervals shared between the two samples.
        3. The intervals "know" whether they were originally associated with u or v.
           If we find an ancestor with both u and v-associated intervals
           this indicates a potential common ancestor, in which coalescence could
           have occurred.
        4. We do not add older regions to the stack if they have coalesced.
        5. We have a time cutoff, so that we don't have to go all the way
           back to the "origin of life"
        """
        if not isinstance(u, (int, np.integer)) or not isinstance(v, (int, np.integer)):
            raise ValueError("u and v must be integers")

        if u == v:
            raise ValueError("u and v must be different nodes")

        def trim(lft_1, rgt_1, lft_2, rgt_2):
            """
            An inversion-aware intersection routine, which returns the intersection
            of two intervals, where the left and right of
            the second interval could be swapped. If they are swapped, return the
            interval in the reversed order. Return None if they do not overlap
            """
            if rgt_2 < lft_2:  # an inversion
                if rgt_1 > rgt_2 and lft_2 > lft_1:
                    return min(rgt_1, lft_2), max(lft_1, rgt_2)
            else:
                if rgt_1 > lft_2 and rgt_2 > lft_1:
                    return max(lft_1, lft_2), min(rgt_1, rgt_2)
            return None

        if time_cutoff is None:
            time_cutoff = np.inf
        # Use a dynamic stack as we hopefully will be visiting a minority of nodes
        node_times = self.tables.nodes.time
        stack = sortedcontainers.SortedDict(lambda k: -node_times[k])

        # Don't use the Portion interval library for *transmitting* intervals, as it
        # can't be used to keep track of the orientation of each interval (required
        # to take account of inversions). Instead represent intervals as (a, b)
        # tuples, where a < b if the interval is in the orientation of the the original
        # u or v genome. We also extend the tuple to include another value
        # x, which tracks the position of the origin in the original genome. This allows
        # us to map a position in the current interval back to a position in u or v.
        #
        # We store *two* lists of (a, b, x) tuples, the first for ancestral regions of
        # u, and the second for ancestral regions of v: if a node has something in both
        # lists, intervals for u and v co-occur (and could therefore coalesce), meaning
        # the node is a potential MRCA.
        U, V = 0, 1  # temp constants. The 1st list (index 0) is for U, the second for V
        stack[u] = ([], [])  # intervals to track for node u (fill on next line)
        stack[u][U].append((0, np.inf, 0))  # whole genome, origin at 0
        stack[v] = ([], [])  # intervals to track for node v (fill on next line)
        stack[v][V].append((0, np.inf, 0))  # whole genome, origin at 0

        result = collections.defaultdict(P.IntervalDict)

        def concat(x, y):
            return (x[U] | y[U], x[V] | y[V])

        while len(stack) > 0:
            c, uv_intervals = stack.popitem()  # node `c` = child
            if node_times[c] > time_cutoff:
                return result
            print(f"Checking child {c}")
            if len(uv_intervals[0]) > 0 and len(uv_intervals[1]) > 0:
                print(f"Potential coalescence in {c}")
                print(f" u: {uv_intervals[U]}")
                print(f" v: {uv_intervals[V]}")
                # check for overlap between uv_intervals[0] and uv_intervals[1]
                # which results in coalescence. The coalescent point can be
                # recorded, and the overlap deleted from the intervals
                #
                # Intervals are not currently sorted, so we need to check
                # all combinations. Note that we could have duplicate positions
                # even in the same interval list, due to segmental duplications
                # within a genome.
                coalesced = P.empty()
                for i, (uL, uR, _) in enumerate(uv_intervals[U]):
                    for j, (vL, vR, _) in enumerate(uv_intervals[V]):
                        if intersection := self.general_intersect(uL, uR, vL, vR):
                            mrca = P.closedopen(*intersection)  # MRCA as a Portion obj
                            coalesced |= mrca
                            result[c] = result[c].combine(
                                P.IntervalDict({mrca: ({i}, {j})}), how=concat
                            )
                if not coalesced.empty:
                    # Work out the mapping of the mrca intervals into intervals in
                    # u and v, given indexes into the uv_intervals lists.
                    for mrca, uv_indexes in result[c].items():
                        mapped_intervals = ([], [])
                        for a, indexes, intervals in zip(
                            mapped_intervals, uv_indexes, uv_intervals
                        ):
                            for index in indexes:
                                l, r, offset = intervals[index]
                                if l < r:
                                    a.append((offset + mrca.lower, offset + mrca.upper))
                                else:
                                    # Original inverted relative to the mrca interval
                                    a.append((offset + mrca.upper, offset + mrca.lower))
                        result[c][mrca] = mapped_intervals
                    # Remove the coalesced segments from the interval lists
                    print(f"Condensed coalescences in {c}: {coalesced}")
                    for u_or_v in (U, V):
                        intervals = [[], []]
                        for ivl in uv_intervals[u_or_v]:
                            if ivl[0] < ivl[1]:
                                for i in P.closedopen(ivl[0], ivl[1]) - coalesced:
                                    intervals[u_or_v].append((i.lower, i.upper, ivl[2]))
                            else:
                                for i in P.closedopen(ivl[1], ivl[0]) - coalesced:
                                    intervals[u_or_v].append((i.upper, i.lower, ivl[2]))
                    uv_intervals = intervals
            for ie in self.iedges_for_child(c):
                # JEROME'S NOTE (from previous algorithm): if we had an index here
                # sorted by left coord we could binary search to first match, and
                # could break once c_right > left (I think?), rather than having
                # to intersect all the intervals with the ie.child_left/right
                for u_or_v, intervals in enumerate(uv_intervals):
                    for i in intervals:
                        if child_ivl := trim(ie.child_left, ie.child_right, i[0], i[1]):
                            parnt_ivl = ie.transform_interval(child_ivl, ROOTWARDS)
                            # The offset of the current coordinate system
                            x = i[2] + child_ivl[0] - parnt_ivl[0]
                            if ie.parent not in stack:
                                stack[ie.parent] = ([], [])
                            print(
                                f"Added to {('u', 'v')[u_or_v]} stack for {ie.parent}: "
                                + str(parnt_ivl)
                            )
                            stack[ie.parent][u_or_v].append((*parnt_ivl, x))
        if as_interval_dict:
            return result
        return {
            k: {(k.lower, k.upper): v for k, v in pv.items()}
            for k, pv in result.items()
        }


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

    def transform_interval(self, interval, direction):
        """
        Transform an interval from child coordinates to parent coordinates.
        If this is an inversion then an edge such as [0, 10] -> [10, 0] means
        that a nested interval such as [1, 8] gets transformed to [9, 2].
        """
        for x in interval:
            if x < self.child_left or x > self.child_right:
                raise ValueError(f"Position {x} not in child interval for {self}")

        if direction == ROOTWARDS:
            if self.is_inversion():
                return tuple(self.child_left - x + self.parent_left for x in interval)
            else:
                return tuple(x - self.child_left + self.parent_left for x in interval)
        raise ValueError(f"Direction must be ROOTWARDS, not {direction}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Node(NodeTableRow):
    """
    A single node in a Graph. Similar to an node table row but with an ID.
    """

    id: int  # NOQA: A003
