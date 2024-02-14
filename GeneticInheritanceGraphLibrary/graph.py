"""
Define a generalised (genetic) inheritance graph object, which is
a validated set of GIG tables with some extra indexes etc for
efficient access.
"""
import collections
import dataclasses
import json
import logging

import numpy as np
import portion as P
import sortedcontainers
import tskit

from .constants import LEAFWARDS
from .constants import ROOTWARDS
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

    def _validate(self):
        assert hasattr(self, "tables")
        # Extra validation for a GIG here, e.g. edge transformations are
        # valid, etc.

        # Cached variables. We define these up top, as all validated GIGs
        # should have them, even if they have no edges
        self.parent_range = -np.ones((len(self.nodes), 2), dtype=np.int32)
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

        last_child = -1
        for ie in self.iedges:
            if ie.child != last_child:
                self.parent_range[ie.child, 0] = ie.id
            if last_child != -1:
                self.parent_range[last_child, 1] = ie.id
            last_child = ie.child
        if last_child != -1:
            self.parent_range[last_child, 1] = len(self.iedges)

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

    @property
    def time_units(self):
        return self.tables.time_units

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
        for i in range(*self.parent_range[u, :]):
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
            child, intervals = stack.popitem()
            if self.nodes[child].is_sample():
                intervals = P.closedopen(0, np.inf)
            for ie in self.iedges_for_child(child):
                parent = ie.parent
                child_ivl = P.closedopen(ie.child_left, ie.child_right)
                is_inversion = ie.is_inversion()
                # JEROME'S NOTE (from previous algorithm): if we had an index here
                # sorted by left coord we could binary search to first match, and
                # could break once c_right > left (I think?), rather than having
                # to intersect all the intervals with the current c_rgt/c_lft
                for interval in intervals:
                    if c := child_ivl & interval:  # intersect
                        p = ie.transform_interval((c.lower, c.upper), ROOTWARDS)
                        if is_inversion:
                            # For passing upwards, interval orientation doesn't matter
                            # so put the lowest position first, as `portion` requires
                            stack[parent] |= P.closedopen(p[1], p[0])
                        else:
                            stack[parent] |= P.closedopen(*p)
                        # Add the trimmed interval to the new edge table
                        tables.iedges.add_row(
                            c.lower, c.upper, *p, child=child, parent=parent
                        )
        tables.sort()
        return self.__class__(tables)

    def find_mrca_regions(self, u, v, time_cutoff=None):
        """
        Find all regions between nodes u and v that share a most recent
        common ancestor in the GIG which is more recent than time_cutoff.
        Returns a dict of dicts of the following form
            {
                MRCA_node_ID1 : {(X, Y): (
                    [(uA, uB), (uC, uD), ...],
                    [(vA, vB), ...]
                )},
                MRCA_node_ID2 : ...,
                ...
            }
        Where in each inner dict, the key (X, Y) gives an interval (with X < Y)
        in the MRCA node, and the value is a 2-tuple giving the corresponding
        intervals in u and v. In the example above there is a list of two
        corresponding intervals in u: (uA, uB) and (uC, uD) representing a
        duplication of the MRCA interval into u. If uA > uB then that interval
        in u is inverted relative to that in the MRCA node.

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
        6. We have to keep track of the offset of the current interval into the
           original genome, to allow mapping back to the original coordinates
        """
        if not isinstance(u, (int, np.integer)) or not isinstance(v, (int, np.integer)):
            raise ValueError("u and v must be integers")

        if u == v:
            raise ValueError("u and v must be different nodes")

        if time_cutoff is None:
            time_cutoff = np.inf
        # Use a dynamic stack as we hopefully will be visiting a minority of nodes
        node_times = self.tables.nodes.time
        stack = sortedcontainers.SortedDict(lambda k: -node_times[k])

        # We need to track intervals separately if they have been transformed
        # (inverted or coordinate-shifted). Within the stack, for each node we
        # therefore store a *dictionary* of portion objects, keyed by offset and
        # orientation.
        #
        # We store *two* such {(offset, orientation): intervals} dicts. The first
        # tracks the ancestral regions of u, and the second the ancestral regions of v.
        # If a node has something in both dicts, intervals for u and v co-occur
        # (and could therefore coalesce), meaning the node is a potential MRCA.
        offset = 0
        # tracked intervals for node u, keyed by offset+orientation (False=not inverted)
        stack[u] = ({(offset, False): P.closedopen(0, np.inf)}, {})
        # tracked intervals for node v, keyed by offset+orientation (False=not inverted)
        stack[v] = ({}, {(offset, False): P.closedopen(0, np.inf)})

        result = collections.defaultdict(P.IntervalDict)

        def concat(x, y):
            # combine a pair of sets in x and a pair of sets in y
            return (x[0] | y[0], x[1] | y[1])

        logging.debug(f"Checking mrca of {u} and {v}: {stack}")
        while len(stack) > 0:
            child, (u_dict, v_dict) = stack.popitem()  # node `c` = child
            if node_times[child] > time_cutoff:
                return result
            logging.debug(f"Checking child {child}")
            if len(u_dict) > 0 and len(v_dict) > 0:
                logging.debug(f"Potential coalescence in {child}")
                logging.debug(f" u: {u_dict}")
                logging.debug(f" v: {v_dict}")
                # check for overlap between all intervals in u_dict and v_dict
                # which result in coalescence. The coalescent point can be
                # recorded, and the overlap deleted from the intervals
                #
                # Intervals are not currently sorted, so we need to check
                # all combinations. Note that we could have duplicate positions
                # even in the same interval list, due to segmental duplications
                # within a genome. We have to use the portion library here to
                # cut the MRCA regions into non-overlapping pieces which have different
                # patterns of descent into u and v
                coalesced = P.empty()
                for u_key, u_intervals in u_dict.items():
                    for i, u_interval in enumerate(u_intervals):
                        for v_key, v_intervals in v_dict.items():
                            for j, v_interval in enumerate(v_intervals):
                                if mrca := u_interval & v_interval:
                                    coalesced |= mrca
                                    details = ({(u_key, i)}, {(v_key, j)})
                                    result[child] = result[child].combine(
                                        P.IntervalDict({mrca: details}), how=concat
                                    )
                if not coalesced.empty:
                    # Work out the mapping of the mrca intervals into intervals in
                    # u and v, given keys into the uv_intervals dicts.
                    for mrca, uv_details in result[child].items():
                        to_store = ([], [])
                        for s, details in zip(to_store, uv_details):
                            # Odd that we don't use the interval dict here: not sure why
                            for key, _ in details:
                                offset, inverted_relative_to_original = key
                                if inverted_relative_to_original:
                                    s.append((offset - mrca.lower, offset - mrca.upper))
                                else:
                                    s.append((mrca.lower - offset, mrca.upper - offset))
                        result[child][mrca] = to_store  # replace

                    # Remove the coalesced segments from the interval lists
                    for uv_dict in (u_dict, v_dict):
                        for key in uv_dict.keys():
                            uv_dict[key] -= coalesced

            for ie in self.iedges_for_child(child):
                # See note in sample resolve algorithm re efficiency and
                # indexing by left coord. Note this is more complex in the
                # mrca case because we have to treat intervals with different
                # offsets / orientations separately
                parent = ie.parent
                child_ivl = P.closedopen(ie.child_left, ie.child_right)  # cache
                is_inversion = ie.is_inversion()
                for u_or_v, interval_dict in enumerate((u_dict, v_dict)):
                    for (offset, already_inverted), intervals in interval_dict.items():
                        for interval in intervals:
                            if c := child_ivl & interval:
                                p = ie.transform_interval((c.lower, c.upper), ROOTWARDS)
                                if is_inversion:
                                    if already_inverted:
                                        # 0 gets flipped backwards
                                        x = offset - (c.lower + p[0])
                                    else:
                                        x = offset + (c.lower + p[0])
                                    already_inverted = not already_inverted
                                    parent_ivl = P.closedopen(p[1], p[0])
                                else:
                                    x = offset - c.lower + p[0]
                                    parent_ivl = P.closedopen(*p)
                                key = (x, already_inverted)
                                if parent not in stack:
                                    stack[parent] = ({}, {})
                                if key in stack[parent][u_or_v]:
                                    stack[parent][u_or_v][key] |= parent_ivl
                                else:
                                    stack[parent][u_or_v][key] = parent_ivl
        return {
            k: {(k.lower, k.upper): v for k, v in pv.items()}
            for k, pv in result.items()
        }

    @staticmethod
    def random_matching_positions(mrcas_structure, rng):
        """
        Given a structure returned by the find_mrca_regions method, choose
        a position uniformly at random from the mrca regions and return
        the equivalent position in u and v.

        Returns an equivalent position in u and v (chosen at random
        if there are multiple equivalent positions for u, or multiple equivalent
        positions for v). If this is a negative number, it indicates a position
        reading in the reverse direction from that in the mrca (i.e. there have been
        an odd number of inversions between the mrca and the sample).

        It is hard to know from the MRCA structure whether intervals are
        adjacent, so if this is used to locate a breakpoint, the choice of
        whether a breakpoint is positioned to the left or right of the returned
        position is left to the user.
        """
        tot_len = sum(x[1] - x[0] for v in mrcas_structure.values() for x in v.keys())
        # Pick a single breakpoint
        loc = rng.integers(tot_len)  # breakpoint is before this base
        for mrca_intervals in mrcas_structure.values():
            for x in mrca_intervals.keys():
                if loc < x[1] - x[0]:
                    u, v = mrca_intervals[x]
                    assert len(u) != 0
                    assert len(v) != 0
                    u = u[0] if len(u) == 1 else rng.choice(u)
                    v = v[0] if len(v) == 1 else rng.choice(v)
                    # go the right number of positions into the interval
                    # If inverted, we take the position minus 1, because an
                    # inversion like (0, 10) -> (10, 0) maps pos 0 to pos 9
                    # (not 10). If an inversion, we also negate the position
                    # to indicate reading in the other direction

                    # We should never choose the RH number in the interval,
                    # Because that position is not included in the interval

                    # TODO: check this works if loc is maxed out at u[1]
                    u = u[0] + loc if u[0] < u[1] else -(u[1] - loc - 1)
                    v = v[0] + loc if v[0] < v[1] else -(v[1] - loc - 1)
                    return u, v
                loc -= x[1] - x[0]


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


@dataclasses.dataclass(frozen=True, kw_only=True)
class Individual(IndividualTableRow):
    """
    A single individual in a Graph. Similar to an individual table row but with an ID.
    """

    id: int  # NOQA: A003
