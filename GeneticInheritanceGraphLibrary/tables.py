import collections
import dataclasses
import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
import portion as P
import sortedcontainers
import tskit

from .constants import IEDGES_CHILD_IDS_ADJACENT
from .constants import IEDGES_PARENT_OLDER_THAN_CHILD
from .constants import IEDGES_VALID_INTERVALS
from .constants import NODE_IS_SAMPLE
from .constants import NULL
from .constants import ROOTWARDS
from .constants import VALID_GIG
from .util import truncate_rows


@dataclasses.dataclass(frozen=True)
class TableRow:
    def asdict(self):
        return dataclasses.asdict(self)

    def _validate(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, field.type):
                raise ValueError(
                    f"Expected {field.name} to be {field.type}, " f"got {repr(value)}"
                )


@dataclasses.dataclass(frozen=True)
class IEdgeTableRow(TableRow):
    child_left: int
    child_right: int
    parent_left: int
    parent_right: int
    _: dataclasses.KW_ONLY
    child: int
    parent: int
    edge: int = NULL
    child_chromosome: int = NULL
    parent_chromosome: int = NULL

    @property
    def parent_span(self):
        return self.parent_right - self.parent_left

    @property
    def child_span(self):
        return self.child_right - self.child_left

    def is_inversion(self):
        return self.parent_span * self.child_span < 0

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
        return max(self.child_right, self.child_left)

    @property
    def child_min(self):
        """
        The lowest child position on this edge
        """
        return min(self.child_right, self.child_left)


@dataclasses.dataclass(frozen=True)
class NodeTableRow(TableRow):
    time: float
    flags: int = 0
    individual: int = NULL
    # TODO: add metadata

    def is_sample(self):
        return (self.flags & NODE_IS_SAMPLE) > 0


@dataclasses.dataclass(frozen=True)
class IndividualTableRow(TableRow):
    parents: tuple = ()


class BaseTable:
    RowClass = None
    _frozen = False

    def __getattr__(self, name):
        # Extract column by name. This is a bit of a hack: can be replaced later
        if name not in self.RowClass.__annotations__:
            raise AttributeError
        return np.array([getattr(d, name) for d in self._data])

    def __init__(self):
        # TODO - at the moment we store each row as a separate dataclass object,
        # in the self._data list. This is not very efficient, but it is simple. We
        # will probably want to follow the tskit paradigm of storing such that
        # each column can be accessed as a numpy array too. I'm not sure how to
        # do this without diving into C implementations.
        self._data = []

    def freeze(self):
        """
        Freeze the table so that it cannot be modified
        """
        # turn the data into a tuple so that it is immutable. This doesn't
        # make a copy, but just passes the data by reference
        self._data = tuple(self._data)
        self._frozen = True

    def __setattr__(self, attr, value):
        if self._frozen:
            raise AttributeError("Trying to set attribute on a frozen instance")
        return super().__setattr__(attr, value)

    def copy(self):
        """
        Returns an unfrozen deep copy of this table
        """
        copy = self.__class__()
        copy._data = list(self._data).copy()
        return copy

    def __eq__(self, other):
        return tuple(self._data) == tuple(other._data)

    def clear(self):
        """
        Deletes all rows in this table.
        """
        self._data = []

    def __len__(self):
        return len(self._data)

    def __str__(self):
        headers, rows = self._text_header_and_rows(limit=20)
        unicode = tskit.util.unicode_table(rows, header=headers, row_separator=False)
        # hack to change hardcoded package name
        newstr = []
        linelen_unicode = unicode.find("\n")
        assert linelen_unicode != -1
        for line in unicode.split("\n"):
            if "skipped (tskit" in line:
                line = line.replace("skipped (tskit", f"skipped ({__package__}")
                if len(line) > linelen_unicode:
                    line = line[: linelen_unicode - 1] + line[-1]
            newstr.append(line)
        return "\n".join(newstr)

    def _repr_html_(self):
        """
        Called e.g. by jupyter notebooks to render tables
        """
        from . import _print_options  # pylint: disable=import-outside-toplevel

        headers, rows = self._text_header_and_rows(limit=_print_options["max_lines"])
        html = tskit.util.html_table(rows, header=headers)
        return html.replace(
            "tskit.set_print_options", f"{__package__}.set_print_options"
        )

    def __getitem__(self, index):
        return self._data[index]

    @staticmethod
    def to_dict(obj):
        try:
            return obj.asdict()
        except AttributeError:
            try:
                return dataclasses.asdict(obj)
            except TypeError:
                pass
        return obj

    def add_row(self, *args, **kwargs) -> int:
        """
        Add a row to a BaseTable: the arguments are passed to the RowClass constructor
        (with the RowClass being dataclass).

        :return: The row ID of the newly added row

        Example:
            new_id = table.add_row(dataclass_field1="foo", dataclass_field2="bar")
        """
        self._data.append(self.RowClass(*args, **kwargs))
        return len(self._data) - 1

    def add_rowlist(self, rowlist):
        """
        Add a list of rows to a BaseTable. Each item in the rowlist must contain
        objects of the required table row type. This is a convenience function.
        For a more flexible and performant method that uses numpy arrays
        and broadcasting, some tables may also implement an ``add_rows`` method.
        """
        for row in rowlist:
            self.append(row)

    def append(self, obj) -> int:
        """
        Append a row to a BaseTable by picking required fields from a passed-in object.
        The object can be a dict, a dataclass, or an object with an .asdict() method.

        :return: The row ID of the newly added row

        Example:
            new_id = table.append({"field1": "foo", "field2": "bar"})
        """

        kwargs = self.to_dict(obj)
        return self.add_row(
            **{k: v for k, v in kwargs.items() if k in self.RowClass.__annotations__}
        )

    def _text_header_and_rows(self, limit=None):
        headers = ("id",)
        headers += tuple(k for k in self.RowClass.__annotations__.keys() if k != "_")
        rows = []
        row_indexes = truncate_rows(len(self), limit)
        for j in row_indexes:
            if j == -1:
                rows.append(f"__skipped__{len(self) - limit}")
            else:
                row = self[j]
                rows.append(
                    [str(j)] + [f"{x}" for x in dataclasses.asdict(row).values()]
                )
        return headers, rows

    @staticmethod
    def _check_int(i, k=None):
        if isinstance(i, (int, np.integer)):
            return i
        try:
            if i.is_integer():
                return int(i)
            raise ValueError(f"Expected {k + ' to be ' if k else ''}an integer not {i}")
        except AttributeError:
            raise TypeError(
                f"Could not convert {k + '=' if k else ''}{i} to an integer"
            )

    def _df(self):
        """
        Temporary hack to convert the table to a Pandas dataframe.
        Shouldn't be used for anything besides exploratory work!
        """
        return pd.DataFrame([dataclasses.asdict(row) for row in self._data])


class IEdgeTable(BaseTable):
    """
    A table containing the iedges. This contains more complex logic than other
    GIG tables as we want to be able to run algorithms on an IEdgeTable without
    freezing it into a valid GIG.

    For example:
    * we define an iedges_by_child() method which is only valid if the iedges
      for a given child are all adjacent
    """

    RowClass = IEdgeTableRow

    def __init__(self):
        # save some flags to indicate if this is a valid
        self.flags = VALID_GIG
        # map each child ID to the index of the first edge with that child
        self._id_range_for_child = {}
        super().__init__()

    def copy(self):
        """
        Returns an unfrozen deep copy of this table
        """
        copy = super().copy()
        copy.flags = self.flags
        copy._id_range_for_child = self._id_range_for_child.copy()
        return copy

    def clear(self):
        """
        Deletes all rows in this table.
        """
        self.flags = VALID_GIG
        # map each child ID to the index of the first edge with that child
        self._id_range_for_child = {}
        super().clear()

    def __eq__(self, other):
        if self.flags != other.flags:
            return False
        if self._id_range_for_child != other._id_range_for_child:
            return False
        return super().__eq__(other)

    def set_bitflag(self, flag):
        self.flags |= flag

    def unset_bitflag(self, flag):
        self.flags &= ~flag

    def has_bitflag(self, flag):
        return bool(self.flags & flag)

    def add_row(
        self, *args, validate_child_adjacency=None, validate_intervals=None, **kwargs
    ) -> int:
        """
        Add a row to an IEdgeTable: the arguments are passed to the RowClass constructor
        (with the RowClass being dataclass).

        :param bool validate_child_adjacency: If True, check the child ID is either
            the same as the child_id of the last edge or is a completely new id.
            If False, do not perform this check and assume the user has checked already
            which should be more efficient but could lead to later algorithmic failures.
            If None (default) do not check, and set flags to indicate the tables
            may not be a valid GIG.
        :param bool validate_intervals: If True, check the child left is less than
            child right and the absolute parent span is the same as the child span.
            If False, do not perform this check and assume the user has
            checked already (which should be more efficient but could lead to later
            algorithmic failures). If None (default) do not check, and set flags to
            indicate the tables may not be a valid GIG.
        :return: The row ID of the newly added row

        Example:
            new_id = iedges.add_row(cl, cr, pl, pr, c_id, p_id)
        """
        c = kwargs["child"]
        num_iedges = len(self)
        if validate_child_adjacency:
            if num_iedges > 0 and self[-1].child != c and c in self._id_range_for_child:
                raise ValueError(
                    f"Adding an edge with child ID {c} would make IDs non-adjacent"
                )
        if validate_child_adjacency is None:
            self.unset_bitflag(IEDGES_CHILD_IDS_ADJACENT)
        try:
            self._id_range_for_child[c][1] = num_iedges + 1
        except KeyError:
            self._id_range_for_child[c] = [num_iedges, num_iedges + 1]

        row_id = super().add_row(*args, **kwargs)
        if validate_intervals:
            e = self[-1]
            if e.child_left >= e.child_right or e.child_span != abs(e.parent_span):
                raise ValueError(
                    "Bad edge intervals (e.g. child left must be < child right)"
                )
        if validate_intervals is None:
            self.unset_bitflag(IEDGES_VALID_INTERVALS)

        # we can't validate the age of the parent and child here.
        # So we set it to invalid the revalidate it if called via tables.add_iedge_row.
        self.unset_bitflag(IEDGES_PARENT_OLDER_THAN_CHILD)
        return row_id

    def _from_tskit(self, kwargs):
        new_kw = {}
        for side in ("left", "right"):
            if side in kwargs:
                new_kw["child_" + side] = new_kw["parent_" + side] = kwargs[side]
        new_kw.update(kwargs)
        return new_kw

    def append(self, obj) -> int:
        """
        Append a row to an IEdgeTable taking the named attributes of the passed-in
        object to populate the row. If a `left` field is present in the object,
        is it placed in the ``parent_left`` and ``child_left`` attributes of this
        row (and similarly for a ``right`` field). This allows tskit conversion.

        To enable tskit conversion, the special add_int_row() method is used,
        which is slower than the standard add_row. If you require efficiency,
        you are therefore recommended to call add_row() directly, ensuring that
        the intervals and node IDs are integers before you pass them in.
        """
        kwargs = self._from_tskit(self.to_dict(obj))
        return self.add_int_row(
            **{k: v for k, v in kwargs.items() if k in self.RowClass.__annotations__}
        )

    def add_int_row(self, *args, **kwargs) -> int:
        """
        Add a row to an IEdgeTable, checking that the first 4 positional arguments
        (genome intervals) and six named keyword args can be converted to integers
        (and converting them if required). This allows edges from e.g. tskit
        (which allows floating-point positions) to be passed in.

        :return: The row ID of the newly added row
        """
        # NB: here we override the default method to allow integer conversion and
        # left -> child_left, parent_left etc., to aid conversion from tskit
        # For simplicity, this only applies for named args, not positional ones
        pcols = (
            "child_left",
            "child_right",
            "parent_left",
            "parent_right",
            "child",
            "parent",
        )
        args = [self._check_int(v) if i < 4 else v for i, v in enumerate(args)]
        kwargs = {k: self._check_int(v) if k in pcols else v for k, v in kwargs.items()}
        return self.add_row(*args, **kwargs)

    def ids_for_child(self, u):
        if not self.has_bitflag(IEDGES_CHILD_IDS_ADJACENT):
            raise ValueError(
                "Cannot use this method unless iedges have adjacent child IDs"
            )
        try:
            return np.arange(*self._id_range_for_child[u])
        except KeyError:
            return np.arange(0)

    def transform_interval(self, edge_id, interval, direction):
        """
        Given an edge ID, use that edge to transform the provided interval.
        If this is an inversion then an edge such as [0, 10] -> [10, 0] means
        that a nested interval such as [1, 8] gets transformed to [9, 2].

        :param int direction: Either LEAFWARDS or ROOTWARDS (only ROOTWARDS implemented)
        """
        # Requires edges to be sane
        if not self.has_bitflag(IEDGES_VALID_INTERVALS):
            raise ValueError("Cannot use the method unless iedges have valid intervals")
        e = self[edge_id]
        for x in interval:
            if x < e.child_left or x > e.child_right:
                raise ValueError(f"Position {x} not in child interval for {e}")

        if direction == ROOTWARDS:
            if e.is_inversion():
                return tuple(e.child_left - x + e.parent_left for x in interval)
            else:
                return tuple(x - e.child_left + e.parent_left for x in interval)
        raise ValueError(f"Direction must be ROOTWARDS, not {direction}")

    @property
    def parent(self) -> npt.NDArray[np.int64]:
        """
        Return a numpy array of parent node IDs
        """
        return np.array([row.parent for row in self.data], dtype=np.int64)

    @property
    def child(self) -> npt.NDArray[np.int64]:
        """
        Return a numpy array of child node IDs
        """
        return np.array([row.child for row in self.data], dtype=np.int64)

    @property
    def child_left(self) -> npt.NDArray[np.int64]:
        """
        Return a numpy array of child node IDs
        """
        return np.array([row.child_left for row in self.data], dtype=np.int64)

    @property
    def child_right(self) -> npt.NDArray[np.int64]:
        """
        Return a numpy array of child node IDs
        """
        return np.array([row.child_right for row in self.data], dtype=np.int64)

    @property
    def parent_left(self) -> npt.NDArray[np.int64]:
        """
        Return a numpy array of child node IDs
        """
        return np.array([row.parent_left for row in self.data], dtype=np.int64)

    @property
    def parent_right(self) -> npt.NDArray[np.int64]:
        """
        Return a numpy array of child node IDs
        """
        return np.array([row.parent_right for row in self.data], dtype=np.int64)

    @property
    def edge(self) -> npt.NDArray[np.int64]:
        """
        Return a numpy array of edge IDs
        """
        return np.array([row.edge for row in self.data], dtype=np.int64)


class NodeTable(BaseTable):
    RowClass = NodeTableRow

    def __init__(self):
        # we use the time array repeatedly so cache it.
        self.time = np.array([], dtype=np.float64)
        super().__init__()

    def copy(self):
        """
        Returns an unfrozen deep copy of this table
        """
        copy = super().copy()
        copy.time = self.time.copy()
        return copy

    def clear(self):
        """
        Deletes all rows in this table.
        """
        self.time = np.array([], dtype=np.float64)
        super().clear()

    @property
    def flags(self) -> npt.NDArray[np.uint32]:
        return np.array([row.flags for row in self.data], dtype=np.uint32)

    def add_row(self, *args, **kwargs) -> int:
        """
        Add a row to a NodeTable: the arguments are passed to the RowClass constructor
        (with the RowClass being dataclass).

        :return: The row ID of the newly added row

        Example:
            new_id = nodes.add_row(dataclass_field1="foo", dataclass_field2="bar")
        """
        node_id = super().add_row(*args, **kwargs)
        # inefficient to use "append" to enlarge the numpy array but worth it because
        # we often access the time array even as it is being dynamically created.
        # A more efficient approach would be to pre-allocate an empty array
        # and return a view onto an appropriately-sized slice of that array.
        # The problem can be aleviated a little by using add_rows instead.
        self.time = np.append(self.time, self[-1].time)
        return node_id

    def add_rows(self, time, flags, individual):
        """
        Adds multiple individuals at a time, efficiently.
        All parameters must be provided as numpy arrays which are broadcast to
        the same shape as the largest array. To specify no individual,
        provide ``NULL`` (-1) as the ``individual`` value.

        Returns a numpy array of numpy IDs whose shape is
        given by the shape of the largest input array.
        """
        # This is more efficient than repeated calls to add_row
        # because it can allocate memory in one go. This is currently only
        # used for the cached .time array but could also be used for
        # the row data itself once that is efficiently stored.
        time, flags, individual = np.broadcast_arrays(time, flags, individual)
        self.time = np.append(self.time, time.flatten())
        result = np.empty_like(time)
        for indexes in np.ndindex(result.shape):
            result[indexes] = super().add_row(
                time[indexes], flags[indexes], individual[indexes]
            )
        return result


class IndividualTable(BaseTable):
    RowClass = IndividualTableRow

    def add_rows(self, parents):
        """
        Adds multiple individuals at a time, efficiently.
        Equivalent to iterating over the "parents"
        and adding each in turn. For efficiency, parents
        should be a numpy array of ints of at least 2 dimensions.

        To create individuals without parents you can pass a
        2D array whose second dimension is of length zero, e.g.
        ``np.array([], dtype=int).reshape(num_individuals, 0)``

        Returns a numpy array of individual IDs whose shape is
        given by the shape of the input array minus the last
        dimension (i.e. of shape ``parents_array.shape[:-1]``)
        """
        # NB: currently no more efficient than calling add_row repeatedly
        return np.apply_along_axis(self.add_row, -1, parents)


# Return value for MRCA finder
MRCAintervals = collections.namedtuple("MRCAintervals", "u, v")

# Return value for random_matching_positions
ComparablePositions = collections.namedtuple(
    "ComparablePositions", "u, v, opposite_orientations"
)


class Tables:
    """
    A group of tables describing a Genetic Inheritance Graph (GIG),
    similar to a tskit TableCollection.
    """

    _frozen = False

    table_classes = {
        "nodes": NodeTable,
        "iedges": IEdgeTable,
        "individuals": IndividualTable,
    }

    def __init__(self, time_units=None):
        for name, cls in self.table_classes.items():
            setattr(self, name, cls())
        self.time_units = "unknown" if time_units is None else time_units

    def __eq__(self, other) -> bool:
        for name in self.table_classes.keys():
            if getattr(self, name) != getattr(other, name):
                return False
        if self.time_units != other.time_units:
            return False
        return True

    def add_iedge_row(self, *args, validate_node_times=None, **kwargs):
        """
        Calls the edge.add_row function and also (optionally)
        validate features of the nodes used (e.g. that the parent
        node time is older than the child node time).

        :param bool validate_node_time: If True, check the nodes referred
            to are valid and in the right time-order. If False do not
            check but assume the user has checked already. If None (default)
            do not check, and set flags to indicate the tables may not be a
            valid GIG.
        """
        if validate_node_times:
            try:
                if (
                    self.nodes[kwargs["child"]].time
                    >= self.nodes[kwargs["parent"]].time
                ):
                    raise ValueError("Child time is not less than parent time")
            except IndexError:
                raise ValueError(
                    "Child or parent ID does not correspond to a node in the node table"
                )
        was_valid = self.iedges.has_bitflag(IEDGES_PARENT_OLDER_THAN_CHILD)
        self.iedges.add_row(*args, **kwargs)
        if validate_node_times is not None and was_valid:
            # We have checked or promised to check the rows we added and they are OK
            self.iedges.set_bitflag(IEDGES_PARENT_OLDER_THAN_CHILD)

    def freeze(self):
        """
        Freeze all tables so they cannot be modified
        """
        for name in self.table_classes.keys():
            getattr(self, name).freeze()
        self._frozen = True

    def __setattr__(self, attr, value):
        if self._frozen:
            raise AttributeError("Trying to set attribute on a frozen instance")
        return super().__setattr__(attr, value)

    def clear(self):
        """
        Clear all tables
        """
        for name in self.table_classes.keys():
            getattr(self, name).clear()

    def copy(self, omit_iedges=None):
        """
        Return an unfrozen deep copy of the tables. If omit_iedges is True
        do not copy the iedges table but use a blank one
        """
        copy = self.__class__()
        for name in self.table_classes.keys():
            if omit_iedges and name == "iedges":
                setattr(copy, name, self.table_classes[name]())
            else:
                setattr(copy, name, getattr(self, name).copy())
        copy.time_units = self.time_units
        return copy

    def __str__(self):
        # To do: make this look nicer
        return "\n\n".join(
            [
                "== NODES ==\n" + str(self.nodes),
                "== I-EDGES ==\n" + str(self.iedges),
            ]
        )

    def sort(self):
        """
        Sort the tables in place. Currently only affects the iedges table. Sorting
        criteria match those used in tskit (see
        https://tskit.dev/tskit/docs/stable/data-model.html#edge-requirements).
        """
        # index to sort edges so they are sorted by child time
        # and grouped by child id. Within each group with the same child ID,
        # the edges are sorted by child_left
        if len(self.iedges) == 0:
            return
        edge_order = np.lexsort(
            (
                self.iedges.child_left,
                self.iedges.child,
                -self.nodes.time[self.iedges.child],  # Primary key
            )
        )
        new_iedges = IEdgeTable()
        for i in edge_order:
            new_iedges.append(self.iedges[i])
        new_iedges.flags = self.iedges.flags  # should be the same only we can assure
        new_iedges.set_bitflag(IEDGES_CHILD_IDS_ADJACENT)
        self.iedges = new_iedges

    def graph(self):
        """
        Return a genetic inheritance graph (Graph) object based on these tables
        """
        from .graph import (
            Graph as GIG,
        )  # Hack to avoid complaints about circular imports

        return GIG(self)

    def samples(self):
        """
        Return the IDs of all samples in the nodes table
        """
        if len(self.nodes) == 0:
            return np.array([], dtype=np.int32)
        return np.where(self.nodes.flags & NODE_IS_SAMPLE)[0]

    def change_times(self, timedelta):
        """
        Add a value to all times (nodes and mutations)
        """
        # TODO: currently node rows are frozen, so we can't change them in-place
        # We should make them more easily editable but also change the cached times
        node_table = NodeTable()
        for node in self.nodes:
            node_table.add_row(
                time=node.time + timedelta,
                flags=node.flags,
                individual=node.individual,
            )
        self.nodes = node_table
        # TODO - same for mutations, when we implement them

    def decapitate(self, time):
        """
        Remove nodes that are older than a certain time, and edges that have those
        as parents. Also remove individuals associated with those nodes. Return a
        mapping of old node ids to new node ids.
        """
        individuals = IndividualTable()
        individual_map = {NULL: NULL}
        nodes = NodeTable()
        node_map = np.full(len(self.nodes), NULL, dtype=np.int64)
        iedges = IEdgeTable()
        for u, nd in enumerate(self.nodes):
            if nd.time < time:
                i = nd.individual
                indiv = self.individuals[i]
                if i not in individual_map:
                    individual_map[i] = individuals.add_row(
                        parents=tuple(
                            individual_map.get(j, NULL) for j in indiv.parents
                        )
                    )
                node_map[u] = nodes.add_row(
                    time=nd.time, flags=nd.flags, individual=individual_map[i]
                )
        for ie in self.iedges:
            if node_map[ie.parent] != NULL and node_map[ie.child] != NULL:
                iedges.add_row(
                    ie.child_left,
                    ie.child_right,
                    ie.parent_left,
                    ie.parent_right,
                    child=node_map[ie.child],
                    parent=node_map[ie.parent],
                )
        self.nodes = nodes
        self.iedges = iedges
        self.individuals = individuals
        self.sort()
        return node_map

    @classmethod
    def from_tree_sequence(cls, ts, *, chromosome=None, timedelta=0, **kwargs):
        """
        Create a GIG Tables object from a tree sequence. To create a GIG
        directly, use GIG.from_tree_sequence() which simply wraps this method.

        :param tskit.TreeSequence ts: The tree sequence on which to base the Tables
            object
        :param int chromosome: The chromosome number to use for all interval edges
        :param float timedelta: A value to add to all node times (this is a hack until
            we can set entire columns like in tskit, see #issues/19)
        :param kwargs: Other parameters passed to the Tables constructor
        """
        ts_tables = ts.tables
        tables = cls(time_units=ts.time_units)
        if ts_tables.migrations.num_rows > 0:
            raise NotImplementedError
        if ts_tables.mutations.num_rows > 0:
            raise NotImplementedError
        if ts_tables.sites.num_rows > 0:
            raise NotImplementedError
        if ts_tables.populations.num_rows > 1:
            # If there is only one population, ignore it
            raise NotImplementedError
        for row in ts_tables.nodes:
            obj = dataclasses.asdict(row)
            obj["time"] += timedelta
            tables.nodes.append(obj)
        for row in ts_tables.edges:
            obj = dataclasses.asdict(row)
            if chromosome is not None:
                obj["parent_chromosome"] = obj["child_chromosome"] = chromosome
            tables.iedges.append(obj)
        for row in ts_tables.individuals:
            obj = dataclasses.asdict(row)
            tables.individuals.append(obj)
        tables.sort()
        return tables

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

        if not self.iedges.has_bitflag(IEDGES_PARENT_OLDER_THAN_CHILD):
            raise ValueError("Must guarantee each iedge has child younger than parent")

        if time_cutoff is None:
            time_cutoff = np.inf
        # Use a dynamic stack as we hopefully will be visiting a minority of nodes
        node_times = self.nodes.time
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
            if len(u_dict) > 0 and len(v_dict) > 0:
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
                        to_store = MRCAintervals([], [])
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

            for ie_id in self.iedges.ids_for_child(child):
                # See note in sample resolve algorithm re efficiency and
                # indexing by left coord. Note this is more complex in the
                # mrca case because we have to treat intervals with different
                # offsets / orientations separately
                ie = self.iedges[ie_id]
                parent = ie.parent
                child_ivl = P.closedopen(ie.child_left, ie.child_right)  # cache
                is_inversion = ie.is_inversion()
                for u_or_v, interval_dict in enumerate((u_dict, v_dict)):
                    for (offset, already_inverted), intervals in interval_dict.items():
                        for interval in intervals:
                            if c := child_ivl & interval:
                                p = self.iedges.transform_interval(
                                    ie_id, (c.lower, c.upper), ROOTWARDS
                                )
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
    def random_match_pos(mrcas_structure, rng):
        """
        Given a structure returned by the find_mrca_regions method, choose
        a position uniformly at random from the mrca regions and return
        the equivalent position in u and v.

        .. note::
            It is hard to know from the MRCA structure whether intervals are
            adjacent, so if this is used to locate a breakpoint, the choice of
            whether a breakpoint is positioned to the left or right of the returned
            position is left to the user.

        :param obj rng: A numpy random number generator with a method ``integers()``
            that behaves like ``np.random.default_rng().integers``.
        :returns: a named tuple of ``(u_position, v_position, opposite_orientations)``
            Positions are chosen at random if there are multiple equivalent positions
            for u, or multiple equivalent positions for v. If one of the sequences is
            inverted relative to the other then ``.opposite_orientations`` is ``True``.
        :rtype: ComparablePositions
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

                    # TODO: check this works if there is an inversion at the
                    # *start* of the coordinate space (e.g. iedge(0, 5, 5, 0, ...))
                    # producing e.g. u=(0, 5), v=(5, 0) at this point. In this case
                    # position loc=4 will calculate v=-(5 - 4 - 1) returning
                    # (4, 0, True). Here's an example
                    #
                    # u = abcdefghi      v = DCBAEFGHI
                    #
                    #           abcdefghi  (u)
                    #      IHGFEABCD       (v)
                    #
                    #  A recombinant after pos 4 should give abcd or IHGFEABCDefghi

                    if u[0] < u[1]:
                        if v[0] < v[1]:
                            return ComparablePositions(u[0] + loc, v[0] + loc, False)
                        else:
                            return ComparablePositions(u[0] + loc, v[0] - loc - 1, True)
                    else:
                        if v[0] < v[1]:
                            return ComparablePositions(u[0] - loc - 1, v[0] + loc, True)
                        else:
                            return ComparablePositions(
                                u[0] - loc - 1, v[0] - loc - 1, False
                            )
                loc -= x[1] - x[0]
