import dataclasses

import numpy as np
import numpy.typing as npt
import pandas as pd
import tskit

from .constants import NODE_IS_SAMPLE
from .constants import NULL
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

    def copy(self):
        """
        Returns a deep copy of this table
        """
        copy = self.__class__()
        copy._data = self._data.copy()
        return copy

    def __eq__(self, other):
        return self._data == other._data

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

    def add_rows(self, rowlist):
        """
        Add a list of rows to a BaseTable. Each row must contain objects of the
        required table row type. This is a convenience function
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
                rows.append(f"__skipped__{len(self)-limit}")
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
    RowClass = IEdgeTableRow

    def _from_tskit(self, kwargs):
        new_kw = {}
        for side in ("left", "right"):
            if side in kwargs:
                new_kw["child_" + side] = new_kw["parent_" + side] = kwargs[side]
        new_kw.update(kwargs)
        return new_kw

    def append(self, obj) -> int:
        kwargs = self._from_tskit(self.to_dict(obj))
        return self.add_row(
            **{k: v for k, v in kwargs.items() if k in self.RowClass.__annotations__}
        )

    def add_row(self, *args, **kwargs) -> int:
        """
        Add a row to an IEdgeTable. If a `left` field is present in the object,
        is it placed in the ``parent_left`` and ``child_left`` attributes of this
        row (and similarly for a ``right`` field).

        To allow tskit conversion, named parameters corresponding to genome
        positions (i.e. child_left, child_right, parent_left, parent_right)
        are converted to integers.

        :return: The row ID of the newly added row

        Example:
            new_id = tables.iedges.append(ts.edge(0))
        """
        # NB: here we override the default method to allow integer conversion and
        # left -> child_left, parent_left etc., to aid conversion from tskit
        # For simplicity, this only applies for named args, not positional ones
        pcols = ("child_left", "child_right", "parent_left", "parent_right")
        args = [self._check_int(v) if i < 4 else v for i, v in enumerate(args)]
        kwargs = {k: self._check_int(v) if k in pcols else v for k, v in kwargs.items()}
        return super().add_row(*args, **kwargs)

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

    @property
    def flags(self) -> npt.NDArray[np.uint32]:
        return np.array([row.flags for row in self.data], dtype=np.uint32)

    @property
    def time(self) -> npt.NDArray[np.float64]:
        return np.array([row.time for row in self.data], dtype=np.float64)


class IndividualTable(BaseTable):
    RowClass = IndividualTableRow


class Tables:
    """
    A group of tables describing a Genetic Inheritance Graph (GIG),
    similar to a tskit TableCollection.
    """

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

    def copy(self):
        """
        Return a deep copy of the tables
        """
        copy = self.__class__()
        for name in self.table_classes.keys():
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
        # index to sort edges so they are sorted by parent time
        # and grouped by parent id. Within each group with the same child ID,
        # the edges are sorted by child_left
        if len(self.iedges) == 0:
            return
        edge_order = np.lexsort(
            (
                self.iedges.child_left,
                self.iedges.child,
                self.iedges.parent,
                self.nodes.time[self.iedges.parent],  # Primary key
            )
        )
        new_iedges = IEdgeTable()
        for i in edge_order:
            new_iedges.append(self.iedges[i])
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
        return tables
