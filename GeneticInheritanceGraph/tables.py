import dataclasses

import numpy as np
import pandas as pd
import tskit

from .util import truncate_rows

NULL = -1


@dataclasses.dataclass
class IntervalTableRow:
    parent: int
    child: int
    parent_left: float
    child_left: float
    parent_right: float
    child_right: float
    parent_chromosome: int = None
    child_chromosome: int = None


@dataclasses.dataclass
class NodeTableRow:
    time: float
    flags: int = 0
    individual: int = NULL


@dataclasses.dataclass
class IndividualTableRow:
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

    def append(self, obj) -> int:
        """
        Append a row to a BaseTable by picking required fields from a passed-in object.
        The object can be a dict, a dataclass, or an object with an .asdict() method.

        :return: The row ID of the newly added row

        Example:
            new_id = table.append({"field1": "foo", "field2": "bar"})
        """
        try:
            obj = obj.asdict()
        except AttributeError:
            try:
                obj = dataclasses.asdict(obj)
            except TypeError:
                pass
        new_dict = {k: v for k, v in obj.items() if k in self.RowClass.__annotations__}
        self._data.append(self.RowClass(**new_dict))
        return len(self._data) - 1

    def _text_header_and_rows(self, limit=None):
        headers = ("id",) + tuple(self.RowClass.__annotations__.keys())
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

    def _df(self):
        """
        Temporary hack to convert the table to a Pandas dataframe.
        Shouldn't be used for anything besides exploratory work!
        """
        return pd.DataFrame([dataclasses.asdict(row) for row in self._data])


class IntervalTable(BaseTable):
    RowClass = IntervalTableRow

    def append(self, obj) -> int:
        """
        Append a row to a BaseTable by picking required fields from a passed-in object.
        The object can be a dict, a dataclass, or an object with an .asdict() method.
        If a `left` field is present in the object, is it placed in the ``parent_left``
        and ``child_left`` attributes of this row (and similarly for a ``right`` field)

        :return: The row ID of the newly added row

        Example:
            new_id = tables.intervals.append(ts.edge(0))
        """
        try:
            obj = obj.asdict()
        except AttributeError:
            pass
        obj["child_left"] = obj["parent_left"] = obj["left"]
        obj["child_right"] = obj["parent_right"] = obj["right"]
        new_dict = {k: v for k, v in obj.items() if k in self.RowClass.__annotations__}
        self._data.append(self.RowClass(**new_dict))
        return len(self._data) - 1


class NodeTable(BaseTable):
    RowClass = NodeTableRow


class IndividualTable(BaseTable):
    RowClass = IndividualTableRow


class TableGroup:
    """
    A group of tables describing a Genetic Inheritance Graph (GIG),
    similar to a tskit TableCollection.
    """

    def __init__(self, time_units=None):
        self.nodes = NodeTable()
        self.intervals = IntervalTable()
        self.individuals = IndividualTable()
        self.time_units = "unknown" if time_units is None else time_units

    def __str__(self):
        # To do: make this look nicer
        return "\n\n".join(
            [
                "== NODES ==\n" + str(self.nodes),
                "== INTERVALS ==\n" + str(self.intervals),
            ]
        )

    @classmethod
    def from_tree_sequence(cls, ts, *, chromosome=None, timedelta=0, **kwargs):
        """
        Create a GIG TableGroup from a tree sequence.

        :param tskit.TreeSequence ts: The tree sequence on which to base the TableGroup
        :param int chromosome: The chromosome number to use for all intervals
        :param float timedelta: A value to add to all node times (this is a hack until
            we can set entire columns like in tskit, see #issues/19)
        :param kwargs: Other parameters passed to the TableGroup constructor
        """
        tables = ts.tables
        gig_tables = cls()
        if tables.migrations.num_rows > 0:
            raise NotImplementedError
        if tables.mutations.num_rows > 0:
            raise NotImplementedError
        if tables.sites.num_rows > 0:
            raise NotImplementedError
        if tables.populations.num_rows > 1:
            # If there is only one population, ignore it
            raise NotImplementedError
        for row in tables.nodes:
            obj = dataclasses.asdict(row)
            obj["time"] += timedelta
            gig_tables.nodes.append(obj)
        for row in tables.edges:
            obj = dataclasses.asdict(row)
            if chromosome is not None:
                obj["parent_chromosome"] = obj["child_chromosome"] = chromosome
            gig_tables.intervals.append(obj)
        return gig_tables
