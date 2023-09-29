import dataclasses

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
    def __init__(self):
        # TODO - at the moment we store each row as a separate dataclass object,
        # in the self.data list. This is not very efficient, but it is simple. We
        # will probably want to follow the tskit paradigm of storing such that
        # each column can be accessed as a numpy array too. I'm not sure how to
        # do this without diving into C implementations.
        self.data = []

    @property
    def num_rows(self):
        return len(self.data)

    def __str__(self):
        headers, rows = self._text_header_and_rows(limit=20)
        return tskit.util.unicode_table(rows, header=headers, row_separator=False)

    def _repr_html_(self):
        """
        Called e.g. by jupyter notebooks to render tables
        """
        headers, rows = self._text_header_and_rows(limit=20)
        return tskit.util.html_table(rows, header=headers)

    def __getitem__(self, index):
        return self.data[index]

    def add_row(self, *args, **kwargs) -> int:
        """
        Add a row to a BaseTable: the arguments are passed to the RowClass constructor
        (with the RowClass being dataclass).

        :return: The row ID of the newly added row

        Example:
            new_id = table.add_row(dataclass_field1="foo", dataclass_field2="bar")
        """
        self.data.append(self.RowClass(*args, **kwargs))
        return len(self.data) - 1

    def append(self, obj) -> int:
        """
        Append a row to a BaseTable by picking the required fields from a passed-in object.
        The object can be a dict or an object (e.g. a tskit TableRow) with an .asdict() method

        :return: The row ID of the newly added row

        Example:
            new_id = table.append({"dataclass_field1": "foo", "dataclass_field2": "bar"})
        """
        try:
            obj = obj.asdict()
        except AttributeError:
            pass
        new_dict = {k: v for k, v in obj.items() if k in self.RowClass.__annotations__}
        self.data.append(self.RowClass(**new_dict))
        return len(self.data) - 1

    def _text_header_and_rows(self, limit=None):
        headers = ("id",) + tuple(self.RowClass.__annotations__.keys())
        rows = []
        row_indexes = truncate_rows(self.num_rows, limit)
        for j in row_indexes:
            if j == -1:
                rows.append(f"__skipped__{self.num_rows-limit}")
            else:
                row = self[j]
                rows.append([str(j)] + [f"{x}" for x in dataclasses.asdict(row).values()])
        return headers, rows



class IntervalTable(BaseTable):
    RowClass = IntervalTableRow
    def append(self, obj) -> int:
        """
        Append a row to a BaseTable by picking the required fields from a passed-in object.
        The object can be a dict or an object (e.g. a tskit TableRow) with an .asdict() method.
        If a `left` field is present in the object, is it placed in the parent_left and child_left
        attributes of this row (and similarly for a `right` field)

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
        self.data.append(self.RowClass(**new_dict))
        return len(self.data) - 1

class NodeTable(BaseTable):
    RowClass = NodeTableRow

class IndividualTable(BaseTable):
    RowClass = IndividualTableRow

class TableCollection:
    def __init__(self):
        self.nodes = NodeTable()
        self.intervals = IntervalTable()
        self.individuals = IndividualTable()
        self.time_units = "unknown"

    def __str__(self):
        # To do: make this look nicer
        return "\n\n".join([
            "== NODES ==\n" + str(self.nodes),
            "== INTERVALS ==\n" + str(self.intervals),
        ])

    @classmethod
    def from_tree_sequence(cls, ts, timedelta=0):
        """
        Import from a tree sequence. NB: timedelta is a hack until we can set entire columns
        like in tskit
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
            obj = row.asdict()
            obj["time"] += timedelta
            gig_tables.nodes.append(obj)
        for row in tables.edges:
            gig_tables.intervals.append(row)
        return gig_tables
