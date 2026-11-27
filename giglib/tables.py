import collections
import dataclasses
import itertools
import logging
import math
import types

import numpy as np
import numpy.typing as npt
import sortedcontainers
import tskit

from . import intervals as interval_sets
from .constants import Const, ValidFlags
from .util import (
    DEFAULT_ALLELE_COMPLEMENTS,
    complement_allele,
    is_unknown_time,
    truncate_rows,
    warn_tskit_name,
)

NULL = Const.NULL  # alias for tskit.NULL
_node_table_versions = itertools.count()
# The ValidFlags as plain ints, for speed when testing many flags (see Tables._check_iedges)
_BITS = types.SimpleNamespace(
    **{name: int(flag) for name, flag in ValidFlags.__members__.items()}
)  # unique versions of node tables (see NodeTable._changed)
NODE_IS_SAMPLE = Const.NODE_IS_SAMPLE  # alias for tskit.NODE_IS_SAMPLE


class _UnpackableAsKeywords:
    """
    A mixin for named tuples, allowing them to be unpacked as keyword arguments
    using ``**``, e.g. ``gig.allele_at(u, **position)``. This works because the
    field names match the argument names used throughout the API. Otherwise, these
    behave exactly as named tuples: iterating gives the values (not the field names),
    and indexing with an integer or slice works as normal.
    """

    __slots__ = ()

    def keys(self):
        return self._fields

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        return super().__getitem__(key)


class Segment(_UnpackableAsKeywords, collections.namedtuple("Segment", "left, right, chromosome")):
    """
    A segment of a chromosome, defined by its left (inclusive) and right (exclusive)
    positions and chromosome ID. This is a named tuple, which can also be unpacked as
    keyword arguments, e.g. ``gig.haplotype(u, **segment)``.
    """

    __slots__ = ()


class Position(_UnpackableAsKeywords, collections.namedtuple("Position", "position, chromosome")):
    """
    A single position on a chromosome, in the coordinates of a particular node.
    Like :class:`Segment`, the chromosome ID comes after the position. This is a
    named tuple, which can also be unpacked as keyword arguments, e.g.
    ``gig.allele_at(u, **position)``.
    """

    __slots__ = ()


class Homology(collections.namedtuple("Homology", "u, v, mrca, mrca_segment")):
    """
    A pair of homologous segments, as returned by :meth:`Tables.find_homology`: the
    :class:`Segment` ``u`` of one genome and the :class:`Segment` ``v`` of another trace
    back to the same segment, ``mrca_segment``, of their most recent common ancestor, the
    node ``mrca``. This is homology by descent, not sequence similarity (see
    :ref:`sec_usage_homology`).

    Positions in ``u`` increase from ``u.left`` to ``u.right``, and correspond, in order, to
    positions running from ``v.left`` towards ``v.right``, and from ``mrca_segment.left``
    towards ``mrca_segment.right``. So if ``v`` is reversed relative to ``u`` (e.g. through
    an inversion), ``v.left > v.right``, as for the parent interval of an inverted iedge
    (and likewise for ``mrca_segment``).
    """

    __slots__ = ()

    @property
    def map(self):
        """
        The :class:`PositionMap` from positions in ``u`` to the homologous positions in
        ``v``
        """
        return PositionMap.from_intervals((self.u.left, self.u.right), (self.v.left, self.v.right))

    @property
    def length(self):
        """
        The number of homologous positions
        """
        return self.u.right - self.u.left

    @property
    def is_reversed(self):
        """
        Whether ``v`` is reversed relative to ``u``
        """
        return self.v.left > self.v.right


class PositionMap(collections.namedtuple("PositionMap", "sign, offset")):
    """
    A map from the coordinates of one genome to those of another, as made by an
    iedge: a shift, which also reverses the order of positions if ``sign`` is
    ``-1`` (an inversion). This is the one place where coordinates are mapped
    through iedges, so all the code that does so uses it.

    There are two kinds of coordinate, which map differently when the order is
    reversed:

    * The *ends of an interval* (e.g. ``child_left`` and ``child_right``) lie
      between positions. An end ``x`` maps to ``sign * x + offset``, so an
      interval ``[left, right)`` maps to the ends ``(sign * left + offset,
      sign * right + offset)``. If the order is reversed, the first end is larger
      than the second, as for the parent interval of an inverted iedge.
    * A *position* ``x`` is the base between the ends ``x`` and ``x + 1``. It maps
      to the base between the ends that they map to, so to ``x + offset``, or
      ``offset - x - 1`` if reversed.

    For example, an iedge whose child interval ``[10, 20)`` is inherited from the
    parent interval ``(30, 20]`` maps the ends 10 and 20 to 30 and 20, and the
    positions 10 and 19 to 29 and 20:

    >>> m = PositionMap.from_intervals((10, 20), (30, 20))
    >>> m
    PositionMap(sign=-1, offset=40)
    >>> m.interval(10, 20), m.position(10), m.position(19)
    ((30, 20), 29, 20)

    Maps along a chain of iedges combine into a single map with :meth:`then`, and
    :meth:`inverse` maps back the other way.
    """

    __slots__ = ()

    @classmethod
    def from_intervals(cls, interval, new_interval):
        """
        Return the map that takes the ends of ``interval`` to the corresponding ends
        of ``new_interval``. If one interval has its ends in increasing order and the
        other in decreasing order, the map reverses the order of positions.

        :param tuple(int, int) interval: The (left, right) ends of the interval
        :param tuple(int, int) new_interval: The ends that ``left`` and ``right`` map
            to, respectively
        :raises ValueError: If the intervals have different lengths
        """
        (left, right), (new_left, new_right) = interval, new_interval
        if abs(right - left) != abs(new_right - new_left):
            raise ValueError(f"Intervals {interval} and {new_interval} have different lengths")
        sign = -1 if (left < right) != (new_left < new_right) else 1
        return cls(sign, new_left - sign * left)

    @classmethod
    def from_iedge(cls, iedge):
        """
        Return the map from the child coordinates of an iedge to its parent
        coordinates. The :meth:`inverse` maps from parent to child coordinates.

        :param iedge: An :class:`~tables.IEdgeTableRow` or :class:`~graph.IEdge`
        """
        return cls.from_intervals((iedge.child_left, iedge.child_right), (iedge.parent_left, iedge.parent_right))

    @property
    def is_reversing(self):
        """
        Whether this map reverses the order of positions (e.g. through an odd
        number of inversions), in which case alleles are complemented.
        """
        return self.sign < 0

    def interval(self, left, right):
        """
        Map the ends of the interval ``[left, right)``. If the map reverses the
        order of positions, the first end returned is larger than the second.

        :return: The mapped ``(left, right)`` ends
        :rtype: tuple(int, int)
        """
        return self.sign * left + self.offset, self.sign * right + self.offset

    def position(self, x):
        """
        Map a single position (base) ``x``.

        :rtype: int
        """
        return x + self.offset if self.sign > 0 else self.offset - x - 1

    def inverse(self):
        """
        Return the map that undoes this one.

        :rtype: PositionMap
        """
        return PositionMap(self.sign, -self.sign * self.offset)

    def then(self, other):
        """
        Return the map that applies this map, followed by ``other``. For example, if
        ``a`` maps a child to its parent, and ``b`` maps the parent to the
        grandparent, ``a.then(b)`` maps the child to the grandparent.

        :rtype: PositionMap
        """
        return PositionMap(other.sign * self.sign, other.sign * self.offset + other.offset)


class _IEdgeMethods:
    """
    Properties and methods shared by iedge table rows (:class:`IEdgeTableRow`) and
    iedge objects in a graph (:class:`~giglib.graph.IEdge`). These do not assume that
    the iedge is valid (e.g. that ``child_left < child_right``, or that the child and
    parent intervals have the same length), as table rows may not have been validated.
    Those that rely on validity are only defined on :class:`~giglib.graph.IEdge`.
    """

    __slots__ = ()

    @property
    def child_span(self):
        """
        The span of the child interval, ``child_right - child_left`` (positive
        for a valid iedge)
        """
        return self.child_right - self.child_left

    @property
    def parent_span(self):
        """
        The span of the parent interval, ``parent_right - parent_left``. This is
        negative if the iedge is an inversion.
        """
        return self.parent_right - self.parent_left

    def is_inversion(self):
        """
        Is the parent interval in the opposite orientation to the child interval
        """
        return self.parent_span * self.child_span < 0

    def is_simple_inversion(self):
        """
        Is this an in-situ inversion, where parent and child coords are simply swapped
        """
        return self.parent_left == self.child_right and self.parent_right == self.child_left

    @property
    def parent_max(self):
        """
        The upper (exclusive) bound of the parent interval, i.e. one more than the
        highest parent position inherited through this iedge. This is
        ``max(parent_left, parent_right)``, as for an inversion, ``parent_left >
        parent_right``. For example, an inverted iedge with ``parent_left=10`` and
        ``parent_right=0`` inherits parent positions 0 to 9, so ``parent_max`` is 10.
        """
        return max(self.parent_right, self.parent_left)

    @property
    def parent_min(self):
        """
        The lower (inclusive) bound of the parent interval, i.e. the lowest parent
        position inherited through this iedge. This is ``min(parent_left,
        parent_right)``, as for an inversion, ``parent_left > parent_right``.
        """
        return min(self.parent_right, self.parent_left)

    @property
    def child_max(self):
        """
        The upper (exclusive) bound of the child interval, i.e. one more than the
        highest child position covered by this iedge. For a valid iedge, this is
        ``child_right``.
        """
        return max(self.child_right, self.child_left)

    @property
    def child_min(self):
        """
        The lower (inclusive) bound of the child interval, i.e. the lowest child
        position covered by this iedge. For a valid iedge, this is ``child_left``.
        """
        return min(self.child_right, self.child_left)

    @property
    def child_segment(self):
        """
        The segment of the child chromosome for this edge
        """
        return Segment(self.child_left, self.child_right, self.child_chromosome)

    @property
    def parent_segment(self):
        """
        The segment of the parent chromosome for this edge
        """
        return Segment(self.parent_left, self.parent_right, self.parent_chromosome)


class IEdgeTableRow(
    _IEdgeMethods,
    collections.namedtuple(
        "IEdgeTableRow",
        ", ".join(
            [
                "child",
                "parent",
                "child_left",
                "child_right",
                "parent_left",
                "parent_right",
                "child_chromosome",
                "parent_chromosome",
                "edge",
            ]
        ),
    ),
):
    """
    A row in an IEdgeTable. This is a named tuple, so the fields can be accessed
    by name as well as by index.
    """


class _NodeMethods:
    """
    Methods shared by node table rows (:class:`NodeTableRow`) and node objects in a
    graph (:class:`~giglib.graph.Node`)
    """

    __slots__ = ()

    def is_sample(self):
        """
        Is this node flagged as a sample
        """
        return bool(self.flags & NODE_IS_SAMPLE)


class NodeTableRow(
    _NodeMethods,
    collections.namedtuple(
        "NodeTableRow",
        ", ".join(["time", "flags", "individual", "metadata"]),
    ),
):
    pass


class IndividualTableRow(collections.namedtuple("IndividualTableRow", "flags, location, parents, metadata")):
    pass


class _MutationMethods:
    """
    Methods shared by mutation table rows (:class:`MutationTableRow`) and mutation
    objects in a graph (:class:`~giglib.graph.Mutation`)
    """

    __slots__ = ()

    def is_time_known(self):
        """
        Return ``True`` if the time of this mutation is known
        """
        return not is_unknown_time(self.time)

    # Unknown times are NaNs, which would otherwise make identical rows compare unequal
    def _comparable(self):
        return tuple(
            "unknown" if (i == self._fields.index("time") and is_unknown_time(x)) else x for i, x in enumerate(self)
        )

    def __eq__(self, other):
        if not isinstance(other, tuple) or len(self) != len(other):
            return False
        if not hasattr(other, "_comparable"):
            return tuple.__eq__(self, other)
        return self._comparable() == other._comparable()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._comparable())


class MutationTableRow(
    _MutationMethods,
    collections.namedtuple("MutationTableRow", "node, position, chromosome, time, derived_state, metadata"),
):
    """
    A row in a :class:`MutationTable`. The ``position`` is in the coordinates of
    ``node`` for the given ``chromosome``, and the ``derived_state`` replaces the
    base at that position. See :ref:`sec_design_mutations` for details.
    """


class BaseTable:
    _RowClass = None
    _non_int64_fieldtypes = types.MappingProxyType({})  # By default all fields are int64
    initial_size = 64  # default
    max_resize = 2**18  # maximum number of rows by which we expand internal storage
    _frozen = None  # Will be overridden during init

    def _create_datastore(self):
        self._datastore = np.empty(
            self.initial_size,
            dtype=[(name, self._non_int64_fieldtypes.get(name, np.int64)) for name in self._RowClass._fields],
        )

    def __init__(self, initial_size=None):
        self._frozen = False
        if initial_size is not None:
            self.initial_size = initial_size
        self._create_datastore()
        # Create a view into the datastore: can be resized without allocating new memory
        self._data = self._datastore[0:0]
        # "Ragged" data is stored separately, as it is of variable length
        self._extra_data = []

    def freeze(self):
        """
        Freeze the table so that it cannot be modified. In future versions, this
        may also shrink the storage required for the table
        """
        self._datastore.setflags(write=False)
        # _data is a view of _datastore created before freezing, so it keeps its own
        # (writeable) flag, which must also be unset
        self._data.setflags(write=False)
        self._frozen = True
        # TODO: use resize(refcheck=False) [dangerous!] to shrink self._datastore to the
        # size of self._data.

    def copy(self):
        copy = self.__class__()
        copy._datastore = self._datastore.copy()
        copy._datastore.setflags(write=True)
        copy._data = copy._datastore[0 : len(self._data)]
        copy._frozen = False
        return copy

    def __setattr__(self, attr, value):
        if self._frozen:
            raise AttributeError("Trying to set attribute on a frozen (read-only) instance")
        super().__setattr__(attr, value)
        if attr == "_datastore":
            # Whenever the storage array is replaced, make read-only views of its
            # columns (in-place changes to the storage are seen through these views)
            read_only = value.view()
            read_only.flags.writeable = False
            super().__setattr__("_read_only_columns", {name: read_only[name] for name in read_only.dtype.names})

    def __eq__(self, other):
        return np.all(self._data == other._data) and self._extra_data == other._extra_data

    def __getitem__(self, index):
        # .tolist() converts the values to Python scalars (ints and floats), which is
        # both faster than unpacking a numpy row, and gives row objects that print nicely
        return self._RowClass(*self._data[index].tolist())

    def clear(self):
        """
        Deletes all rows in this table.
        """
        self._create_datastore()
        self._data = self._datastore[0:0]

    def _expand_datastore(self):
        old = self._datastore
        try:
            self._datastore = np.empty_like(
                self._datastore,
                # Double and add one, so that a table preallocated with 0 rows can grow
                shape=min(2 * len(self._datastore) + 1, len(self._datastore) + self.max_resize),
            )
        except MemoryError:
            # should try dumping the table to disk here, I suppose?
            raise
        self._datastore[0 : len(old)] = old
        logging.debug(f"preallocated space for {len(self._datastore) - len(old)} new rows")
        # Hopefully force garbage collection of old array: won't work not if the user
        # has a variable accessing the old array
        del old

    def __len__(self):
        return len(self._data)

    def _column(self, name):
        """
        Return a read-only view of the named column. Columns are read-only because
        editing them directly would bypass the table's indexes and validity flags
        (e.g. ``iedges.child[0] = 2`` would leave ``ids_for_child()`` returning the
        wrong iedges). Code in this library that needs to edit a column should use
        ``self._data[name]``, and then update any indexes and flags.

        For speed, this slices a read-only view of the whole column in the storage
        array, which is only created when the storage array is replaced (see
        ``__setattr__``), rather than setting the read-only flag on every access.
        """
        return self._read_only_columns[name][: len(self._data)]

    @property
    def _num_preallocated_rows(self):
        return len(self._datastore)

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

    def _text_header_and_rows(self, limit=None):
        headers = ("id",)
        headers += tuple(k for k in self._RowClass._fields if k != "_")
        rows = []
        row_indexes = truncate_rows(len(self), limit)
        for j in row_indexes:
            if j == -1:
                rows.append(f"__skipped__{len(self) - limit}")
            else:
                row = self[j]
                rows.append([str(j)] + [f"{x}" for x in row])
        return headers, rows

    def _repr_html_(self):
        """
        Called e.g. by jupyter notebooks to render tables
        """
        from . import _print_options  # pylint: disable=import-outside-toplevel

        headers, rows = self._text_header_and_rows(limit=_print_options["max_lines"])
        html = tskit.util.html_table(rows, header=headers)
        return html.replace("tskit.set_print_options", f"{__package__}.set_print_options")

    @staticmethod
    def to_dict(obj):
        try:
            return obj._asdict()
        except AttributeError:
            try:
                return dataclasses.asdict(obj)
            except TypeError:
                pass
        return obj


class BaseExtraTable(BaseTable):
    """
    A table that has extra data associated with each row (e.g. metadata)
    that can't be stored in the main numpy data array because it is ragged.
    Most tables will have metadata, so we define this be default
    """

    _extra_names = ("metadata",)

    def _create_datastore(self):
        self._datastore = np.empty(
            self.initial_size,
            dtype=[
                (name, self._non_int64_fieldtypes.get(name, np.int64))
                for name in self._RowClass._fields
                if name not in self._extra_names
            ],
        )

    def __init__(self, initial_size=None):
        super().__init__(initial_size=initial_size)
        # list of lists containing the extra data in order declared in the RowClass
        self._extra_data = []
        self._extra_data_cols = {name: i for i, name in enumerate(self._extra_names)}

    def copy(self):
        copy = super().copy()
        copy._extra_data = [row.copy() for row in self._extra_data]
        return copy

    def clear(self):
        super().clear()
        self._extra_data = []

    def __getitem__(self, index):
        return self._RowClass(*self._data[index].tolist(), *self._extra_data[index])


class IEdgeTable(BaseTable):
    """
    A table containing the iedges. This contains more complex logic than other
    GIG tables as we want to store and check various forms of
    :ref:`validity<sec_python_api_tables_validity>`, so that we can run algorithms
    directly on the tables rather than on a frozen GIG. For example, the
    :meth:`ids_for_child` method requires
    :data:`~giglib.ValidFlags.IEDGES_FOR_CHILD_ADJACENT` to be true.
    """

    _RowClass = IEdgeTableRow
    _non_int64_fieldtypes = types.MappingProxyType(
        {
            "child_chromosome": np.int16,  # Save some space
            "parent_chromosome": np.int16,  # Save some space
        }
    )

    # define each property by hand, for speed
    @property
    def child_left(self):
        return self._column("child_left")

    @property
    def child_right(self):
        return self._column("child_right")

    @property
    def parent_left(self):
        return self._column("parent_left")

    @property
    def parent_right(self):
        return self._column("parent_right")

    @property
    def child(self):
        return self._column("child")

    @property
    def parent(self):
        return self._column("parent")

    @property
    def child_chromosome(self):
        return self._column("child_chromosome")

    @property
    def parent_chromosome(self):
        return self._column("parent_chromosome")

    @property
    def edge(self):
        return self._column("edge")

    def __init__(self, initial_size=None):
        super().__init__(initial_size=initial_size)
        self.clear()

    def copy(self):
        """
        Returns an unfrozen deep copy of this table
        """
        copy = super().copy()
        copy._flags = self._flags
        copy._num_validated = self._num_validated
        copy._nodes_version = self._nodes_version
        # Deep copy the id ranges
        copy._id_range_for_child = {}
        for child, chr_range in self._id_range_for_child.items():
            copy._id_range_for_child[child] = {}
            for chromosome, erange in chr_range.items():
                copy._id_range_for_child[child][chromosome] = erange.copy()
        return copy

    def clear(self):
        """
        Deletes all rows in this table.
        """
        # map each child ID to a set of chromosome numbers,
        # each of which is mapped to an IEdgeRange
        self._id_range_for_child = {}
        super().clear()
        self.flags = ValidFlags.GIG  # an empty table satisfies all the requirements
        # The version of the node table that the node-time flags were checked against
        # (see Tables.validate)
        self._nodes_version = None

    @property
    def flags(self):
        """
        The :class:`~giglib.ValidFlags` guaranteed for the whole table (see
        :ref:`sec_validation`). Adding a row removes all the flags, until
        :meth:`~giglib.Tables.validate` checks the new rows.
        """
        # Internally, _flags are guaranteed for the first _num_validated rows
        return ValidFlags(self._flags if self._num_validated == len(self) else 0)

    @flags.setter
    def flags(self, value):
        self._flags = ValidFlags(value)
        self._num_validated = len(self)

    @property
    def num_validated(self):
        """
        The number of :ref:`validated rows<sec_validation_validated_rows>`: the first
        rows of the table, which were checked when :meth:`~giglib.Tables.validate` was
        last called (or marked as valid with :meth:`set_bitflag`). Rows added since then
        are not validated. Some methods, such as :meth:`ids_for_child`, only need the
        rows they read to be validated.
        """
        return self._num_validated

    def _require_flags_for_child(self, u, flags, message):
        """
        Raise a ValueError unless the given flags hold for all the iedges of child
        ``u``: either because they hold for the whole table, or because they hold for
        the validated rows, which include all of ``u``'s iedges. (The index of iedges
        for each child gives the last row of ``u``, even if its rows are not adjacent.)
        """
        if (self._flags & flags) != flags:
            raise ValueError(message)
        if self._num_validated < len(self):
            if any(end > self._num_validated for _, end in self._id_range_for_child.get(u, {}).values()):
                raise ValueError(
                    f"The iedges of node {u} include rows that have not been validated: call tables.validate() first"
                )

    def __eq__(self, other):
        if self.flags != other.flags:
            return False
        if self._id_range_for_child != other._id_range_for_child:
            return False
        return super().__eq__(other)

    def set_bitflag(self, flags):
        """
        Mark the given validity flags as guaranteed for the whole table, including any
        rows added since it was last validated. The flags are trusted without being
        checked, so this avoids the cost of checking (see :ref:`sec_validation_choosing`).
        To check them instead, use :meth:`~giglib.Tables.validate`.

        :param int flags: One or more :class:`~giglib.ValidFlags`, combined using ``|``
        """
        self.flags = self.flags | flags

    def unset_bitflag(self, flags):
        """
        Mark the given validity flags as no longer guaranteed for this table.

        :param int flags: One or more :class:`~giglib.ValidFlags`, combined using ``|``
        """
        self._flags &= ~flags

    def has_all_bitflags(self, flags):
        """
        Check whether this table is guaranteed to satisfy *all* of the given validity
        flags. For example,
        ``iedges.has_all_bitflags(ValidFlags.IEDGES_FOR_CHILD_ADJACENT | ValidFlags.IEDGES_INTERVALS)``
        returns ``True`` only if both flags are set. Passing ``0`` (no flags) returns ``True``.

        :param int flags: One or more :class:`~giglib.ValidFlags`, combined using ``|``
        :return: ``True`` if every bit in ``flags`` is set for this table, otherwise ``False``
        :rtype: bool
        """
        return (self.flags & flags) == flags

    def _increment_id_range_for_child(self, child, child_chromosome, rownum):
        # Update internal data structures
        try:
            self._id_range_for_child[child][child_chromosome][1] = rownum + 1
        except KeyError:
            if child not in self._id_range_for_child:
                self._id_range_for_child[child] = {}
            self._id_range_for_child[child][child_chromosome] = [rownum, rownum + 1]

    def add_row(
        self,
        child,
        parent,
        *,
        child_left=None,
        child_right=None,
        parent_left=None,
        parent_right=None,
        child_chromosome=0,
        parent_chromosome=0,
        edge=NULL,
    ) -> int:
        """
        The canonical way to add data to an IEdgeTable. All the values must be
        integers (a ``ValueError`` is raised otherwise), but no other checks are made
        when adding a row: call :meth:`~giglib.Tables.validate` afterwards to check that the
        iedges satisfy the requirements of a GIG (see :ref:`sec_validation`).

        :param int child: The ID of the child node
        :param int parent: The ID of the parent node
        :param int child_left: The left (inclusive) position of the interval, in the
            child's coordinates. This and the other coordinates must be given by
            keyword. If ``None`` (default), this is 0.
        :param int child_right: The right (exclusive) position of the interval, in the
            child's coordinates. If ``None`` (default), this is 1, as for the default
            sequence length in msprime, so that ``add_row(c, p)`` can be used to build
            simple examples. It must be given if ``child_left`` is given. These defaults
            may change if lengths are stored for each chromosome (see
            https://github.com/hyanwong/giglib/issues/158).
        :param int parent_left: The position in the parent's coordinates corresponding
            to ``child_left``. If ``parent_left > parent_right``, the interval is
            inverted. If ``parent_left`` and ``parent_right`` are both ``None``
            (default), they are taken to be the same as ``child_left`` and
            ``child_right``, i.e. the interval is inherited without being moved or
            inverted. Either both or neither must be given.
        :param int parent_right: The position in the parent's coordinates corresponding
            to ``child_right`` (see ``parent_left``)
        :param int child_chromosome: The chromosome in the child's genome (default 0)
        :param int parent_chromosome: The chromosome in the parent's genome (default 0)
        :param int edge: An optional edge ID (default: ``NULL``, i.e. -1)
        :return: The row ID of the newly added row

        For example:

        .. code-block::

            # Child c inherits positions 0 to 100 of parent p unchanged
            new_id = iedges.add_row(c, p, child_right=100)
            # Child c inherits an inverted copy of parent positions 50 to 100
            new_id = iedges.add_row(
                c,
                p,
                child_left=100,
                child_right=150,
                parent_left=100,
                parent_right=50,
            )
        """
        if child_right is None:
            if child_left is not None:
                raise ValueError("child_right must be given if child_left is given")
            child_right = 1
        if child_left is None:
            child_left = 0
        if parent_left is None and parent_right is None:
            parent_left, parent_right = child_left, child_right
        elif parent_left is None or parent_right is None:
            raise ValueError("Either both or neither of parent_left and parent_right must be given")
        values = (
            child,
            parent,
            child_left,
            child_right,
            parent_left,
            parent_right,
            child_chromosome,
            parent_chromosome,
            edge,
        )
        # Check before storing, as numpy silently converts floats to integers on assignment
        for val in values:
            if int(val) != val:
                raise ValueError(f"Iedge data must be integers, not {val!r}")
        num_rows = len(self._data)
        if len(self._datastore) == num_rows:
            self._expand_datastore()
        self._datastore[num_rows] = values
        # The new row is not yet checked, so the table loses its validity flags until
        # Tables.validate() is called (which then only needs to check the new rows)
        # Index keys are Python ints, even if e.g. numpy integers were passed in
        self._increment_id_range_for_child(int(child), int(child_chromosome), num_rows)
        # expand the visible data array
        self._data = self._datastore[0 : num_rows + 1]

        return num_rows

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

        This is slower than calling :meth:`add_row` directly, which is recommended
        if you require efficiency.
        """
        kwargs = self._from_tskit(self.to_dict(obj))
        return self.add_row(
            **{k: v for k, v in kwargs.items() if k in self._RowClass._fields},
        )

    def squash(self):
        """
        Merge adjacent iedges, in place. Two iedges are merged if they have the same
        child, parent, chromosomes and edge ID, the same orientation (i.e. both or
        neither are inversions), and if the second starts where the first ends, in both
        the child's and the parent's coordinates. So e.g. iedges inheriting child
        positions 0 to 5 and 5 to 8 from parent positions 10 to 15 and 15 to 18 become a
        single iedge inheriting child positions 0 to 8 from parent positions 10 to 18.
        This does not change the ancestry that the iedges describe, so the validity
        flags are kept.

        This requires the iedges for each child to be adjacent, and in order of
        chromosome and then position, e.g. as after :meth:`~giglib.Tables.sort`
        (i.e. the :data:`~giglib.ValidFlags.IEDGES_WITHIN_CHILD_SORTED` flags).
        """
        if self._frozen:
            raise AttributeError("Cannot squash the iedges of frozen tables: use a copy")
        if not self.has_all_bitflags(ValidFlags.IEDGES_WITHIN_CHILD_SORTED):
            raise ValueError(
                "Squashing requires the iedges for each child to be adjacent, and ordered by chromosome and "
                "position (e.g. by sorting the tables)"
            )
        merged = []
        for row in self:
            if merged:
                prev = merged[-1]
                if (
                    (row.child, row.parent, row.child_chromosome, row.parent_chromosome, row.edge)
                    == (prev.child, prev.parent, prev.child_chromosome, prev.parent_chromosome, prev.edge)
                    and row.is_inversion() == prev.is_inversion()
                    and row.child_left == prev.child_right
                    and row.parent_left == prev.parent_right
                ):
                    merged[-1] = prev._replace(child_right=row.child_right, parent_right=row.parent_right)
                    continue
            merged.append(row)
        if len(merged) == len(self):
            return
        flags, nodes_version = self.flags, self._nodes_version
        self.clear()
        for row in merged:
            self.add_row(**row._asdict())
        # merging adjacent iedges keeps all the guarantees
        self.flags, self._nodes_version = flags, nodes_version

    def edges_exist_for_child(self, u, *, chromosome=None):
        if chromosome is None:
            return u in self._id_range_for_child
        else:
            try:
                return chromosome in self._id_range_for_child[u]
            except KeyError:
                return False

    def ids_for_child(self, u, *, chromosome=None):
        """
        Return all the iedge ids for a child. If chromosome is not None, return
        only the iedge IDs for that chromosome.

        .. note::
            The returned IDs are only guaranteed to be ordered by left position if
            :data:`~giglib.ValidFlags.IEDGES_FOR_CHILD_SECONDARY_ORDER_LEFT_ASC` is set.

        :param int u: The child ID
        :param int chromosome: The chromosome number. If ``None`` (default), ruturn
            all the iedge IDs for the child, sorted by chromosome.

        :return: A numpy array of iedge IDs
        :raises ValueError: If the iedges for a child are not adjacent in the table,
            or not ordered by chromosome (as these flags are needed for this child's
            iedges, it is enough for them to be among the
            :ref:`validated rows<sec_validation_validated_rows>`)
        """
        self._require_flags_for_child(
            u,
            ValidFlags.IEDGES_FOR_CHILD_ADJACENT | ValidFlags.IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC,
            "Cannot use this method unless iedges have adjacent child IDs and are ordered by chromosome",
        )
        try:
            if chromosome is None:
                chrom_ranges = self._id_range_for_child[u].values()
                # most efficient way to get first and last values from a dict
                for first, last in zip(chrom_ranges, reversed(chrom_ranges)):
                    return np.arange(first[0], last[1])
            else:
                return np.arange(*self._id_range_for_child[u][chromosome])
        except KeyError:
            return np.arange(0)

    def max_pos_as_child(self, u, *, chromosome=None):
        """
        Return the maximum child position of the iedges for child ``u`` on the given
        chromosome, or on any of its chromosomes if ``chromosome`` is ``None``.
        Returns None if there are no such iedges.

        This is fast because it relies on the iedges for each child being sorted
        and nonoverlapping (:data:`~giglib.ValidFlags.IEDGES_WITHIN_CHILD_SORTED`
        and :data:`~giglib.ValidFlags.IEDGES_FOR_CHILD_NONOVERLAPPING`), and raises
        an error unless both flags are set, at least for the
        :ref:`validated rows<sec_validation_validated_rows>`, which must include all
        the iedges of ``u``.

        .. note::
            This does not look at edges for which this node is a parent,
            so e.g. root nodes may not have a maximum position defined
        """
        self._require_flags_for_child(
            u,
            ValidFlags.IEDGES_WITHIN_CHILD_SORTED | ValidFlags.IEDGES_FOR_CHILD_NONOVERLAPPING,
            "Cannot use this method unless iedges for a child are adjacent, "
            "nonoverlapping, and ordered by chromosome then position",
        )
        try:
            if chromosome is None:
                return max(
                    self[self._id_range_for_child[u][c][1] - 1].child_max for c in self._id_range_for_child[u].keys()
                )
            else:
                return self[self._id_range_for_child[u][chromosome][1] - 1].child_max
        except KeyError:
            return None

    def min_pos_as_child(self, u, *, chromosome=None):
        """
        Return the minimum child position of the iedges for child ``u`` on the given
        chromosome, or on any of its chromosomes if ``chromosome`` is ``None``.
        Returns None if there are no such iedges.

        This is fast because it relies on the iedges for each child being sorted
        and nonoverlapping (:data:`~giglib.ValidFlags.IEDGES_WITHIN_CHILD_SORTED`
        and :data:`~giglib.ValidFlags.IEDGES_FOR_CHILD_NONOVERLAPPING`), and raises
        an error unless both flags are set, at least for the
        :ref:`validated rows<sec_validation_validated_rows>`, which must include all
        the iedges of ``u``.

        .. note::
            This does not look at edges for which this node is a parent,
            so e.g. root nodes may not have a minimum position defined
        """
        self._require_flags_for_child(
            u,
            ValidFlags.IEDGES_WITHIN_CHILD_SORTED | ValidFlags.IEDGES_FOR_CHILD_NONOVERLAPPING,
            "Cannot use this method unless iedges for a child are adjacent, "
            "nonoverlapping, and ordered by chromosome then position",
        )
        try:
            if chromosome is None:
                return min(
                    self[self._id_range_for_child[u][c][0]].child_min for c in self._id_range_for_child[u].keys()
                )
            else:
                return self[self._id_range_for_child[u][chromosome][0]].child_min
        except KeyError:
            return None

    def chromosomes_as_child(self, u):
        """
        Return the chromosome IDs of node ``u`` that it inherits from its parents,
        i.e. those with iedges above ``u``, as a tuple in ascending order. Chromosomes
        of ``u`` which are not traced by any iedge above it (e.g. all the chromosomes
        of a root) are not included.

        :param int u: The node ID
        :return: The chromosome IDs, in ascending order
        :rtype: tuple(int)
        """
        return tuple(sorted(self._id_range_for_child.get(u, {})))


class NodeTable(BaseExtraTable):
    """
    A table containing the nodes with their times
    """

    _RowClass = NodeTableRow
    _non_int64_fieldtypes = types.MappingProxyType({"time": np.float64, "flags": np.uint32})

    def __init__(self, initial_size=None):
        super().__init__(initial_size=initial_size)
        self._changed()

    def _changed(self):
        """
        Record that existing rows have been changed or removed (adding rows does not
        count). The iedges table records the version of the node table that its
        node-time flags were checked against, so that they are checked again if the
        node table has changed since (see :meth:`~giglib.Tables.validate`).
        """
        self._version = next(_node_table_versions)

    def clear(self):
        super().clear()
        self._changed()

    def copy(self):
        # The copy has its own version (it may be changed independently of this table)
        return super().copy()

    # define each property by hand, for speed
    @property
    def time(self) -> npt.NDArray[np.float64]:
        return self._column("time")

    @property
    def flags(self) -> npt.NDArray[np.uint32]:
        return self._column("flags")

    @property
    def individual(self):
        return self._column("individual")

    def add_row(self, time, *, flags=None, individual=None, metadata=None) -> int:
        """
        The canonical way to add data to a NodeTable

        :param float time: The time of the node
        :param int flags: The flags for this node, e.g. ``giglib.NODE_IS_SAMPLE``
            (default: ``None``, treated as 0). This and the other arguments must be
            given by keyword, e.g. ``nodes.add_row(0, flags=NODE_IS_SAMPLE)``, partly
            because the equivalent in tskit takes ``flags`` before ``time``.
        :param int individual: The ID of the individual to which this node belongs
            (default: ``None``, treated as ``NULL``, i.e. no individual)
        :param metadata: Any metadata for this node (default: ``None``)
        :return: The row ID of the newly added row

        Example:
            new_id = nodes.add_row(0, flags=NODE_IS_SAMPLE)
        """
        if flags is None:
            flags = 0
        if individual is None:
            individual = NULL

        num_rows = len(self._data)
        if len(self._datastore) == num_rows:
            self._expand_datastore()
        self._datastore[num_rows] = (time, flags, individual)
        self._extra_data.append([metadata])
        self._data = self._datastore[0 : len(self._data) + 1]
        return num_rows

    def append(self, obj) -> int:
        return self.add_row(**{k: v for k, v in self.to_dict(obj).items() if k in self._RowClass._fields})

    def filter(self, flags=None, time=None, *, candidates=None):
        """
        Return the IDs of nodes matching all the given criteria.

        :param int flags: If not ``None``, only return nodes for which *all* the
            bits in ``flags`` are set, e.g. ``flags=giglib.NODE_IS_SAMPLE``
            returns sample nodes.
        :param time: If not ``None``, only return nodes at this time. This can be
            a single number, in which case node times must be exactly equal to this
            value, or a ``(start, end)`` pair, in which case node times must satisfy
            ``start <= time < end``.
        :param numpy.ndarray candidates: If not ``None``, an array of node IDs
            to filter, rather than all the nodes in the table (mainly for internal use).
        :return: An array of matching node IDs, in the same order as ``candidates``
            (i.e. increasing order, if ``candidates`` is ``None``)
        :rtype: numpy.ndarray
        """
        ids = np.arange(len(self), dtype=np.int64) if candidates is None else np.asarray(candidates)
        if flags is not None:
            ids = ids[(self.flags[ids] & flags) == flags]
        if time is not None:
            times = self.time[ids]
            if np.ndim(time) == 0:
                ids = ids[times == time]
            else:
                start, end = time
                if start > end:
                    raise ValueError("time interval must have start <= end")
                ids = ids[np.logical_and(times >= start, times < end)]
        return ids


class IndividualTable(BaseExtraTable):
    _RowClass = IndividualTableRow
    _non_int64_fieldtypes = types.MappingProxyType({"flags": np.uint32})
    _extra_names = ("location", "parents", "metadata")

    @property
    def flags(self) -> npt.NDArray[np.uint32]:
        return self._column("flags")

    @property
    def parents(self):
        idx = self._extra_data_cols["parents"]
        return [row[idx] for row in self._extra_data]

    def add_row(self, *, flags=None, location=None, parents=None, metadata=None) -> int:
        """
        Add an individual to the table.

        :param int flags: The flags for this individual (default: ``None``, treated as 0)
        :param location: The location of this individual, as a sequence of numbers
            (default: ``None``). This is stored as a tuple of floats.
        :param parents: The IDs of the parents of this individual, as a sequence of
            integers (default: ``None``). This is stored as a tuple of ints.
        :param metadata: Any metadata for this individual (default: ``None``)
        :return: The ID of the newly added individual
        :rtype: int
        """
        if flags is None:
            flags = 0
        # Store sequences (e.g. numpy arrays from tskit) as tuples of Python numbers, so
        # that rows compare equal by value, and print nicely
        if location is not None:
            location = tuple(float(x) for x in location)
        if parents is not None:
            parents = tuple(int(x) for x in parents)
        num_rows = len(self._data)
        if len(self._datastore) == num_rows:
            self._expand_datastore()
        self._datastore[num_rows] = (flags,)
        self._extra_data.append([location, parents, metadata])
        self._data = self._datastore[0 : len(self._data) + 1]
        return num_rows

    def append(self, obj) -> int:
        return self.add_row(**{k: v for k, v in self.to_dict(obj).items() if k in self._RowClass._fields})


class MutationTable(BaseExtraTable):
    """
    A table of mutations. Each mutation lies above a node, at a single position in
    that node's own coordinates (there is no sites table). See
    :ref:`sec_design_mutations` for the design.

    Mutations for the same node must be adjacent in the table, and ordered by
    chromosome then position, for :meth:`ids_for_node` to work: :meth:`~giglib.Tables.sort`
    puts mutations into this order. Mutations at the same node, chromosome and
    position are assumed to have occurred in table order.
    """

    _RowClass = MutationTableRow
    _non_int64_fieldtypes = types.MappingProxyType({"chromosome": np.int16, "time": np.float64})
    _extra_names = ("derived_state", "metadata")

    def __init__(self, initial_size=None):
        super().__init__(initial_size=initial_size)
        self._reset_index()

    def _reset_index(self):
        # As for the iedges table, maintain an index of {node: {chromosome: [start, end]}}
        # row ranges, updated as rows are added. This is only meaningful if mutations
        # are grouped by node and ordered by chromosome then position, which is
        # recorded in ``_node_ranges_valid``.
        self._id_range_for_node = {}
        self._node_ranges_valid = True

    def _update_index(self, node, chromosome, position, row):
        ranges = self._id_range_for_node
        if row > 0:
            prev = self._data[row - 1]
            prev_node = prev["node"]
            if prev_node == node:
                prev_chrom = prev["chromosome"]
                if chromosome < prev_chrom or (chromosome == prev_chrom and position < prev["position"]):
                    self._node_ranges_valid = False
            elif node in ranges:
                self._node_ranges_valid = False  # mutations for this node are not adjacent
        chrom_ranges = ranges.setdefault(node, {})
        if chromosome in chrom_ranges:
            chrom_ranges[chromosome][1] = row + 1
        else:
            chrom_ranges[chromosome] = [row, row + 1]

    def _rebuild_index(self):
        self._reset_index()
        for row, (node, chromosome, position) in enumerate(
            zip(self.node.tolist(), self.chromosome.tolist(), self.position.tolist())
        ):
            self._update_index(node, chromosome, position, row)

    # define each property by hand, for speed
    @property
    def node(self):
        return self._column("node")

    @property
    def chromosome(self):
        return self._column("chromosome")

    @property
    def position(self):
        return self._column("position")

    @property
    def time(self) -> npt.NDArray[np.float64]:
        return self._column("time")

    @property
    def derived_state(self):
        idx = self._extra_data_cols["derived_state"]
        return [row[idx] for row in self._extra_data]

    def add_row(self, node, position, derived_state, *, chromosome=0, time=None, metadata=None) -> int:
        """
        Add a mutation to the table.

        :param int node: The ID of the node above which the mutation occurs
        :param int position: The position of the mutation, in the coordinates of
            ``node`` for the given ``chromosome``
        :param str derived_state: The allele produced by the mutation, which replaces
            the base at ``position``. This is usually a single character, but can be
            any string, e.g. to represent a small indel (see :ref:`sec_design_mutations`)
        :param int chromosome: The chromosome in ``node``'s genome (default: 0)
        :param float time: The time of the mutation. If ``None`` (default), the time is
            :data:`~giglib.UNKNOWN_TIME`.
        :param metadata: Any metadata for the mutation
        :return: The ID of the newly added mutation
        :rtype: int
        """
        if not isinstance(derived_state, str):
            raise TypeError(f"derived_state must be a string, not {derived_state!r}")
        if time is None:
            time = Const.UNKNOWN_TIME
        num_rows = len(self._data)
        if len(self._datastore) == num_rows:
            self._expand_datastore()
        self._datastore[num_rows] = (node, position, chromosome, time)
        self._extra_data.append([derived_state, metadata])
        self._data = self._datastore[0 : num_rows + 1]
        self._update_index(int(node), int(chromosome), position, num_rows)
        return num_rows

    def append(self, obj) -> int:
        return self.add_row(**{k: v for k, v in self.to_dict(obj).items() if k in self._RowClass._fields})

    def clear(self):
        super().clear()
        self._reset_index()

    def copy(self):
        copy = super().copy()
        copy._rebuild_index()
        return copy

    def __eq__(self, other):
        # Unknown times are NaNs, so must be compared specially
        if len(self) != len(other):
            return False
        for name in self._data.dtype.names:
            if not np.array_equal(self._data[name], other._data[name], equal_nan=(name == "time")):
                return False
        return self._extra_data == other._extra_data

    def _reorder(self, order):
        """
        Reorder the rows of the table using an array of row indexes
        """
        self._subset(order)

    def _subset(self, ids):
        """
        Replace the rows of the table with the rows with the given IDs, in that order
        """
        ids = np.asarray(ids, dtype=np.int64)
        self._datastore[0 : len(ids)] = self._data[ids]  # indexing makes a copy
        self._data = self._datastore[0 : len(ids)]
        self._extra_data = [self._extra_data[i] for i in ids]
        self._rebuild_index()

    def ids_for_node(self, u, *, chromosome=None):
        """
        Return the IDs of the mutations above node ``u``, ordered by chromosome then
        position (and then by the order in which they occurred).

        :param int u: The node ID
        :param int chromosome: If not ``None``, only return mutations on this chromosome
        :return: A numpy array of mutation IDs
        :raises ValueError: If the mutations for a node are not adjacent in the table,
            or are not ordered by chromosome then position (use :meth:`~giglib.Tables.sort`)
        """
        return np.arange(*self._range_for_node(u, chromosome))

    def _range_for_node(self, u, chromosome=None):
        """
        Return the (start, end) range of IDs of the mutations above node ``u`` (on the
        given chromosome, if not ``None``), without creating an array of IDs. The range
        is (0, 0) if there are no such mutations.
        """
        if not self._node_ranges_valid:
            raise ValueError(
                "Mutations for the same node must be adjacent in the table, and ordered by "
                "chromosome, then position: call tables.sort() first"
            )
        ranges = self._id_range_for_node.get(u, {})
        if chromosome is None:
            if len(ranges) == 0:
                return (0, 0)
            # chromosome ranges are adjacent and in increasing chromosome order
            return (next(iter(ranges.values()))[0], next(reversed(ranges.values()))[1])
        return tuple(ranges.get(chromosome, (0, 0)))

    def _chromosomes_for_node(self, u):
        """
        Return the chromosome IDs of the mutations above node ``u``, as a tuple in
        ascending order
        """
        self._range_for_node(u)  # (checks that the index is valid)
        return tuple(self._id_range_for_node.get(u, {}))

    def _max_pos_for_node(self, u, *, chromosome=None):
        """
        Return one more than the highest position of the mutations above node ``u``, on
        the given chromosome (or on any chromosome, if ``None``), or None if there are no
        such mutations. Like the maximum position of an iedge, this is an exclusive bound.
        """
        self._range_for_node(u)  # (checks that the index is valid)
        ranges = self._id_range_for_node.get(u, {})
        if chromosome is not None:
            ranges = {chromosome: ranges[chromosome]} if chromosome in ranges else {}
        if len(ranges) == 0:
            return None
        # Mutations are ordered by position within each node and chromosome
        return max(int(self.position[end - 1]) for _, end in ranges.values()) + 1


TableClasses = collections.namedtuple("TableClasses", "nodes iedges individuals mutations")


class Tables:
    """
    A group of tables which can describe a Genetic Inheritance Graph (GIG). This
    class is intentionally similar to a *tskit* :class:`~tskit:tskit.TableCollection`.
    """

    _frozen = False

    table_classes = TableClasses(NodeTable, IEdgeTable, IndividualTable, MutationTable)

    def __init__(
        self,
        time_units=None,
        *,
        initial_sizes=None,
    ):
        """
        :param str time_units: The units of time used, e.g. "generations" or "years"
        :param dict initial_sizes: A dictionary of the initial number of rows
            expected for each table,
            e.g. ``{"nodes": 1000, "iedges": 1000, "mutations": 1000}``. Missing
            keys will be set to the default ``initial_size`` for that type of table.
        """
        if initial_sizes is None:
            initial_sizes = {}
        for name, cls in zip(self.table_classes._fields, self.table_classes):
            setattr(self, name, cls(initial_size=initial_sizes.get(name, None)))
        self.time_units = "unknown" if time_units is None else time_units
        #: A dict mapping alleles to their complements, used when decoding alleles
        #: that are inherited through an inversion (see
        #: :ref:`sec_design_mutations_strands`). Defaults to the IUPAC nucleotide
        #: codes plus ``"-"``. Set to ``{}`` to turn complementing off.
        self.allele_complements = dict(DEFAULT_ALLELE_COMPLEMENTS)

    def __eq__(self, other) -> bool:
        for name in self.table_classes._fields:
            if getattr(self, name) != getattr(other, name):
                return False
        if self.time_units != other.time_units:
            return False
        if dict(self.allele_complements) != dict(other.allele_complements):
            return False
        return True

    def _repr_html_(self):
        """
        Called e.g. by jupyter notebooks to render tables
        """
        html = ""
        for name in self.table_classes._fields:
            html += f"<h3>{name}</h3>"
            html += getattr(self, name)._repr_html_()
        return html

    def freeze(self):
        """
        Freeze all tables so they cannot be modified
        """
        for name in self.table_classes._fields:
            getattr(self, name).freeze()
        self.allele_complements = types.MappingProxyType(dict(self.allele_complements))
        self._frozen = True

    def __setattr__(self, attr, value):
        if self._frozen:
            raise AttributeError("Trying to set attribute on a frozen (read-only) instance")
        return super().__setattr__(attr, value)

    def clear(self):
        """
        Clear all tables
        """
        for name in self.table_classes._fields:
            getattr(self, name).clear()

    def copy(self):
        """
        Return an unfrozen deep copy of the tables.

        :return: A deep copy of the tables
        :rtype: giglib.Tables
        """
        copy = self.__class__()
        for name in self.table_classes._fields:
            setattr(copy, name, getattr(self, name).copy())
        # The copied node table has a new version, but has the same rows, so iedge flags
        # checked against this node table also hold for the copy
        if self.iedges._nodes_version == self.nodes._version:
            copy.iedges._nodes_version = copy.nodes._version
        copy.time_units = self.time_units
        copy.allele_complements = dict(self.allele_complements)
        return copy

    def __str__(self):
        # To do: make this look nicer
        return "\n\n".join(
            [
                "== NODES ==\n" + str(self.nodes),
                "== I-EDGES ==\n" + str(self.iedges),
                "== MUTATIONS ==\n" + str(self.mutations),
            ]
        )

    def sort(self, shrink_memory=False):
        """
        Sort the tables in place. This affects the iedges and mutations tables.
        Iedges are grouped by child, in order of decreasing child time (then
        increasing child ID), and within each child are ordered by chromosome then
        left position. Mutations are grouped by node in the same way, and within each
        node are ordered by chromosome then position. The sort is stable, so mutations
        at the same node, chromosome and position keep their existing order (which
        defines the order in which they occurred).
        """
        self._sort_mutations()
        # index to sort edges so they are sorted by child time
        # and grouped by child id. Within each group with the same child ID,
        # the edges are sorted by child_left
        iedges = self.iedges
        if len(iedges) == 0:
            return
        edge_order = np.lexsort(
            (
                iedges.child_left,
                iedges.child_chromosome,
                iedges.child,
                -self.nodes.time[iedges.child],  # Primary key
            )
        )
        if shrink_memory:
            iedges._datastore = iedges._data.copy()[edge_order]
            iedges._data = iedges._datastore[:]
        else:
            iedges._datastore[0 : len(iedges)] = iedges._data[edge_order]
        iedges._id_range_for_child = {}
        for i, (c, c_chr) in enumerate(zip(iedges.child.tolist(), iedges.child_chromosome.tolist())):
            iedges._increment_id_range_for_child(c, c_chr, i)

        # The sort order uses the current node times. Other node-time flags may be out
        # of date if the node table has changed, so they are removed.
        if iedges._nodes_version != self.nodes._version:
            iedges.unset_bitflag(ValidFlags.IEDGES_COMBO_NODE_TABLE)
        iedges.set_bitflag(ValidFlags.IEDGES_SORTED)
        iedges._nodes_version = self.nodes._version

    def _mutation_sort_order(self):
        """
        Return the (stable) order of mutations, grouped by node in order of decreasing
        node time then increasing node ID, then by chromosome, then by position
        """
        mutations = self.mutations
        node = mutations.node
        if len(node) > 0 and (node.min() < 0 or node.max() >= len(self.nodes)):
            raise ValueError("Mutations refer to nodes that are not in the node table")
        return np.lexsort((mutations.position, mutations.chromosome, node, -self.nodes.time[node]))

    def _sort_mutations(self):
        if len(self.mutations) > 0:
            self.mutations._reorder(self._mutation_sort_order())

    @property
    def edges(self):
        """
        Not part of the giglib API (emits a :class:`~giglib.TskitNameWarning`): use
        ``tables.iedges``
        """
        warn_tskit_name("edges", "tables.iedges", obj="tables", tskit_obj="tables")
        return self.iedges

    def graph(self):
        """
        Return a genetic inheritance graph (Graph) object based on these tables. This
        requires the tables to satisfy all the validity requirements of a GIG
        (:data:`~giglib.ValidFlags.GIG`). In summary:

            * Iedges have non-negative integer node IDs and coordinates, and child and
              parent intervals of the same length, with ``child_left < child_right``.
            * Each position of a child's chromosome is covered by at most one iedge
              (even if the iedges share a parent), and every parent is older than
              its child.
            * Iedges with the same ``edge`` ID (if not ``NULL``) have the same child and
              parent.
            * The iedges are in the canonical order: grouped by child, with children
              ordered by time (oldest first) and then ID, and each child's iedges
              ordered by chromosome and then position. Calling :meth:`sort` first
              gives this order.

        The nodes are not required to be in any particular order (e.g. by time). The
        requirements, and their flags, are described in full in :ref:`sec_validation`.

        Requirements whose flags are already set are trusted (apart from those that
        depend on node times, which are always rechecked), and the others are checked
        (and their flags set) using :meth:`validate`.

        :return: A genetic inheritance graph object
        :rtype: giglib.Graph

        """
        from .graph import Graph

        return Graph(self)

    def validate(self, flags=None, *, recheck=False):
        """
        Check that the iedges satisfy the given validity requirements, and record each
        one that is satisfied by setting its flag (see :ref:`sec_validation`). An error
        is raised for the first requirement that is not satisfied.

        Validation is incremental: requirements that were already satisfied when the
        tables were last validated are only checked for the rows added since then, so
        validating after adding a few rows is cheap, however large the tables are. The
        exceptions are requirements that depend on node times, which are checked for
        all the rows if the node table has been changed (other than by adding rows)
        since they were checked. Flags set directly with
        :meth:`~tables.IEdgeTable.set_bitflag` are trusted in the same way, without
        being checked.

        :param int flags: The :class:`~giglib.ValidFlags` to check, combined using ``|``
            (default: ``None``, meaning all the requirements of a GIG, i.e.
            :data:`~giglib.ValidFlags.GIG`)
        :param bool recheck: If ``True``, check all the requirements in ``flags`` for
            all the rows, even if they have been checked before. This is mainly intended
            for testing. Default: ``False``.
        :raises ValueError: If a requirement is not satisfied
        """
        if flags is None:
            flags = ValidFlags.GIG
        iedges = self.iedges
        num_rows = len(iedges)
        self._recheck_node_flags(flags)
        # The requirements known to hold for the validated rows (those before `start`)
        known, start = iedges._flags, iedges._num_validated
        if recheck:
            self._check_iedges(flags)
        else:
            self._check_iedges(flags & ~known)  # all rows
            self._check_iedges(flags & known, start=start)  # only the new rows
        # Requirements that were not checked only hold for the whole table if there
        # are no new rows
        new = (ValidFlags(flags), num_rows, self.nodes._version)
        if start == num_rows:
            new = (new[0] | known, *new[1:])
        current = (iedges._flags, iedges._num_validated, iedges._nodes_version)
        if current != new:
            if self._frozen and current[:2] == new[:2]:
                return  # only the node table version would change, which needn't be recorded
            iedges._flags, iedges._num_validated, iedges._nodes_version = new  # (fails if frozen)

    def _recheck_node_flags(self, flags):
        """
        If the node table has changed (other than by adding rows) since the iedge
        requirements that depend on node times were checked, check those in ``flags``
        again for the validated rows, raising a ValueError if they no longer hold. The
        others that depend on node times are removed.
        """
        iedges = self.iedges
        if iedges._nodes_version == self.nodes._version:
            return
        node_flags = ValidFlags.IEDGES_COMBO_NODE_TABLE
        recheck = flags & iedges._flags & node_flags
        self._check_iedges(recheck, stop=iedges._num_validated)
        if self._frozen:
            return  # (the node table of frozen tables can't change, so no need to record this)
        iedges._flags &= ~(node_flags & ~recheck)
        iedges._nodes_version = self.nodes._version

    def _keep_iedge_flags(self, flags):
        """
        Set the given iedge flags (e.g. those kept by a table algorithm), which must
        already have been checked against the current node table
        """
        self.iedges.set_bitflag(flags)
        self.iedges._nodes_version = self.nodes._version

    def _order_independent_flags(self):
        """
        Return the order-independent iedge flags of these tables, for keeping on iedges
        that are rebuilt from the current ones (e.g. by trimming them, or removing some,
        with node times unchanged). Those that depend on node times are rechecked first
        if the node table has changed, raising an error if they no longer hold.
        """
        flags = self.iedges.flags & ValidFlags.IEDGES_ORDER_INDEPENDENT
        self._recheck_node_flags(flags)
        return flags

    def _require_iedge_flags(self, flags, method, *, validated_rows=False):
        """
        Raise a ValueError naming the missing flags, if the iedges table does not have
        all of the given validity flags. If ``validated_rows`` is True, it is enough for
        them to hold for the validated rows (in which case the method must only read
        those rows, e.g. by using :meth:`IEdgeTable.ids_for_child`). Flags that depend on
        node times are rechecked first if the node table has changed since they were
        checked.
        """
        self._recheck_node_flags(flags)
        have = self.iedges._flags if validated_rows else self.iedges.flags
        missing = ValidFlags(flags & ~have)
        if missing:
            names = ", ".join(flag.name for flag in ValidFlags if flag in missing and flag.bit_count() == 1)
            raise ValueError(
                f"{method} requires the iedges to have these validity flags, which are not set: {names}. "
                "Use tables.validate() to check and set them (see the Validation page of the docs)."
            )

    def _check_iedges(self, flags, start=0, stop=None):
        """
        Raise a ValueError if the rows of the iedges table from ``start`` onwards (up to,
        but not including, ``stop``, if given) do not satisfy each of the requirements in
        ``flags``, given that the earlier rows do.
        This is the only implementation of the validity rules. Checks are made on whole
        columns, in an order that reports the most fundamental problem first.

        Most rules only involve each row, or each row and the previous one, so only the
        new rows (and the one before them) are read. The exceptions are
        :data:`~giglib.ValidFlags.IEDGES_FOR_CHILD_ADJACENT` and
        :data:`~giglib.ValidFlags.IEDGES_FOR_CHILD_NONOVERLAPPING`, which use the index
        of rows for each child, and :data:`~giglib.ValidFlags.IEDGES_SAME_PARENT_CHILD_FOR_EDGE`,
        which looks at any earlier rows with the same edge IDs as the new ones.
        """
        iedges = self.iedges
        stop = len(iedges) if stop is None else stop
        if flags == 0 or start >= stop:
            return
        flags = int(flags)  # (testing bits of an int is much faster than of a ValidFlags)
        # Rules that compare each row with the previous one also need the row before
        # `start`. Columns are read from `prev`, and new rows start at index `first`.
        prev = max(start - 1, 0)
        first = start - prev

        def column(name):
            return getattr(iedges, name)[prev:stop]

        child, parent = column("child"), column("parent")
        child_left, child_right = column("child_left"), column("child_right")
        parent_left, parent_right = column("parent_left"), column("parent_right")
        child_chromosome = column("child_chromosome")
        needs_times = _BITS.IEDGES_COMBO_NODE_TABLE

        def fail_if(bad, message, row_offset=start):
            # `bad` is a boolean array, whose first element refers to row `row_offset`
            if bad.any():
                bad = np.flatnonzero(bad) + row_offset
                raise ValueError(message.format(first=int(bad[0]), all=bad))

        def first_row_of(u):
            # The first row of child u, from the index of rows for each child
            return min(first for first, _ in iedges._id_range_for_child[u].values())

        # Per-row rules: check the new rows only
        if flags & needs_times:
            # Node IDs must refer to nodes (guaranteed by IEDGES_PARENT_OLDER_THAN_CHILD)
            for name, nodes in (("child", child[first:]), ("parent", parent[first:])):
                fail_if(nodes < 0, f"Iedge {{first}} contains negative {name} ID")
                fail_if(nodes >= len(self.nodes), f"Iedge {{first}} contains {name} ID >= num nodes")
        if flags & _BITS.IEDGES_INTEGERS:
            # Values are always integers (add_row refuses others), so only
            # non-negativity needs checking. Chromosome IDs may be negative (e.g. for
            # circular chromosomes), as may the edge ID (NULL).
            for name, values in (
                ("child_left", child_left),
                ("child_right", child_right),
                ("parent_left", parent_left),
                ("parent_right", parent_right),
            ):
                fail_if(values[first:] < 0, f"Iedge {{first}} has a negative {name}")
        times = self.nodes.time
        if flags & _BITS.IEDGES_PARENT_OLDER_THAN_CHILD:
            # (written so that NaN times, and so self-edges with NaN times, also fail)
            bad = np.flatnonzero(~(times[parent[first:]] > times[child[first:]]))
            if len(bad) > 0:
                i = int(bad[0]) + start
                raise ValueError(f"Edge {i}: parent node {iedges.parent[i]} not older than child {iedges.child[i]}")
        if flags & _BITS.IEDGES_INTERVALS:

            def span(a, b):
                # The length of an interval, which can be up to 2**64 - 1 if the ends are
                # not checked to be non-negative: the int64 difference may wrap around,
                # but is then correct as a uint64
                return (np.maximum(a, b) - np.minimum(a, b)).astype(np.uint64)

            fail_if(
                span(parent_left[first:], parent_right[first:]) != span(child_left[first:], child_right[first:]),
                "iedges {all} have different parent and child spans",
            )
        if flags & _BITS.IEDGES_CHILD_INTERVAL_POSITIVE:
            fail_if(child_left[first:] >= child_right[first:], "child_left >= child_right for iedges {all}")
        if flags & _BITS.IEDGES_SAME_PARENT_CHILD_FOR_EDGE:
            edge = iedges.edge[:stop]
            new_ids = np.unique(edge[start:])
            new_ids = new_ids[new_ids != NULL]
            if len(new_ids) > 0:
                rows = np.isin(edge, new_ids)  # including any earlier rows with these IDs
                combos = np.unique(
                    np.column_stack((edge[rows], iedges.child[:stop][rows], iedges.parent[:stop][rows])), axis=0
                )
                if len(combos) != len(new_ids):
                    raise ValueError("iedges with the same edge ID have different parent/child combinations")

        # Ordering rules: compare each row with the previous one (row_offset=prev + 1).
        # Values are compared directly, as differences could overflow.
        same_child = child[1:] == child[:-1]
        same_chromosome = same_child & (child_chromosome[1:] == child_chromosome[:-1])
        if flags & (_BITS.IEDGES_PRIMARY_ORDER_CHILD_TIME_DESC | _BITS.IEDGES_SECONDARY_ORDER_CHILD_ID_ASC):
            child_time = times[child]
        if flags & _BITS.IEDGES_PRIMARY_ORDER_CHILD_TIME_DESC:
            fail_if(child_time[1:] > child_time[:-1], "iedges are not sorted by child time, descending", prev + 1)
        if flags & _BITS.IEDGES_SECONDARY_ORDER_CHILD_ID_ASC:
            fail_if(
                (child_time[1:] == child_time[:-1]) & (child[1:] < child[:-1]),
                "iedges are not sorted by child time (desc) then ID (asc)",
                prev + 1,
            )
        if flags & _BITS.IEDGES_FOR_CHILD_ADJACENT:
            message = "iedges for each child are not adjacent"
            if np.count_nonzero(~same_child) + 1 != len(np.unique(child)):
                raise ValueError(message)
            if start > 0:
                # The children of the new rows (other than that of the row before them)
                # must not have earlier rows
                for u in set(np.unique(child[first:]).tolist()) - {int(child[0])}:
                    if first_row_of(u) < start:
                        raise ValueError(message)
        if flags & _BITS.IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC:
            fail_if(
                same_child & (child_chromosome[1:] < child_chromosome[:-1]),
                "iedges for a given child are not sorted by chromosome",
                prev + 1,
            )
        if flags & _BITS.IEDGES_FOR_CHILD_SECONDARY_ORDER_LEFT_ASC:
            fail_if(
                same_chromosome & (child_left[1:] <= child_left[:-1]),
                "iedges for a given child/chr are not sorted by left pos",
                prev + 1,
            )
        sorted_within_child = _BITS.IEDGES_WITHIN_CHILD_SORTED | _BITS.IEDGES_CHILD_INTERVAL_POSITIVE
        if flags & _BITS.IEDGES_FOR_CHILD_NONOVERLAPPING and (flags & sorted_within_child) == sorted_within_child:
            # The iedges of each child are (just checked to be) sorted by position, with
            # positive spans, so (given that the earlier rows don't overlap) they don't
            # overlap if each starts at or after the end of the previous one
            bad = np.flatnonzero(same_chromosome & (child_left[1:] < child_right[:-1]))
            if len(bad) > 0:
                i = prev + 1 + int(bad[0])
                raise ValueError(
                    f"Node {iedges.child[i]} has multiple or duplicate parents at position {iedges.child_left[i]}"
                )
        elif flags & _BITS.IEDGES_FOR_CHILD_NONOVERLAPPING:
            # Check all the rows of the children of the new rows
            if start == 0:
                rows = np.arange(stop)
            elif flags & _BITS.IEDGES_FOR_CHILD_ADJACENT:
                # (just checked) the rows of each child are adjacent
                rows = np.arange(first_row_of(int(child[0])), stop)
            else:
                rows = np.flatnonzero(np.isin(iedges.child[:stop], child[first:]))
            r_child, r_chromosome = iedges.child[rows], iedges.child_chromosome[rows]
            r_left, r_right = iedges.child_left[rows], iedges.child_right[rows]
            # Sort by child, chromosome and left position, so this works whatever the
            # order of the iedges
            order = np.lexsort((r_left, r_chromosome, r_child))
            same = (np.diff(r_child[order]) == 0) & (np.diff(r_chromosome[order]) == 0)
            overlap = same & (r_left[order][1:] < r_right[order][:-1])
            bad = np.flatnonzero(overlap)
            if len(bad) > 0:
                i = rows[order[bad[0] + 1]]
                raise ValueError(
                    f"Node {iedges.child[i]} has multiple or duplicate parents at position {iedges.child_left[i]}"
                )

    def filter_nodes(self, flags=None, time=None):
        """
        Return the IDs of nodes matching the given criteria. Like other ``Tables``
        accessors, this returns IDs: to get :class:`~giglib.graph.Node` objects
        instead, use :meth:`Graph.filter_nodes`. See :meth:`~giglib.tables.NodeTable.filter`
        for the meaning of the parameters.

        :rtype: numpy.ndarray
        """
        return self.nodes.filter(flags=flags, time=time)

    def sample_nodes(self, time=None):
        """
        Return the IDs of sample nodes. This is an alias for
        ``filter_nodes(flags=giglib.NODE_IS_SAMPLE, time=time)``.

        :rtype: numpy.ndarray
        """
        return self.filter_nodes(flags=NODE_IS_SAMPLE, time=time)

    def change_times(self, timedelta):
        """
        Add a value to all times (nodes and mutations)
        """
        self.nodes._data["time"] += timedelta
        self.nodes._changed()
        known = ~is_unknown_time(self.mutations.time)
        self.mutations._data["time"][known] += timedelta

    def decapitate(self, time, *, discard_old_mutations=False):
        """
        Remove nodes that are older than a certain time, and edges that have those
        as parents. Also remove individuals associated with those nodes.

        Mutations above removed nodes, and mutations whose (known) time is at least
        ``time``, are removed. Since a GIG has no per-site ancestral states (see
        :ref:`sec_design_mutations`), information about alleles would then be lost.
        So by default, wherever a remaining node loses its parent or its own old
        mutations, mutations with unknown time are added above that node, recording the
        alleles it inherited from above at those positions (and any mutations at that
        node and position that came before the removed ones are also removed). This
        means that the haplotypes of all remaining nodes are unchanged by decapitation.

        :param float time: The time at or before which nodes should be removed
        :param bool discard_old_mutations: If False (default), the alleles produced by
            mutations at least as old as ``time`` are recorded by new mutations above
            the nodes that lose their parents, as described above, so that haplotypes
            are unchanged. If True, these old mutations are simply discarded, as in
            :meth:`tskit:tskit.TreeSequence.decapitate`.

            .. warning::
                With ``discard_old_mutations=True``, ancestral states (which are given
                by mutations above roots) are removed along with the roots, so
                positions whose alleles were only defined by older mutations become
                missing data, and haplotypes may change.

        :return: An array mapping old node ids to new node ids (or -1 if removed)
        :rtype: numpy.ndarray

        """
        replacements, superseded = {}, []
        if not discard_old_mutations and len(self.mutations) > 0:
            replacements, superseded = self._alleles_inherited_across(time)
        # The kept iedges are copies of old ones, between nodes whose times are unchanged,
        # so keep the guarantees that don't depend on the order of the iedges (sorting
        # gives those that do)
        kept_flags = self._order_independent_flags()
        individuals = IndividualTable()
        individual_map = {NULL: NULL}
        nodes = NodeTable()
        node_map = np.full(len(self.nodes), NULL, dtype=np.int64)
        iedges = IEdgeTable()
        for u, nd in enumerate(self.nodes):
            if nd.time < time:
                i = nd.individual
                if i not in individual_map:  # NB: NULL (no individual) is already mapped
                    parents = self.individuals[i].parents
                    individual_map[i] = individuals.add_row(
                        parents=tuple(individual_map.get(j, NULL) for j in (parents if parents is not None else ()))
                    )
                node_map[u] = nodes.add_row(time=nd.time, flags=nd.flags, individual=individual_map[i])
        for ie in self.iedges:
            if node_map[ie.parent] != NULL and node_map[ie.child] != NULL:
                iedges.add_row(
                    node_map[ie.child],
                    node_map[ie.parent],
                    child_left=ie.child_left,
                    child_right=ie.child_right,
                    parent_left=ie.parent_left,
                    parent_right=ie.parent_right,
                    child_chromosome=ie.child_chromosome,
                    parent_chromosome=ie.parent_chromosome,
                )
        self.nodes = nodes
        self.iedges = iedges
        self.individuals = individuals

        old_mutations = self.mutations
        mutations = MutationTable()
        # Replacements come first, so that they precede (i.e. are older than) any
        # retained mutations at the same node and position after the (stable) sort
        for (u, chromosome, position), allele in sorted(replacements.items()):
            mutations.add_row(node_map[u], position, allele, chromosome=chromosome)
        keep = (node_map[old_mutations.node] != NULL) & ~(old_mutations.time >= time)  # unknown times are NaN
        keep[superseded] = False
        for i in np.flatnonzero(keep):
            m = old_mutations[i]
            mutations.add_row(
                node_map[m.node],
                m.position,
                m.derived_state,
                chromosome=m.chromosome,
                time=None if is_unknown_time(m.time) else m.time,
                metadata=m.metadata,
            )
        self.mutations = mutations
        self.sort()
        self._keep_iedge_flags(kept_flags)
        return node_map

    def _alleles_inherited_across(self, time):
        """
        Work out the mutations needed by :meth:`decapitate` to keep the haplotypes of
        nodes younger than ``time`` unchanged. Requires the tables to form a valid GIG.

        Returns a tuple ``(replacements, superseded)``. ``replacements`` is a dict
        mapping ``(node, chromosome, position)`` to the allele that should be recorded
        by a new (unknown-time) mutation above that node, to precede any mutations that
        are kept at that node and position. These record:

        1. The effect of mutations at least as old as ``time`` above nodes younger than
           ``time`` (these mutations are removed by decapitation). Only the last such
           mutation at a node and position is needed.
        2. Otherwise, the alleles inherited through iedges from parents at least as old
           as ``time`` (these iedges are removed by decapitation). Missing alleles are
           omitted.

        ``superseded`` is a list of the IDs of mutations that precede (in table order)
        the last removed mutation at the same node and position. Their effect is
        overwritten by that mutation, so they should also be removed, as otherwise they
        would follow the replacement.
        """
        from .graph import Graph

        node_time = self.nodes.time
        complements = self.allele_complements
        mutations = self.mutations
        replacements = {}
        last_removed = {}
        removed = (mutations.time >= time) & (node_time[mutations.node] < time)
        for i in np.flatnonzero(removed):  # in table order, so the last one wins
            m = mutations[i]
            key = (m.node, m.chromosome, m.position)
            last_removed[key] = i
            replacements[key] = m.derived_state
        superseded = []
        if len(last_removed) > 0:
            nodes = np.unique([key[0] for key in last_removed])
            for j in np.flatnonzero(np.isin(mutations.node, nodes)):
                key = (mutations.node[j], mutations.chromosome[j], mutations.position[j])
                if j < last_removed.get(key, -1):
                    superseded.append(j)

        gig = Graph(self.copy())
        for ie in gig.iedges:
            if node_time[ie.child] >= time or node_time[ie.parent] < time:
                continue
            known = gig._known_alleles(ie.parent, ie.parent_chromosome, ie.parent_min, ie.parent_max)
            to_child = PositionMap.from_iedge(ie).inverse()
            for x, allele in known.items():
                if to_child.is_reversing:
                    allele = complement_allele(allele, complements)
                replacements.setdefault((ie.child, ie.child_chromosome, to_child.position(x)), allele)
        return replacements, superseded

    @classmethod
    def from_tree_sequence(cls, ts, *, chromosome=None, **kwargs):
        """
        Create a Tables object from a :class:`tskit:tskit.TreeSequence`. To create a GIG
        directly, use :func:`Graph.from_tree_sequence` which simply wraps this method.

        Sites and mutations are converted as described in :ref:`sec_design_mutations`.
        Each mutation is placed above the same node, at the position of its site.
        Since a GIG has no sites table, each site's ancestral state becomes a mutation
        with unknown time above each root of the local tree at that position. Isolated
        nodes (roots with no children) get no such mutation, so their state is missing
        data, matching the default treatment of isolated samples in tskit.

        Any reference sequence in the tree sequence is not converted. So unlike
        :meth:`tskit:tskit.TreeSequence.alignments`, haplotypes from the GIG have missing
        data (rather than reference bases) at positions without sites.

        :param tskit.TreeSequence ts: The tree sequence on which to base the Tables
            object. Site positions must be integers.
        :param int chromosome: The chromosome number to use for all interval edges
            and mutations (default: 0)
        :param kwargs: Other parameters passed to the Tables constructor
        :return: A Tables object representing the tree sequence
        :rtype: Tables
        """
        ts_tables = ts.tables
        tables = cls(
            time_units=ts.time_units,
            initial_sizes={
                "nodes": ts.num_nodes,
                "iedges": ts.num_edges,
                "individuals": ts.num_individuals,
            },
        )
        if ts_tables.migrations.num_rows > 0:
            raise NotImplementedError
        site_positions = ts_tables.sites.position
        if not np.all(np.floor(site_positions) == site_positions):
            raise ValueError("Site positions must be integers to convert to a GIG")
        if ts_tables.populations.num_rows > 1:
            # If there is only one population, ignore it
            raise NotImplementedError
        for row in ts_tables.nodes:
            obj = dataclasses.asdict(row)
            tables.nodes.append(obj)
        for row in ts_tables.edges:
            obj = dataclasses.asdict(row)
            if chromosome is not None:
                obj["parent_chromosome"] = obj["child_chromosome"] = chromosome
            tables.iedges.append(obj)  # (raises an error if coordinates are not integers)
        for row in ts_tables.individuals:
            obj = dataclasses.asdict(row)
            tables.individuals.append(obj)
        if ts.num_sites > 0:
            mut_chromosome = 0 if chromosome is None else chromosome
            for tree in ts.trees():
                if tree.num_sites == 0:
                    continue
                # Roots with children (excluding the virtual root at the end of the arrays)
                parent = tree.parent_array[:-1]
                num_children = tree.num_children_array[:-1]
                roots = np.flatnonzero((parent == NULL) & (num_children > 0))
                for site in tree.sites():
                    position = int(site.position)
                    # Ancestral states first: the (stable) sort keeps them before any
                    # other mutations at the same node and position
                    for root in roots:
                        tables.mutations.add_row(root, position, site.ancestral_state, chromosome=mut_chromosome)
                    for mut in site.mutations:
                        tables.mutations.add_row(
                            mut.node,
                            position,
                            mut.derived_state,
                            chromosome=mut_chromosome,
                            time=None if tskit.is_unknown_time(mut.time) else mut.time,
                            metadata=mut.metadata,
                        )
        tables.sort()
        # A valid tree sequence guarantees these properties of its edges (the
        # sort above guarantees the canonical iedge ordering flags)
        tables.iedges.set_bitflag(ValidFlags.IEDGES_FROM_TSKIT)
        return tables

    def sample_resolve(self, sort=True):
        """
        Sample resolve the Tables, keeping only those edge regions which
        transmit information to the current samples. This is rather
        like running the Hudson algorithm on a fixed graph, but without
        counting up the number of samples under each node. This requires
        the equivalent of making a new edges table, so is a costly operation.

        The algorithm is implemented by using a stack that contains intervals
        for each node, ordered by node time (oldest first). When considering a
        node, we pop the (youngest) node off the end of the stack, which
        ensures that we have collected all the intervals with that node as a
        parent, before passing inheritance information upwards.

        Nodes are not removed, so mutations stay above the same nodes. Mutations at
        positions that are not inherited by any sample are removed (mutations above
        sample nodes are always kept).
        """
        # == Implementation details ==
        # A. The intervals passed up to a node are collected in a list, and merged into
        #    a set of intervals (see giglib.intervals) when the node is visited, joining
        #    those that adjoin or overlap (coalesce). This reduces the number of
        #    intervals and their fragmentation as we go up the stack.
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
        self._require_iedge_flags(ValidFlags.IEDGES_FOR_TRACING, "sample_resolve()")

        stack = {c: collections.defaultdict(list) for c in reversed(np.argsort(self.nodes.time))}
        old_iedges = self.iedges
        # The new iedges are trimmed copies of the old ones, so keep the guarantees that
        # don't depend on the order of the iedges (sorting gives those that do)
        kept_flags = self._order_independent_flags()
        self.iedges = IEdgeTable()  # Make a new one
        # The regions of each (non-sample) node inherited by samples, for filtering mutations
        ancestral_regions = {}
        while len(stack):
            child, chromosomes = stack.popitem()
            chromosomes = {chrom: interval_sets.normalise(intervals) for chrom, intervals in chromosomes.items()}
            ancestral_regions[child] = chromosomes
            # An internal sample might have some material passed-up from its children
            # here, but we can replace that with everything passed up to parents
            if self.nodes[child].is_sample():
                chromosomes = {chrom: [(0, math.inf)] for chrom in old_iedges.chromosomes_as_child(child)}
            # We can't add in the edges in the correct order because we are adding
            # the youngest edges first, so we don't need to keep chromosomes in the
            # right order (we will have to sort anyway). If we *did* want to keep
            # chromosome order for edges, we could use sorted(chromosomes.keys())
            # here instead.
            for child_chr in chromosomes.keys():
                for ie_id in old_iedges.ids_for_child(child, chromosome=child_chr):
                    ie = old_iedges[ie_id]
                    parent = ie.parent
                    parent_chr = ie.parent_chromosome
                    to_parent = PositionMap.from_iedge(ie)
                    # JEROME'S NOTE (from previous algorithm): if we had an index here
                    # sorted by left coord we could binary search to first match, and
                    # could break once c_right > left (I think?), rather than having
                    # to intersect all the intervals with the current c_rgt/c_lft
                    for left, right in chromosomes[child_chr]:
                        # The part of this interval that the iedge passes to the parent
                        left, right = max(left, ie.child_left), min(right, ie.child_right)
                        if left < right:
                            p = to_parent.interval(left, right)
                            # For passing up, interval orientation doesn't matter, so
                            # put the lowest position first
                            stack[parent][parent_chr].append((min(p), max(p)))
                            # Add the trimmed interval to the new edge table
                            self.iedges.add_row(
                                child,
                                parent,
                                child_left=left,
                                child_right=right,
                                parent_left=p[0],
                                parent_right=p[1],
                                child_chromosome=child_chr,
                                parent_chromosome=parent_chr,
                            )
        self._remove_non_ancestral_mutations(ancestral_regions)
        if sort:
            self.sort()
        self._keep_iedge_flags(kept_flags)

    def _remove_non_ancestral_mutations(self, ancestral_regions):
        """
        Remove mutations that are not in the regions of their nodes that are inherited
        by samples. Mutations above sample nodes are always kept.
        """
        mutations = self.mutations
        if len(mutations) == 0:
            return
        is_sample = (self.nodes.flags & NODE_IS_SAMPLE) != 0
        keep = [
            i
            for i, (u, chrom, x) in enumerate(
                zip(mutations.node.tolist(), mutations.chromosome.tolist(), mutations.position.tolist())
            )
            if is_sample[u] or interval_sets.contains(ancestral_regions.get(u, {}).get(chrom, []), x)
        ]
        if len(keep) < len(mutations):
            mutations._subset(keep)

    def simplify(self, samples=None):
        """
        Simplify the tables in place, keeping only what is needed to describe the
        ancestry of the samples: the samples, and the branch points of the position tree
        (see :ref:`sec_design_simplification_branch_points`). Each position of a genome
        inherits from a single position of a single parent, so the lineages of the
        samples' positions form a tree. A node is kept in the regions where lineages of
        sample positions merge, either from different children, or through two iedges of
        the same child (a copy, e.g. the parent of a tandem duplication). Elsewhere it is
        bypassed, by composing the iedges above and below it, and a node that is not a
        branch point anywhere is removed. A node is also kept where a lineage ends without
        merging (at a root, or a node that doesn't inherit those positions), so that every
        sample position still has an iedge above it. This is a generalisation of _tskit_'s
        ``simplify()``, with ``keep_input_roots=True``.

        The ancestry of the samples is unchanged:

        * their haplotypes are the same, including missing data. A GIG has no record of
          genome lengths (:meth:`Graph.sequence_length` is the furthest position known from
          the iedges and mutations), so the one exception is a sample whose genome was
          only known to be longer from descendants that are removed: it can become shorter,
          losing only missing data;
        * their homology is the same, as found by :meth:`find_homology` in every mode
          (after relabelling the nodes);
        * simplifying again changes nothing.

        Mutations above a bypassed region of a node are moved down to the node below it
        that is kept (complemented if the path between them includes an odd number of
        inversions), and mutations at positions with no descendant samples are removed.
        The individuals table is unchanged.

        :param list samples: The node IDs of the samples to simplify with respect to. In
            the result, these are the first nodes, in this order, and are the only nodes
            flagged as samples. If None (default), use the current sample nodes.
        :return: For each node of the original tables, its ID in the simplified tables, or
            ``NULL`` if it was removed
        :rtype: numpy.ndarray
        """
        self._require_iedge_flags(ValidFlags.IEDGES_FOR_TRACING, "simplify()")
        if samples is None:
            samples = self.sample_nodes()
        old_nodes, old_iedges, old_mutations = self.nodes, self.iedges, self.mutations
        samples = [int(u) for u in samples]
        if len(set(samples)) != len(samples):
            raise ValueError("The samples must be distinct")
        for u in samples:
            if not 0 <= u < len(old_nodes):
                raise ValueError(f"Sample {u} is not a node ID")
        node_map = np.full(len(old_nodes), NULL, dtype=np.int64)
        nodes = NodeTable()

        def output_node(u, is_sample):
            # The ID of node u in the simplified tables, adding it if needed
            if node_map[u] == NULL:
                row = old_nodes[u]
                flags = row.flags | NODE_IS_SAMPLE if is_sample else row.flags & ~NODE_IS_SAMPLE
                node_map[u] = nodes.add_row(row.time, flags=flags, individual=row.individual, metadata=row.metadata)
            return int(node_map[u])

        for u in samples:
            output_node(u, is_sample=True)

        # The ancestral material at each node: for each chromosome of the node, a dict
        # mapping (output node, output chromosome, map) to a list of intervals of the
        # node's genome, where the map is the PositionMap from the node's coordinates to
        # the output node's. So the intervals are ancestral to the output node's positions
        # that they map to. The lists are filled as material is passed up from children.
        material = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
        identity = PositionMap(1, 0)
        iedge_rows = []
        mutation_rows = []
        mutation_origins = []  # for each mutation row, (time of its original node, original ID)

        def add_iedge(key, parent, parent_chromosome, left, right):
            # An output iedge, by which the output node of ``key`` inherits the interval
            # [left, right) of ``parent`` (on ``parent_chromosome``), through the key's map
            child, child_chromosome, to_child = key
            child_ends = to_child.interval(left, right)
            parent_ends = (left, right)
            if child_ends[0] > child_ends[1]:
                # Child intervals increase: reverse the parent's interval instead
                child_ends, parent_ends = child_ends[::-1], parent_ends[::-1]
            iedge_rows.append(
                dict(
                    child=child,
                    parent=parent,
                    child_left=int(child_ends[0]),
                    child_right=int(child_ends[1]),
                    parent_left=int(parent_ends[0]),
                    parent_right=int(parent_ends[1]),
                    child_chromosome=child_chromosome,
                    parent_chromosome=parent_chromosome,
                )
            )

        sample_set = set(samples)
        # Visit the nodes from the youngest to the oldest (in order of ID, at equal times)
        for u in np.lexsort((np.arange(len(old_nodes)), old_nodes.time)):
            u = int(u)
            if u not in material and u not in sample_set:
                continue
            arriving = material.pop(u, {})
            # The material that u passes up: for each of u's chromosomes, a dict mapping keys
            # (as above) to sets of intervals
            passing_up = {}
            if u in sample_set:
                # A sample is kept: the material from its descendants is inherited from it,
                # and it passes up its whole genome
                out = int(node_map[u])
                for chromosome, keys in arriving.items():
                    for key, intervals in keys.items():
                        for left, right in interval_sets.normalise(intervals):
                            add_iedge(key, out, chromosome, left, right)
                for chromosome in old_iedges.chromosomes_as_child(u):
                    passing_up[chromosome] = {(out, chromosome, identity): [(0, math.inf)]}
            else:
                for chromosome, keys in arriving.items():
                    sets = {key: interval_sets.normalise(intervals) for key, intervals in keys.items()}
                    # The positions of u that it inherits: lineages end at u elsewhere
                    inherited = interval_sets.normalise(
                        [
                            (old_iedges[ie_id].child_left, old_iedges[ie_id].child_right)
                            for ie_id in old_iedges.ids_for_child(u, chromosome=chromosome)
                        ]
                    )
                    here = collections.defaultdict(list)
                    for left, right, covering in interval_sets.coverage(sets):
                        if len(covering) == 1:
                            (key,) = covering
                            # Where the lineage ends here without merging, keep u (like
                            # tskit's keep_input_roots), as otherwise the positions would
                            # have no iedge, and a sample could lose the end of its genome
                            # (e.g. missing data at the end)
                            ended = interval_sets.subtract([(left, right)], inherited)
                            if ended:
                                out = output_node(u, is_sample=False)
                                for end_left, end_right in ended:
                                    add_iedge(key, out, chromosome, end_left, end_right)
                                    here[(out, chromosome, identity)].append((end_left, end_right))
                            # Elsewhere, not a branch point: bypass u, passing the material
                            # straight up
                            for pass_left, pass_right in interval_sets.subtract([(left, right)], ended):
                                here[key].append((pass_left, pass_right))
                        else:
                            # A branch point: keep u here, inherited by each output node
                            out = output_node(u, is_sample=False)
                            for key in covering:
                                add_iedge(key, out, chromosome, left, right)
                            here[(out, chromosome, identity)].append((left, right))
                    passing_up[chromosome] = here

            # Move u's mutations to the output node whose material covers them (u itself,
            # if u is kept there), or remove them if no material covers them. A sample keeps
            # all its mutations (even on chromosomes it doesn't pass up, e.g. as a root).
            for i in old_mutations.ids_for_node(u):
                m = old_mutations[i]
                if u in sample_set:
                    keys = {(int(node_map[u]), m.chromosome, identity): [(0, math.inf)]}
                else:
                    keys = passing_up.get(m.chromosome, {})
                for (out, out_chromosome, to_out), intervals in keys.items():
                    if interval_sets.contains(intervals, m.position):
                        derived_state = m.derived_state
                        if to_out.is_reversing:
                            derived_state = complement_allele(derived_state, self.allele_complements)
                        # (with the time of the node it was above, to order it below)
                        mutation_rows.append(
                            dict(
                                node=out,
                                position=int(to_out.position(m.position)),
                                derived_state=derived_state,
                                chromosome=out_chromosome,
                                time=m.time,
                                metadata=m.metadata,
                            )
                        )
                        mutation_origins.append((old_nodes.time[u], i))
                        break

            # Pass the material up through the iedges above u
            for ie_id in old_iedges.ids_for_child(u):
                ie = old_iedges[ie_id]
                to_parent = PositionMap.from_iedge(ie)
                from_parent = to_parent.inverse()
                for (out, out_chromosome, to_out), intervals in passing_up.get(ie.child_chromosome, {}).items():
                    parent_key = (out, out_chromosome, from_parent.then(to_out))
                    for left, right in intervals:
                        # The part of this interval that the iedge passes to the parent
                        left, right = max(left, ie.child_left), min(right, ie.child_right)
                        if left < right:
                            ends = to_parent.interval(left, right)
                            material[ie.parent][ie.parent_chromosome][parent_key].append((min(ends), max(ends)))

        iedges = IEdgeTable()
        for row in iedge_rows:
            iedges.add_row(**row)
        # Mutations moved down to a node are older than that node's own mutations, so must
        # come before them (at the same position): add them from the oldest original node
        # (the sort below is stable, so keeps this order at each node and position)
        mutations = MutationTable()
        for _, row in sorted(zip(mutation_origins, mutation_rows), key=lambda pair: (-pair[0][0], pair[0][1])):
            mutations.add_row(**row)
        self.nodes, self.iedges, self.mutations = nodes, iedges, mutations
        self.sort()
        self.iedges.squash()
        self.validate()
        return node_map

    def remove_non_ancestral_nodes(self):
        """
        Subset the node table to remove any nonsample nodes that are not referenced
        in the iedges table. Mutations above removed nodes are also removed, and the
        node IDs of the remaining mutations are updated.
        """
        # Requires the child and parent IDs to refer to existing nodes
        self._require_iedge_flags(ValidFlags.IEDGES_PARENT_OLDER_THAN_CHILD, "remove_non_ancestral_nodes()")
        keep_nodes = np.zeros(len(self.nodes), dtype=bool)
        keep_nodes[self.iedges.child] = True
        keep_nodes[self.iedges.parent] = True
        mapping = np.where(keep_nodes)[0]
        inv_mapping = -np.ones(len(self.nodes), dtype=np.int64)
        inv_mapping[mapping] = np.arange(len(mapping))
        self.nodes._datastore[: len(mapping)] = self.nodes._datastore[mapping]
        self.nodes._data = self.nodes._datastore[: len(mapping)]
        ed = self.nodes._extra_data
        self.nodes._extra_data = [ed[i] for i in mapping]
        # Node times are unchanged, and the order of node IDs is kept, so the iedges'
        # node-time flags still hold if they did before
        nodes_were_checked = self.iedges._nodes_version == self.nodes._version
        self.nodes._changed()
        if nodes_were_checked:
            self.iedges._nodes_version = self.nodes._version

        iedges = self.iedges
        iedges._data["child"] = inv_mapping[iedges.child]
        assert np.all(iedges.child >= 0)
        iedges._data["parent"] = inv_mapping[iedges.parent]
        assert np.all(iedges.parent >= 0)

        # must change the built-in indexes. However, edges are in the samme order,
        # so just chnage the keys
        iedges._id_range_for_child = {
            int(inv_mapping[child]): data for child, data in iedges._id_range_for_child.items()
        }

        mutations = self.mutations
        if len(mutations) > 0:
            mutations._subset(np.flatnonzero(keep_nodes[mutations.node]))
            # The mapping preserves the order of node IDs, so the sort order is unchanged
            mutations._data["node"] = inv_mapping[mutations.node]
            mutations._rebuild_index()

        return mapping

    def find_homology(self, u, v, *, mode="first_mrca", u_chromosome=None, v_chromosome=None, time_cutoff=None):
        """
        Find the homology between genomes ``u`` and ``v``: pairs of segments, one in each,
        that trace back to the same segment of a common ancestor, their most recent common
        ancestor (MRCA). This is homology by descent, recorded in the graph, not sequence
        similarity: homologous segments can have different sequences, and segments with the
        same sequence need not be homologous. See :ref:`sec_usage_homology`.

        If ``u`` and ``v`` are the same genome, this finds the homology between different
        positions of it (paralogous homology, e.g. between the copies of a duplicated
        region): each position's trivial homology with itself is left out.

        The ``mode`` says which homology to find. Each position of ``u`` (or ``v``) is traced
        back through its ancestors, and meets the lineages of positions of the other genome
        at their MRCAs, whether these are where the lineages split (allelic homology) or
        where a region was duplicated (paralogous homology):

        * ``"first_mrca"`` (the default): each lineage stops at the first MRCA that it
          reaches. This is the least work, and finds the homology that is usually wanted
          for placing crossovers. But it can miss the closest homology of some positions:
          e.g. if a region was duplicated in a common ancestor of ``u`` and ``v``, and
          ``v`` later lost one copy, the copy that ``v`` lost has no homology in ``v``
          through the duplication, as the lineage it would meet has already stopped.
        * ``"closest_homologues"``: the closest homologues of each position, i.e. its
          homologous positions in the other genome with the most recent MRCA (several, if
          tied), for each position of ``u`` and of ``v``. Lineages are traced as for
          ``"all"``, but only the closest homology of each position is reported, which is
          much less, as it excludes the *remote* homology through older ancestors. Tracing
          stops once every position still being traced has found homology (and
          ``time_cutoff`` can also limit the work).
        * ``"all"``: all the homology, through every common ancestor, each pair of
          positions at its MRCA. This can be much more work, and in the extreme (if every
          position descends from a single ancestral position) covers every pair of
          positions, so ``time_cutoff`` can be used to limit it.

        This needs the iedges to have the flags in
        :data:`~giglib.ValidFlags.IEDGES_FOR_TRACING`, but only for the rows that have been
        validated, so it can be used on tables that are still growing (see
        :ref:`sec_validation`).

        :param int u: The first genome (node ID)
        :param int v: The second genome (node ID)
        :param str mode: Which homology to find: ``"first_mrca"``, ``"closest_homologues"``
            or ``"all"`` (see above)
        :param int u_chromosome: Only find homology on this chromosome of ``u``. If None
            (default), use all of ``u``'s chromosomes
        :param int v_chromosome: Only find homology on this chromosome of ``v``. If None
            (default), use all of ``v``'s chromosomes
        :param float time_cutoff: If given, only find homology whose MRCA is at most this
            old
        :return: The homologous segments, sorted by their position in ``u``
        :rtype: list(Homology)
        """
        modes = ("first_mrca", "closest_homologues", "all")
        if mode not in modes:
            raise ValueError(f"Unknown mode {mode!r}: must be one of {', '.join(modes)}")
        if not isinstance(u, (int, np.integer)) or not isinstance(v, (int, np.integer)):
            raise ValueError("u and v must be integers")

        # Only the iedges above u and v are read (using ids_for_child, which checks that
        # each node's iedges are validated), so the validated rows are enough
        self._require_iedge_flags(ValidFlags.IEDGES_FOR_TRACING, "find_homology()", validated_rows=True)

        u_chromosomes = self.iedges.chromosomes_as_child(u) if u_chromosome is None else [u_chromosome]
        v_chromosomes = self.iedges.chromosomes_as_child(v) if v_chromosome is None else [v_chromosome]
        if time_cutoff is None:
            time_cutoff = np.inf
        # Use a dynamic stack, ordered by node time, as we hopefully will be visiting a
        # minority of nodes
        node_times = self.nodes.time
        stack = sortedcontainers.SortedDict(lambda k: -node_times[k])

        # The material of u and v traced up to each node. Intervals need tracking
        # separately if they have been transformed (inverted or coordinate-shifted), so
        # for each node, the stack stores *two* dicts, the first for the material of u and
        # the second for that of v. Each maps a chromosome of the node to a dict of lists
        # of (left, right) intervals in the node's coordinates (see giglib.intervals),
        # keyed by (to_original, original chromosome), where to_original is the
        # PositionMap from the coordinates of the node back to those of u (or v). If a node
        # has material in both dicts, positions of u and v can meet there, making it their
        # MRCA. Intervals are added to a node's lists as they arrive, and the lists are
        # tidied into sets of intervals once, when the node is visited.
        identity = PositionMap(1, 0)
        stack[u] = ({c: {(identity, c): [(0, math.inf)]} for c in u_chromosomes}, {})
        if v not in stack:
            stack[v] = ({}, {})
        stack[v][1].update({c: {(identity, c): [(0, math.inf)]} for c in v_chromosomes})

        result = []
        # What has been reported so far. As nodes are visited from the most recent, a pair
        # of positions is first met at its MRCA. For mode "all": the positions of u already
        # reported, for each map from u to v (and pair of chromosomes), so that a pair isn't
        # reported again when it meets in older ancestors. For mode "closest_homologues":
        # the positions of u and of v (for each chromosome) with homology at a more recent
        # time ("found"), and at the time of the current node ("pending", added to "found"
        # once all the nodes at that time have been visited, so that ties are kept).
        reported = {"all": {}, "found": ({}, {}), "pending": ({}, {}), "time": -np.inf}

        while len(stack) > 0:
            node, (u_dict, v_dict) = stack.popitem()
            if node_times[node] > time_cutoff:
                break
            for material in (u_dict, v_dict):
                for interval_dict in material.values():
                    for key, intervals in interval_dict.items():
                        interval_dict[key] = interval_sets.normalise(intervals)
            if node_times[node] > reported["time"]:
                for found, pending in zip(reported["found"], reported["pending"]):
                    for chromosome, intervals in pending.items():
                        found[chromosome] = interval_sets.union(found.get(chromosome, []), intervals)
                    pending.clear()
                reported["time"] = node_times[node]
                # In mode "closest_homologues", a pair is only reported if this is the
                # most recent MRCA of one of its positions. So once every position still
                # being traced (in u and in v) has found homology, nothing older can be
                # reported, and tracing can stop.
                if mode == "closest_homologues":
                    traced = [(u_dict, v_dict), *stack.values()]
                    if not any(self._has_unfound(dicts, reported["found"]) for dicts in traced):
                        break
            # The chromosomes of u (or v) are found from the iedges above it, but it can
            # also have chromosomes without a parent (e.g. if it is a root). These only
            # matter if the other genome descends from it, i.e. if the other genome's
            # material reaches it, so add them (all of each one) when it is visited
            for genome, chromosome, own, other in (
                (u, u_chromosome, u_dict, v_dict),
                (v, v_chromosome, v_dict, u_dict),
            ):
                if node == genome and chromosome is None:
                    for c in other.keys() - own.keys():
                        own[c] = {(identity, c): [(0, math.inf)]}
            for chrom in u_dict.keys() & v_dict.keys():
                # Cut the node's chromosome into pieces that each have the same material of
                # u and of v (there can be several of each, e.g. copies of a duplicated
                # region)
                # If u and v are the same genome, each position meets itself, along the same
                # route (the same key) on both sides: that trivial homology is left out
                met = []
                for left, right, u_keys, v_keys in interval_sets.pieces(u_dict[chrom], v_dict[chrom]):
                    pairs = [(u_key, v_key) for u_key in u_keys for v_key in v_keys if u != v or u_key != v_key]
                    if pairs:
                        met.append((left, right))
                    for u_key, v_key in pairs:
                        mrca_segment = Segment(left, right, chrom)
                        self._add_homology(result, reported, mode, node, mrca_segment, u_key, v_key)
                if mode == "first_mrca" and len(met) > 0:
                    # Each lineage stops at its first MRCA: remove what has met here
                    met = interval_sets.normalise(met)
                    for uv_dict in (u_dict, v_dict):
                        for key, intervals in uv_dict[chrom].items():
                            uv_dict[chrom][key] = interval_sets.subtract(intervals, met)

            # Transmit the remaining intervals upwards
            for ie_id in self.iedges.ids_for_child(node):
                # See note in sample resolve algorithm re: efficiency and
                # indexing by left coord. Note this is more complex here
                # because we have to treat intervals with different
                # offsets / orientations separately
                ie = self.iedges[ie_id]
                parent = ie.parent
                to_parent = PositionMap.from_iedge(ie)
                from_parent = to_parent.inverse()
                for u_or_v, u_or_v_dict in enumerate((u_dict, v_dict)):
                    for chromosome, interval_dict in u_or_v_dict.items():
                        if chromosome != ie.child_chromosome:
                            continue
                        for (to_original, orig_chr), intervals in interval_dict.items():
                            key = (from_parent.then(to_original), orig_chr)
                            for left, right in intervals:
                                # The part of this interval that the iedge passes to the parent
                                left, right = max(left, ie.child_left), min(right, ie.child_right)
                                if left < right:
                                    p = to_parent.interval(left, right)
                                    if parent not in stack:
                                        stack[parent] = ({}, {})
                                    at_parent = stack[parent][u_or_v].setdefault(ie.parent_chromosome, {})
                                    at_parent.setdefault(key, []).append((min(p), max(p)))

        result.sort(key=lambda h: (h.u.chromosome, h.u.left, h.u.right, h.v.chromosome, h.v.left, h.v.right, h.mrca))
        return result

    @staticmethod
    def _has_unfound(dicts, found):
        """
        Whether the material of u and of v at a node (``dicts``, as in the stack of
        :meth:`find_homology`) includes a position of u or v that is not in ``found`` (the
        positions of u and of v with homology found so far, for each chromosome)
        """
        for material, found_here in zip(dicts, found):
            for interval_dict in material.values():
                for (to_original, original_chromosome), intervals in interval_dict.items():
                    for left, right in intervals:
                        ends = to_original.interval(left, right)
                        original = [(min(ends), max(ends))]
                        if interval_sets.subtract(original, found_here.get(original_chromosome, [])):
                            return True
        return False

    @staticmethod
    def _add_homology(result, reported, mode, mrca, mrca_segment, u_key, v_key):
        """
        Add the homology between u and v through a segment of their MRCA to the result
        of :meth:`find_homology`, given the keys ``(to_original, original chromosome)`` of
        the material of u and v, where ``to_original`` maps from the MRCA's coordinates to
        those of u or v. In mode "all", only add the positions of u not already reported
        with the same map to v (as they met at a more recent MRCA). In mode
        "closest_homologues", only add pairs of positions for which this is the most
        recent homology of the position in u, or of the position in v.
        """
        (to_u, u_chr), (to_v, v_chr) = u_key, v_key
        u_to_mrca = to_u.inverse()
        u_to_v = u_to_mrca.then(to_v)
        u_ends = to_u.interval(mrca_segment.left, mrca_segment.right)
        u_left, u_right = int(min(u_ends)), int(max(u_ends))
        here = [(u_left, u_right)]
        if mode == "all":
            key = (u_to_v, u_chr, v_chr)
            already = reported["all"].get(key, [])
            new = interval_sets.subtract(here, already)
            reported["all"][key] = interval_sets.union(already, new)
        elif mode == "closest_homologues":
            (u_found, v_found), (u_pending, v_pending) = reported["found"], reported["pending"]
            # The positions of u, and those (in u's coordinates) whose homologues in v,
            # have no homology at a more recent time
            new = interval_sets.subtract(here, u_found.get(u_chr, []))
            v_ends = u_to_v.interval(u_left, u_right)
            v_to_u = u_to_v.inverse()
            for left, right in interval_sets.subtract([(min(v_ends), max(v_ends))], v_found.get(v_chr, [])):
                ends = v_to_u.interval(left, right)
                new = interval_sets.union(new, [(min(ends), max(ends))])
            for left, right in new:
                u_pending[u_chr] = interval_sets.union(u_pending.get(u_chr, []), [(left, right)])
                ends = u_to_v.interval(left, right)
                v_pending[v_chr] = interval_sets.union(v_pending.get(v_chr, []), [(min(ends), max(ends))])
        else:
            new = here
        for left, right in new:
            result.append(
                Homology(
                    Segment(int(left), int(right), u_chr),
                    Segment(*(int(x) for x in u_to_v.interval(left, right)), v_chr),
                    int(mrca),
                    Segment(*(int(x) for x in u_to_mrca.interval(left, right)), mrca_segment.chromosome),
                )
            )
