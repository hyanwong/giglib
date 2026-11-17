import collections
import dataclasses
import logging
import types

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import portion as P
import sortedcontainers
import tskit

from .constants import Const, ValidFlags
from .util import (
    DEFAULT_ALLELE_COMPLEMENTS,
    add_rectangle,
    add_row_label,
    add_triangle,
    complement_allele,
    is_unknown_time,
    truncate_rows,
    warn_tskit_name,
)

NULL = Const.NULL  # alias for tskit.NULL
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

    def _df(self):
        """
        Temporary hack to convert the table to a Pandas dataframe.
        Shouldn't be used for anything besides exploratory work!
        """
        return pd.DataFrame([row._asdict() for row in self._data])


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
        copy.flags = self.flags
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
        # save some flags to indicate if this is a valid
        self.flags = ValidFlags.GIG
        # map each child ID to a set of chromosome numbers,
        # each of which is mapped to an IEdgeRange
        self._id_range_for_child = {}
        super().clear()

    def __eq__(self, other):
        if self.flags != other.flags:
            return False
        if self._id_range_for_child != other._id_range_for_child:
            return False
        return super().__eq__(other)

    def set_bitflag(self, flags):
        """
        Mark the given validity flags as guaranteed for this table.

        :param int flags: One or more :class:`~giglib.ValidFlags`, combined using ``|``
        """
        self.flags |= flags

    def unset_bitflag(self, flags):
        """
        Mark the given validity flags as no longer guaranteed for this table.

        :param int flags: One or more :class:`~giglib.ValidFlags`, combined using ``|``
        """
        self.flags &= ~flags

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
        validate=None,
        skip_validate=None,
    ) -> int:
        """
        The canonical way to add data to an IEdgeTable

        .. seealso::

            :meth:`.Tables.add_iedge_row` which is a wrapper around this method
            that also allows validation of parent and child node times

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
        :param int validate: A set of bitflags (as listed in :class:`~giglib.ValidFlags`)
            specifying which iedge table validation checks
            should be performed when adding this data. If the existing data is valid, and
            the new data is added in a way that preserves the existing validity, then
            calling :meth:`has_all_bitflags` with the flags in this set will return
            True. If any of the bits in ``validate`` are ``0``, that particular
            validation will not be performed: in this case :meth:`has_all_bitflags`
            will return False for those flags, and some table algorithms will not run.
            For instance, using the :meth:`ids_for_child()` method is only valid if
            :data:`~giglib.ValidFlags.IEDGES_FOR_CHILD_ADJACENT` and
            :data:`~giglib.ValidFlags.IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC`
            are set, so if you wish to use that method you should add those flags to
            ``validate``. Defaults to ``None`` which is treated as ``0``, meaning that
            all ``IEDGE`_`` validation flags will be zeroed, no validation checks will
            be performed, and hence the table will be marked as containing potentially
            invalid iedge information.
        :param bool skip_validate: If True, assume the user has checked that this
            operation will pass the validation tests implied by the ``validate``
            flags. This means that the validation routines will not be run, but the
            tables may claim that they are valid when they are not. If False, and any of
            the ``iedges_validation`` flags are set, perform the appropriate validations
            (default: ``None`` treated as False)

            .. warning::

                    This parameter is only intended to speed up well-tested
                    code. It should only be set to ``True`` when you are sure that any
                    calling code has been tested with validation.

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
                validate=ValidFlags.IEDGES_INTERVALS,
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
        # Stick into the datastore but don't allow it to be accessed via ._data yet
        num_rows = len(self._data)
        if len(self._datastore) == num_rows:
            self._expand_datastore()
        self._datastore[num_rows] = (
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

        if validate is None:
            validate = ~ValidFlags.IEDGES_ALL

        # only try validating if any IEDGES flags are set
        if (not skip_validate) and bool(validate & ValidFlags.IEDGES_ALL):
            # need to check the values before they were put into the data array,
            # as numpy silently converts floats to integers on assignment
            if validate & ValidFlags.IEDGES_INTEGERS:
                for i, val in enumerate(
                    (
                        child,  # 0
                        parent,  # 1
                        child_left,  # 2
                        child_right,  # 3
                        parent_left,  # 4
                        parent_right,  # 5
                        child_chromosome,  # 6
                        parent_chromosome,  # 7
                        edge,  # 8
                    )
                ):
                    if int(val) != val:
                        raise ValueError("Iedge data must be integers")
                    if i < 6 and val < 0:
                        raise ValueError("Iedge data must be non-negative (except edge ID)")
            self._validate_add_row(validate, self._RowClass(*self._datastore[num_rows].tolist()))

        # Passed validation: keep those flags (unset others). Note that we can't validate
        # the age of the parent and child here so we have to set those flags to invalid
        # (they can be set to valid by using the table.add_iedge_row wrapper instead)
        self.flags &= validate | ~ValidFlags.IEDGES_ALL

        # Index keys are Python ints, even if e.g. numpy integers were passed in
        self._increment_id_range_for_child(int(child), int(child_chromosome), num_rows)
        # expand the visible data array
        self._data = self._datastore[0 : num_rows + 1]

        return num_rows

    def _validate_add_row(self, vflags, row):
        # All iedge independent validations performed here. The self.flags
        # bits are set after all these validations have passed.

        if vflags & ~ValidFlags.IEDGES_COMBO_STANDALONE:
            raise ValueError(
                "Validation cannot be performed within edges.add_row() for flags " "involving the node table."
            )
        prev_iedge = None if len(self) == 0 else self[-1]
        same_child = prev_iedge is not None and prev_iedge.child == row.child
        chrom = row.child_chromosome

        if vflags & ValidFlags.IEDGES_FOR_CHILD_NONOVERLAPPING:
            # This is easy to check if previous edges for a child+chromosome are
            # adjacent and ordered by left position, but hard in the general case
            if self.has_all_bitflags(ValidFlags.IEDGES_WITHIN_CHILD_SORTED):
                if same_child and prev_iedge.child_chromosome == chrom:
                    if prev_iedge.child_right > row.child_left:
                        raise ValueError(
                            f"Adding an iedge with left position {row.child_left} "
                            f"for child {row.child} would make iedges overlap"
                        )
            else:
                raise ValueError("Can't validate non-overlapping iedges unless they are " "guaranteed to be sorted")

        if vflags & ValidFlags.IEDGES_FOR_CHILD_ADJACENT:
            if prev_iedge is not None and row.child != prev_iedge.child and row.child in self._id_range_for_child:
                raise ValueError(f"Adding an iedge with child ID {row.child} would make IDs " "non-adjacent")

        if vflags & ValidFlags.IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC:
            if same_child:
                has_chrom = chrom in self._id_range_for_child.get(row.child, {})
                if prev_iedge.child_chromosome != chrom and (has_chrom or chrom < prev_iedge.child_chromosome):
                    raise ValueError(
                        f"Adding an iedge with chromosome ID {chrom} for child "
                        f"{row.child} would make chromosome IDs out of order"
                    )

        if vflags & ValidFlags.IEDGES_FOR_CHILD_SECONDARY_ORDER_LEFT_ASC:
            if same_child and prev_iedge.child_chromosome == chrom:
                if prev_iedge.child_left >= row.child_left:
                    raise ValueError(
                        f"Adding an iedge with left position {row.child_left} for "
                        f"child {row.child} would break edge_left ordering"
                    )

        if vflags & ValidFlags.IEDGES_INTERVALS:
            if abs(row.child_span) != abs(row.parent_span):
                raise ValueError(f"Bad intervals ({row}): child & parent absolute spans differ")

        if vflags & ValidFlags.IEDGES_CHILD_INTERVAL_POSITIVE:
            if row.child_left >= row.child_right:
                raise ValueError(f"Bad intervals ({row}): child left must be < child right")

        if vflags & ValidFlags.IEDGES_SAME_PARENT_CHILD_FOR_EDGE:
            if row.edge != NULL:
                # TODO: this is a very slow validation. Hopefully hardly every used
                same_edge = np.where(self.edge == row.edge)[0]
                for e in same_edge:
                    if self[e].child != row.child or self[e].parent != row.parent:
                        raise ValueError(
                            f"Edge ID {row.edge} already exists in the table for a "
                            "different parent/child combination"
                        )

    def _from_tskit(self, kwargs):
        new_kw = {}
        for side in ("left", "right"):
            if side in kwargs:
                new_kw["child_" + side] = new_kw["parent_" + side] = kwargs[side]
        new_kw.update(kwargs)
        return new_kw

    def append(self, obj, validate=None, skip_validate=None) -> int:
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
        return self.add_row(
            **{k: v for k, v in kwargs.items() if k in self._RowClass._fields},
            validate=validate,
            skip_validate=skip_validate,
        )

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
            or not ordered by chromosome
        """
        if not self.has_all_bitflags(
            ValidFlags.IEDGES_FOR_CHILD_ADJACENT | ValidFlags.IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC
        ):
            raise ValueError(
                "Cannot use this method unless iedges have adjacent child IDs" " and are ordered by chromosome"
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
        Return the maximum child position for a given child ID and chromosome.
        This should be equivalent np.max(child_right[child == u]), but faster
        because it relies on
        :data:`~giglib.ValidFlags.IEDGES_WITHIN_CHILD_SORTED`.

        If chromosome is not given, this will return the maximum child
        position in *any* chromosome.

        Returns None if there are no iedges for the given child.

        .. note::
            This does not look at edges for which this node is a parent,
            so e.g. root nodes may not have a maximum position defined
        """
        if not self.has_all_bitflags(
            ValidFlags.IEDGES_WITHIN_CHILD_SORTED | ValidFlags.IEDGES_FOR_CHILD_NONOVERLAPPING
        ):
            raise ValueError(
                "Cannot use this method unless iedges for a child are adjacent, "
                "nonoverlapping, and ordered by chromosome then position"
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
        Return the minimum child position for a given child ID and chromosome.
        This should be equivalent np.max(child_right[child == u]), but faster
        because it relies on :data:`~giglib.ValidFlags.IEDGES_WITHIN_CHILD_SORTED`.

        If chromosome is not given, this will return the minimum child
        position in *any* chromosome.

        Returns None if there are no iedges for the given child.

        .. note::
            This does not look at edges for which this node is a parent,
            so e.g. root nodes may not have a minimum position defined
        """
        if not self.has_all_bitflags(
            ValidFlags.IEDGES_WITHIN_CHILD_SORTED | ValidFlags.IEDGES_FOR_CHILD_NONOVERLAPPING
        ):
            raise ValueError(
                "Cannot use this method unless iedges for a child are adjacent, "
                "nonoverlapping, and ordered by chromosome then position"
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

    def transform_interval(self, iedge_id, interval, direction):
        """
        Given an iedge ID, use that edge to transform the provided interval.
        If this is an inversion then an iedge such as ``[0, 10, chrA] -> [10, 0, chrB]``
        means that a interval such as ``[1, 8]`` gets transformed to
        ``[9, 2]``. The chromosome transformation is not reported: it is assumed
        that the intervals are on chromosomes specified by the iedge

        :param int edge_id: The edge specifying the transformation
        :param tuple(int, int) interval: The (left, right) interval to transform,
            assumed to be on the appropriate chromosome for this iedge
        :param int direction: Either Const.LEAFWARDS or Const.ROOTWARDS
            (only ROOTWARDS currently implemented)
        """
        # Requires edges to be sane
        if not self.has_all_bitflags(ValidFlags.IEDGES_INTERVALS):
            raise ValueError("Cannot use the method unless iedges have valid intervals")
        ie = self[iedge_id]
        for x in interval:
            if x < ie.child_left or x > ie.child_right:
                raise ValueError(f"Position {x} not in child interval for {ie}")

        if direction == Const.ROOTWARDS:
            if ie.is_inversion():
                return tuple(ie.child_left - x + ie.parent_left for x in interval)
            else:
                return tuple(x - ie.child_left + ie.parent_left for x in interval)
        raise ValueError(f"Direction must be Const.ROOTWARDS, not {direction}")


class NodeTable(BaseExtraTable):
    """
    A table containing the nodes with their times
    """

    _RowClass = NodeTableRow
    _non_int64_fieldtypes = types.MappingProxyType({"time": np.float64, "flags": np.uint32})

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


class MRCAdict(dict):
    """
    A dictionary to store the results of the MRCA finder. This is a dict of dicts
    of the following form
            {
                MRCA_node_ID1 : {segment_x: (
                    u = [segment_u1, segment_u2, ...],
                    v = [segment_v1, ...]
                )},
                MRCA_node_ID2 : ...,
                ...
            }
    In each inner dict, the key is a :class:`Segment` ``(X, Y, CHRZ)`` giving an interval
    (with X < Y) in the MRCA node for chromosome Z (a plain tuple of the same form can also
    be used), and the value is an ``MRCAintervals`` tuple which gives the corresponding
    intervals and chromosome numbers in u and v. In the example above there
    is a list of two corresponding intervals in u: (uA, uB, CHRuC) and (uC, uD, CHRuE)
    representing a duplication of the MRCA interval into u.
    If uA > uB then that interval in u is inverted relative to that in the MRCA node.

    Subclassing the default Python ``dict`` means that we can add some useful functions
    such as the ability to locate a random point in the MRCA and return the equivalent
    points in ``u`` and ``v`` as well as visualization routines.
    """

    # Convenience tuples
    MRCAintervals = collections.namedtuple("MRCAintervals", "u, v")  # store in the dict
    # Store equivalent positions in u & v (as Position tuples), and whether one is
    # inverted relative to the other (a property of the pair, not of either position)
    MRCApos = collections.namedtuple("MRCApos", "u, v, opposite_orientations")

    def random_match_pos(self, rng):
        """
        Choose a position uniformly at random from the mrca regions and return
        an equivalent position in u and v.

        .. note::
            It is hard to know from the MRCA structure whether intervals are
            adjacent, so if this is used to locate a breakpoint, the choice of
            whether a breakpoint is positioned to the left or right of the returned
            position is left to the user.

        :param obj rng: A numpy random number generator with a method ``integers()``
            that behaves like ``np.random.default_rng().integers``.
        :returns: a named tuple of ``(u, v, opposite_orientations)``, where ``u`` and
            ``v`` are :class:`Position` tuples giving the position and chromosome in
            each node. Positions are chosen at random if there are multiple equivalent positions
            for u, or multiple equivalent positions for v. If one of the sequences is
            inverted relative to the other then ``.opposite_orientations`` is ``True``.
        :rtype: MRCApos
        """
        # Keys are Segments, but unpack them as tuples, so plain (left, right, chromosome)
        # tuples also work (and unpacking is faster than indexing a Segment)
        tot_len = sum(right - left for v in self.values() for left, right, _ in v.keys())
        # Pick a single breakpoint
        loc = rng.integers(tot_len)  # breakpoint is before this base
        for mrca_intervals in self.values():
            for x in mrca_intervals.keys():
                left, right, _ = x
                if loc < right - left:
                    u_matches, v_matches = mrca_intervals[x]
                    assert len(u_matches) != 0
                    assert len(v_matches) != 0
                    u = u_matches[0] if len(u_matches) == 1 else rng.choice(u_matches)
                    v = v_matches[0] if len(v_matches) == 1 else rng.choice(v_matches)
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
                    # (convert to Python ints, as rng.choice returns numpy values)
                    u0, u1, uCHR = (int(x) for x in u)
                    v0, v1, vCHR = (int(x) for x in v)
                    loc = int(loc)
                    if u0 < u1:
                        if v0 < v1:
                            return self.MRCApos(Position(u0 + loc, uCHR), Position(v0 + loc, vCHR), False)
                        else:
                            return self.MRCApos(Position(u0 + loc, uCHR), Position(v0 - loc - 1, vCHR), True)
                    else:
                        if v0 < v1:
                            return self.MRCApos(Position(u0 - loc - 1, uCHR), Position(v0 + loc, vCHR), True)
                        else:
                            return self.MRCApos(Position(u0 - loc - 1, uCHR), Position(v0 - loc - 1, vCHR), False)
                loc -= right - left

    def _plot(
        self,
        highlight_position=None,
        ax=None,
        *,
        fontsize=14,
        u_pos=0.5,
        v_pos=1.5,
        y_offset=0.4,
        x_offset=-1.5,
    ):
        """
        Plot the MRCA dict structure, showing:
        1. The MRCA node and corresponding MRCA intervals (rows 3 and up)
        2. All of the U and V intervals corresponding to the MRCAs (rows 1 and 2)

        If ``highlight_position`` is given, a specific position in one of the
        MRCA intervals is determined, and matching positions in U and V are
        highlighted. The position is determined by concatenating the MRCA intervals
        end-to-end and then choosing the position from this concatenation. Note that
        this may not reflect genomic position in the original MRCA node(s).

        If ``ax`` is not specified, a new figure is created. To change the size of
        a figure, create your own axis and pass it in e.g.
        ``fig, ax = plt.subplots(figsize=(10, 5)); mrca_dict._plot(ax=ax)``

        .. note::
            This is currently for testing only, and is subject to major API changes. E.g.
            there are a bunch of plotting parameters and ideally we want them
            to be calculated automatically instead.
        """
        if ax is None:
            _, ax = plt.subplots(1)
        x_max = 0
        y_pos = 3
        found_breakpoint = False
        if highlight_position is None:
            # ensure that x is never highlighted
            highlight_position = np.inf
        else:
            tot_len = sum(X[1] - X[0] for v in self.values() for X in v.keys())
            assert highlight_position < tot_len

        for mrca_node, mrca_intervals in self.items():
            for X in mrca_intervals.keys():
                x_max = max(x_max, X[1])
                if highlight_position < X[1] - X[0] and found_breakpoint is False:
                    # TODO - use default colours rather than hard-coding
                    triX = X[0] + highlight_position
                    lab = f"MRCA: {mrca_node}"
                    add_rectangle(ax, X, y_pos, "#a6cee3")
                    add_triangle(ax, (triX, triX + 1), y_pos, "right", "#1f78b4")
                    add_row_label(
                        ax,
                        x_offset,
                        y_pos + y_offset,
                        lab,
                        fontsize,
                        color="#1f78b4",
                    )
                    U_list, V_list = mrca_intervals[X]
                    for U in U_list:
                        add_rectangle(ax, U, u_pos, "#b2df8a")
                        x_max = max(x_max, *U)
                        if U[0] < U[1]:
                            u_0 = U[0] + highlight_position
                            add_triangle(ax, (u_0, u_0 + 1), u_pos, "right", "#33a02c")
                        else:
                            u_0 = U[0] - highlight_position - 1
                            add_triangle(ax, (u_0, u_0 + 1), u_pos, "left", "#33a02c")
                    for V in V_list:
                        add_rectangle(ax, V, v_pos, "#fb9a99")
                        x_max = max(x_max, *V)
                        if V[0] < V[1]:
                            v_0 = V[0] + highlight_position
                            add_triangle(ax, (v_0, v_0 + 1), v_pos, "right", "#e31a1c")
                        else:
                            v_0 = V[0] - highlight_position - 1
                            add_triangle(ax, (v_0, v_0 + 1), v_pos, "left", "#e31a1c")

                    found_breakpoint = True
                else:
                    highlight_position -= X[1] - X[0]
                    add_rectangle(ax, X, y_pos, "#b2b2b2")
                    add_row_label(
                        ax,
                        x_offset,
                        y_pos + y_offset,
                        f"MRCA: {mrca_node}",
                        fontsize,
                        color="black",
                    )
                    for lists, pos in zip(mrca_intervals[X], (u_pos, v_pos)):
                        for interval in lists:
                            add_rectangle(ax, interval, pos, "#838383")
                            x_max = max(x_max, *interval)
            y_pos += 1

        add_row_label(ax, x_offset, u_pos + y_offset, "U", fontsize, color="black")
        add_row_label(ax, x_offset, v_pos + y_offset, "V", fontsize, color="black")
        ax.set_ylim(0, y_pos)
        ax.set_xlim(0, x_max)
        ax.tick_params(axis="x", labelsize=fontsize - 2)
        ax.set_xlabel("Position", fontsize=fontsize)
        ax.yaxis.set_visible(False)


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

    def _validate_add_iedge_row(self, vflags, child, parent):
        try:
            child_time = self.nodes[child].time
            parent_time = self.nodes[parent].time
        except IndexError as err:
            raise ValueError("Child or parent ID does not correspond to a node in the node table") from err

        if vflags & ValidFlags.IEDGES_PARENT_OLDER_THAN_CHILD:
            if child_time >= parent_time:
                raise ValueError("Child time is not less than parent time")
        if len(self.iedges) > 0:
            # All other validations can only fail if iedges exist already
            prev_child_id = self.iedges[-1].child
            prev_child_time = self.nodes[prev_child_id].time
            if vflags & ValidFlags.IEDGES_PRIMARY_ORDER_CHILD_TIME_DESC:
                if prev_child_time < child_time:
                    raise ValueError(
                        f"Added iedge has child time of {child_time} which is older "
                        f"than the previous iedge child time of {prev_child_time}"
                    )
            if vflags & ValidFlags.IEDGES_SECONDARY_ORDER_CHILD_ID_ASC:
                if (prev_child_time == child_time) and (prev_child_id > child):
                    raise ValueError("Added iedge has lower child ID than the previous one")

    def add_iedge_row(self, child, parent, *, validate=None, skip_validate=None, **kwargs):
        """
        Similar to the :meth:`~tables.IEdgeTable.add_row` method of the
        :class:`~tables.IEdgeTable` stored in this set of tables, but also (optionally)
        can be used to validate features of the nodes used (e.g. that the parent
        node time is older than the child node time).

        :param int child: The ID of the child node
        :param int parent: The ID of the parent node
        :param int validate: A set of bitflags (as listed in :class:`~giglib.ValidFlags`)
            specifying which iedge table validation checks to perform. In particular,
            this can include the :data:`~giglib.ValidFlags.IEDGES_PARENT_OLDER_THAN_CHILD` and
            :data:`~giglib.ValidFlags.IEDGES_PRIMARY_ORDER_CHILD_TIME_DESC` flags, which will
            check the node table for those properties. Other flags will be passed to
            :meth:`tables.IEdgeTable.add_row`, along with any other (keyword) arguments,
            such as the coordinates, e.g. ``tables.add_iedge_row(1, 0, child_right=100)``.
        :param bool skip_validate: If True, assume the user has checked that this
            operation will pass the validation tests implied by the ``validate`` flags.
            This means that the validation routines will not be run, but the tables
            may claim that they are valid when they are not. If False, and any of the
            ``IEDGES_...`` :ref:`validation flags<sec_python_api_tables_validity>` are
            set, perform the appropriate validations. Default: ``None`` treated as False.

            .. warning::
                The ``skip_validate`` parameter is only intended to speed up well-tested
                code. It should only be set to ``True`` when you are sure that any
                calling code has been tested with validation.
        """
        if validate is None:
            validate = ~ValidFlags.IEDGES_ALL
        node_validate = validate & ValidFlags.IEDGES_COMBO_NODE_TABLE
        store_node_validation = self.iedges.flags & node_validate
        if (not skip_validate) and bool(node_validate):
            self._validate_add_iedge_row(validate, child, parent)
        # Only pass the validation flags that can be checked to the lower level
        self.iedges.add_row(
            child,
            parent,
            **kwargs,
            validate=validate & ValidFlags.IEDGES_COMBO_STANDALONE,
            skip_validate=skip_validate,
        )
        # The add_row routine will have unset the node-specific iedge flags
        # So we set them back to the previous flags (but only if validated)
        self.iedges.set_bitflag(store_node_validation)

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

    def copy(self, omit_iedges=None):
        """
        Return an unfrozen deep copy of the tables.

        :param bool omit_iedges: If True do not copy the iedges table, use an empty one.
        :return: A deep copy of the tables
        :rtype: giglib.Tables
        """
        copy = self.__class__()
        for name, cls in zip(self.table_classes._fields, self.table_classes):
            if omit_iedges and name == "iedges":
                setattr(copy, name, cls())
            else:
                setattr(copy, name, getattr(self, name).copy())
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

        iedges.set_bitflag(ValidFlags.IEDGES_SORTED)

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
        requires the following iedge conditions to be met:

            * The child_left, child_right, parent_left, parent_right, parent, and child
              values must all be provided and be non-negative integers. If
              ``child_chromosome`` and ``parent_chromosome`` are provided, they also
              must be non-negative integers (otherwise a default chromosome ID of 0 is
              assumed). This corresponds to the flag
              :data:`~giglib.ValidFlags.IEDGES_INTEGERS`.
            * The intervals must be valid (i.e. ``abs(child_left - child_right)`` is
              finite, nonzero, and equal to ``abs(parent_left - parent_right)``. This
              corresponds to the flag :data:`~giglib.ValidFlags.IEDGES_INTERVALS`.
            * Each chromosomal position in a child is covered by only one interval (i.e.
              for any particular chromosome, intervals for a given child do not
              overlap). This corresponds to the flag
              :data:`~giglib.ValidFlags.IEDGES_FOR_CHILD_NONOVERLAPPING`
            * The ``child`` and ``parent`` IDs of an iedge must correspond to nodes in
              in the node table in which the parent is older (has a strictly greater
              time) than the child. This corresponds to the flag
              :data:`~giglib.ValidFlags.IEDGES_PARENT_OLDER_THAN_CHILD`
            * If an ``edge`` ID is provided for an iedge, and it is not ``NULL`` (-1),
              then other iedges with the same ``edge`` ID must have the same
              ``child`` and ``parent`` IDs. This corresponds to the flag
              :data:`~giglib.ValidFlags.IEDGES_SAME_PARENT_CHILD_FOR_EDGE`
            * For consistency and as an enforced convention, the ``child_left`` position
              for an iedge must be strictly less than the ``child_right`` position.
              This corresponds to the flag
              :data:`~giglib.ValidFlags.IEDGES_CHILD_INTERVAL_POSITIVE`

        To create a valid GIG, the iedges must also be sorted into a canonical order
        such that the following conditions are met (you can also accomplish this by
        calling :meth:`~giglib.Tables.sort` on the tables first):

            * The iedges must be grouped by child ID. This corresponds to the flag
              :data:`~giglib.ValidFlags.IEDGES_FOR_CHILD_ADJACENT`
            * Within each group of iedges with the same child ID, the iedges must be
              ordered by chromosome ID and then by left position. This corresponds to
              the flags :data:`~giglib.ValidFlags.IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC` and
              :data:`~giglib.ValidFlags.IEDGES_FOR_CHILD_SECONDARY_ORDER_LEFT_ASC`
            * The groups of iedges with the same child ID must be ordered
              by time of child node and then (if nodes have identical times) by
              child node ID. This corresponds to the flags
              :data:`~giglib.ValidFlags.IEDGES_PRIMARY_ORDER_CHILD_TIME_DESC` and
              :data:`~giglib.ValidFlags.IEDGES_SECONDARY_ORDER_CHILD_ID_ASC`.

        .. note::
            The nodes are not required to be in any particular order (e.g. by time)

        :return: A genetic inheritance graph object
        :rtype: giglib.Graph

        """
        from .graph import Graph

        return Graph(self)

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
            for x, allele in known.items():
                if ie.is_inversion():
                    position = ie.child_left + ie.parent_left - 1 - x
                    allele = complement_allele(allele, complements)
                else:
                    position = ie.child_left + x - ie.parent_left
                replacements.setdefault((ie.child, ie.child_chromosome, position), allele)
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
            tables.iedges.append(obj, validate=ValidFlags.IEDGES_INTEGERS)
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
        tables.iedges.set_bitflag(
            ValidFlags.IEDGES_INTERVALS
            | ValidFlags.IEDGES_FOR_CHILD_NONOVERLAPPING
            | ValidFlags.IEDGES_PARENT_OLDER_THAN_CHILD
            | ValidFlags.IEDGES_SAME_PARENT_CHILD_FOR_EDGE
            | ValidFlags.IEDGES_CHILD_INTERVAL_POSITIVE
        )
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
        if not self.iedges.has_all_bitflags(ValidFlags.IEDGES_PARENT_OLDER_THAN_CHILD):
            raise ValueError("Can only run if children younger than parents")

        stack = {c: collections.defaultdict(P.interval.Interval) for c in reversed(np.argsort(self.nodes.time))}
        old_iedges = self.iedges
        self.iedges = IEdgeTable()  # Make a new one
        # The regions of each (non-sample) node inherited by samples, for filtering mutations
        ancestral_regions = {}
        while len(stack):
            child, chromosomes = stack.popitem()
            ancestral_regions[child] = chromosomes
            # An internal sample might have some material passed-up from its children
            # here, but we can replace that with everything passed up to parents
            if self.nodes[child].is_sample():
                chromosomes = {chrom: P.closedopen(0, np.inf) for chrom in old_iedges.chromosomes_as_child(child)}
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
                    child_ivl = P.closedopen(ie.child_left, ie.child_right)
                    is_inversion = ie.is_inversion()
                    # JEROME'S NOTE (from previous algorithm): if we had an index here
                    # sorted by left coord we could binary search to first match, and
                    # could break once c_right > left (I think?), rather than having
                    # to intersect all the intervals with the current c_rgt/c_lft
                    for intervals in chromosomes[child_chr]:
                        for interval in intervals:
                            if c := child_ivl & interval:  # intersect
                                p = old_iedges.transform_interval(ie_id, (c.lower, c.upper), Const.ROOTWARDS)
                                if is_inversion:
                                    # For passing up, interval orientation doesn't matter
                                    # so put lowest position first, as `portion` requires
                                    stack[parent][parent_chr] |= P.closedopen(p[1], p[0])
                                else:
                                    stack[parent][parent_chr] |= P.closedopen(*p)
                                # Add the trimmed interval to the new edge table
                                self.add_iedge_row(
                                    child,
                                    parent,
                                    child_left=c.lower,
                                    child_right=c.upper,
                                    parent_left=p[0],
                                    parent_right=p[1],
                                    child_chromosome=child_chr,
                                    parent_chromosome=parent_chr,
                                )
        self._remove_non_ancestral_mutations(ancestral_regions)
        if sort:
            self.sort()

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
            if is_sample[u] or x in ancestral_regions.get(u, {}).get(chrom, P.empty())
        ]
        if len(keep) < len(mutations):
            mutations._subset(keep)

    def remove_non_ancestral_nodes(self):
        """
        Subset the node table to remove any nonsample nodes that are not referenced
        in the iedges table. Mutations above removed nodes are also removed, and the
        node IDs of the remaining mutations are updated.
        """
        if not self.iedges.has_all_bitflags(ValidFlags.IEDGES_INTEGERS):
            raise ValueError("Requires edges to contain non-negative child & parent ids")
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

    def find_mrca_regions(self, u, v, *, u_chromosomes=None, v_chromosomes=None, time_cutoff=None):
        """
        Find all regions between nodes u and v (potentially restricted to a list
        of chromosomes in u and v) that share a most recent
        common ancestor in the GIG which is more recent than time_cutoff.
        Returns a dict of dicts of the following form

        .. code-block:: python

            {
                MRCA_node_ID1 : {(X, Y, CHR): (
                    [(uA, uB, uCHRa), (uC, uD, uCHRb), ...],
                    [(vA, vB, vCHR), ...]
                )},
                MRCA_node_ID2 : ...,
                ...
            }

        Where in each inner dict, the key (X, Y, CHR) gives an interval (with X < Y)
        in the MRCA node, and the value is a 2-tuple giving the corresponding
        intervals in u and v. In the example above there is a list of two
        corresponding intervals in u: (uA, uB, cCHRa) and (uC, uD, cCHRb)
        representing a duplication of the MRCA interval into u. If uA > uB then
        that interval in u is inverted relative to that in the MRCA node.

        Implementation-wise, this is similar to the sample_resolve algorithm, but

        #. Instead of following the ancestry of *all* samples upwards, we
           follow just two of them. This means we are unlikely to visit all
           nodes in the graph, and so we create a dynamic stack (using the
           sortedcontainers.SortedDict class, which is kept ordered by node time.
        #. We do not edit/change the edge intervals as we go. Instead we simply
           return the intervals shared between the two samples.
        #. The intervals "know" whether they were originally associated with u or v.
           If we find an ancestor with both u and v-associated intervals
           this indicates a potential common ancestor, in which coalescence could
           have occurred.
        #. We do not add older regions to the stack if they have coalesced.
        #. We have a time cutoff, so that we don't have to go all the way
           back to the "origin of life"
        #. We have to keep track of the offset of the current interval into the
           original genome, to allow mapping back to the original coordinates

        :param int u: The first node
        :param int v: The second node
        :param int u_chromosomes: Restrict the MRCAs search to this list of chromosomes
            from node ``u``. If None (default), return MRCAs for all ``u``'s chromosomes
        :param int v_chromosomes: Restrict MRCAs search to this list of chromosomes
            from node ``v``. If None (default), return MRCAs for all ``v``'s chromosomes
        """
        if u_chromosomes is None:
            u_chromosomes = list(self.iedges.chromosomes_as_child(u))
        if v_chromosomes is None:
            v_chromosomes = list(self.iedges.chromosomes_as_child(v))
        if not isinstance(u, (int, np.integer)) or not isinstance(v, (int, np.integer)):
            raise ValueError("u and v must be integers")

        if u == v:
            raise ValueError("u and v must be different nodes")

        if not self.iedges.has_all_bitflags(ValidFlags.IEDGES_PARENT_OLDER_THAN_CHILD):
            raise ValueError("Must guarantee each iedge has child younger than parent")

        if time_cutoff is None:
            time_cutoff = np.inf
        # Use a dynamic stack as we hopefully will be visiting a minority of nodes
        node_times = self.nodes.time
        stack = sortedcontainers.SortedDict(lambda k: -node_times[k])

        # We need to track intervals separately if they have been transformed
        # (inverted or coordinate-shifted). Within the stack, for each node we
        # therefore store a *dictionary* of portion objects, keyed by offset,
        # original chromosome, and orientation.
        #
        # We store *two* such {(offset, orientation): intervals} dicts. The first
        # tracks the ancestral regions of u, and the second the ancestral regions of v.
        # If a node has something in both dicts, intervals for u and v co-occur
        # (and could therefore coalesce), meaning the node is a potential MRCA.
        offset = 0
        # tracked intervals for node u, keyed by offset+orientation (False=not inverted)
        stack[u] = (
            {c: {(offset, c, False): P.closedopen(0, np.inf)} for c in u_chromosomes},
            {},
        )
        # tracked intervals for node v, keyed by offset+orientation (False=not inverted)
        stack[v] = (
            {},
            {c: {(offset, c, False): P.closedopen(0, np.inf)} for c in v_chromosomes},
        )

        result = MRCAdict()

        def concat(x, y):
            # combine a pair of sets in x and a pair of sets in y
            return (x[0] | y[0], x[1] | y[1])

        logging.debug(f"Checking mrca of {u} and {v}: {stack}")
        while len(stack) > 0:
            child, (u_dict, v_dict) = stack.popitem()
            if node_times[child] > time_cutoff:
                break
            if len(u_dict) > 0 and len(v_dict) > 0:
                # check for overlap between all intervals in u_dict and v_dict
                # which result in coalescence. The coalescent point can be
                # recorded, and the overlap deleted from the intervals
                #
                # Intervals are not currently sorted, so we need to check
                # all combinations. Note that we could have duplicate positions
                # even in the same interval list, due to segmental duplications
                # within a genome. We have to use the portion library here to
                # cut the MRCA regions into non-overlapping pieces which have
                # different patterns of descent into u and v
                for chrom in u_dict.keys() & v_dict.keys():
                    coalesced = P.IntervalDict()
                    # u_key and v_key are (offset, is_inverted) tuples
                    for u_key, u_intervals in u_dict[chrom].items():
                        for i, u_interval in enumerate(u_intervals):
                            for v_key, v_intervals in v_dict[chrom].items():
                                for j, v_interval in enumerate(v_intervals):
                                    if mrca := u_interval & v_interval:
                                        # we temporarily track the index of the intervals
                                        # that are unique to that specific chromosome and
                                        # (offset, chr, is_inverted) tuple. This avoids
                                        # double-counting intervals in the nested loop.
                                        # We replace these details with the
                                        # interval-plus-chromosome  on another pass
                                        details = ({(u_key, i)}, {(v_key, j)})
                                        coalesced = coalesced.combine(P.IntervalDict({mrca: details}), how=concat)
                    if len(coalesced) > 0:
                        # Work out the mapping of the mrca intervals into intervals in
                        # u and v, given keys into the uv_intervals dicts.
                        if child not in result:
                            result[child] = {}
                        for mrca_interval, uv_details in coalesced.items():
                            key = Segment(mrca_interval.lower, mrca_interval.upper, chrom)
                            if key not in result[child]:
                                result[child][key] = MRCAdict.MRCAintervals([], [])
                            for uv_list, details in zip(result[child][key], uv_details):
                                # Odd that we don't use the values() of `details` here:?
                                for uv_key, _ in details:
                                    (
                                        offset,
                                        orig_chr,
                                        inverted_relative_to_original,
                                    ) = uv_key
                                    if inverted_relative_to_original:
                                        uv_interval = Segment(
                                            offset - mrca_interval.lower,
                                            offset - mrca_interval.upper,
                                            orig_chr,
                                        )
                                    else:
                                        uv_interval = Segment(
                                            mrca_interval.lower - offset,
                                            mrca_interval.upper - offset,
                                            orig_chr,
                                        )
                                    uv_list.append(uv_interval)

                        # Remove the coalesced segments from the interval lists
                        for uv_dict in (u_dict, v_dict):
                            for key in uv_dict[chrom].keys():
                                uv_dict[chrom][key] -= coalesced.domain()

            # Transmit the remaining intervals upwards
            for ie_id in self.iedges.ids_for_child(child):
                # See note in sample resolve algorithm re: efficiency and
                # indexing by left coord. Note this is more complex in the
                # mrca case because we have to treat intervals with different
                # offsets / orientations separately
                ie = self.iedges[ie_id]
                parent = ie.parent
                child_ivl = P.closedopen(ie.child_left, ie.child_right)  # cache
                is_inversion = ie.is_inversion()
                for u_or_v, u_or_v_dict in enumerate((u_dict, v_dict)):
                    for chromosome, interval_dict in u_or_v_dict.items():
                        if chromosome != ie.child_chromosome:
                            continue
                        for (
                            offset,
                            orig_chr,
                            invrt,
                        ), intervals in interval_dict.items():
                            for interval in intervals:
                                if c := child_ivl & interval:
                                    p = self.iedges.transform_interval(ie_id, (c.lower, c.upper), Const.ROOTWARDS)
                                    if is_inversion:
                                        if invrt:
                                            # 0 gets flipped backwards
                                            x = offset - (c.lower + p[0])
                                        else:
                                            x = offset + (c.lower + p[0])
                                        invrt = not invrt
                                        parent_ivl = P.closedopen(p[1], p[0])
                                    else:
                                        x = offset - c.lower + p[0]
                                        parent_ivl = P.closedopen(*p)
                                    key = (x, orig_chr, invrt)
                                    if parent not in stack:
                                        stack[parent] = ({}, {})
                                    if ie.parent_chromosome not in stack[parent][u_or_v]:
                                        stack[parent][u_or_v][ie.parent_chromosome] = {}
                                    if key in stack[parent][u_or_v][ie.parent_chromosome]:
                                        stack[parent][u_or_v][ie.parent_chromosome][key] |= parent_ivl
                                    else:
                                        stack[parent][u_or_v][ie.parent_chromosome][key] = parent_ivl

        return result
