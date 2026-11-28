(sec_design_storage)=

# Internal storage

**Status: Implemented** (the "Current implementation" and "Public contract" sections describe
the code), **and Proposed** (the "Towards a bespoke core" section).
Related: {issue}`86` (speeding up forward simulations), {issue}`136` (immutable tables),
{issue}`138` (C and Rust APIs and core), {issue}`70` (iedge order), {issue}`71` (efficient
interval intersections).

## Summary

- Each table stores its fixed-width columns in a single preallocated numpy *structured array*,
  which grows by doubling. Variable-length ("ragged") columns, such as metadata, are stored in
  Python lists.
- Tables maintain *indexes* (e.g. the rows for each child) and *validity flags* as rows are
  added, so that algorithms can run directly on tables during a simulation, without first
  freezing them into a {class}`~giglib.Graph`.
- Columns are exposed as read-only numpy arrays. Rows are returned as named tuples created on
  access, and a {class}`~giglib.Graph` wraps a frozen set of tables, adding further indexes.
- This is a proof-of-concept implementation in Python. The likely endpoint is a bespoke core
  (probably in Rust). The public API, and the behaviour tests in the test suite, are intended
  to survive that change. The storage described here, and the tests in
  `tests/test_implementation.py`, are not.

## History

The storage has been rewritten several times, mainly for speed in forward simulations:

1. **Lists of objects** (September 2023). Each table was a Python list of row objects
   (initially dataclasses), with column arrays built on demand. This was simple, but slow:
   profiling forward simulations ({issue}`86`) showed much of the time going on creating and
   converting row objects.
2. **Validation on entry** (February 2024). Tables gained validity flags, checked as rows are
   added, and an index of the iedges for each child. This allowed algorithms such as
   `find_homology` to run directly on tables, rather than on a `Graph`, so simulations
   don't need to repeatedly freeze their tables. Tables can also be frozen, to make a `Graph`
   immutable. Iedges were also reordered to be grouped by child, in order of decreasing child
   time ({issue}`70`).
3. **numpy structured arrays** (March 2024). First for the iedges table, then for all tables.
   Rows are written into a preallocated array, which grows by doubling, so adding a row is
   cheap and columns are numpy arrays without copying. Initial sizes can be given when
   creating tables (e.g. when converting from _tskit_, where they are known), and chromosome
   columns use `int16` to save space. Sorting became in-place (April 2024).
4. **Mutations and read-only columns** (November 2026). A mutations table was added
   ({pr}`148`), with an index of the mutations for each node. Column accessors were then
   made to return read-only views ({pr}`151`, and {pr}`150` for mutations), so that indexes
   and validity flags cannot be silently invalidated by editing a column (see below).

## Current implementation

### Fixed-width columns

Each table class (a subclass of `BaseTable` in `giglib/tables.py`) has a row class (a named
tuple) whose fields define the columns. The fixed-width columns are stored together in a numpy
structured array, `_datastore`, with one field per column. Columns are `int64` by default, with
some exceptions (e.g. `float64` for times, `uint32` for flags, and `int16` for chromosomes).

- **Preallocation and growth.** `_datastore` is preallocated (64 rows by default, or the size
  given in `initial_sizes`). `_data` is a view of its first `len(table)` rows. Adding a row
  writes into the next free row of `_datastore` and extends the `_data` view. When
  `_datastore` is full, it is replaced by one twice the size plus one row (so that a table
  preallocated with 0 rows can grow), or at most 2<sup>18</sup> rows larger, and the existing
  rows are copied across.
- **Row access.** `table[i]` creates a row object (a named tuple) from row `i`. This is
  convenient, but much slower than reading columns, so performance-critical code should use
  columns.
- **Column access.** Column properties (e.g. `tables.iedges.child`) return read-only views
  of `_data`. To keep this cheap, read-only views of each whole column of `_datastore` are created
  only when `_datastore` is *replaced* (detected in `BaseTable.__setattr__`), and each access
  slices one of these to the length of the table. This takes about 0.12 µs per access,
  compared with about 0.07 µs for a writable view.
- **Sorting** reorders `_datastore` in place (or into a new, exactly-sized array, with
  `sort(shrink_memory=True)`).

### Ragged columns

Columns with variable-length values (e.g. node and mutation `metadata`, individual `parents` and
`location`, and mutation `derived_state`) are stored in a Python list, `_extra_data`, with one
list of values per row. These are listed in the table class's `_extra_names`.

### Indexes

Tables and graphs maintain indexes, to avoid scanning whole tables:

- **Iedges for each child** (`IEdgeTable._id_range_for_child`): a dict mapping each child to a
  dict of `{chromosome: [start, end]}` row ranges. It is updated as each row is added (this is
  only meaningful if iedges are grouped by child, which the validity flags record), and rebuilt
  by `Tables.sort()`. It is used by `ids_for_child()` and many algorithms.
- **Mutations for each node** (`MutationTable._id_range_for_node`): the equivalent for
  mutations. It is updated as each row is added, with an internal flag
  (`_node_ranges_valid`) recording whether mutations are still grouped by node, and ordered
  by chromosome then position within each node. It is rebuilt when rows are reordered or
  removed (e.g. by sorting), when copying, and (as a safeguard against edits through the
  internal arrays) when a `Graph` is created. It is used by `ids_for_node()`, and when
  decoding alleles (via `_range_for_node()`, which avoids creating an array of IDs).
- **Iedges for each parent** (`Graph._iedge_map_sorted_by_parent` and
  `Graph._id_range_for_parent`): a permutation of the iedges sorted by parent, and the ranges
  for each parent and chromosome. These are only built when a `Graph` is created, so there is
  not yet an equivalent at the tables level.
- **Sample IDs** are cached when a `Graph` is created.

### Validity flags

The iedges table has a bitmask of {class}`~giglib.ValidFlags`, each recording a guarantee about
the table (e.g. that iedges for each child are sorted, or that parents are older than children).
They are checked by {meth}`~giglib.Tables.validate`, which works on whole columns, and only checks
the rows added since it was last called (the table records how many rows have been checked).
Each table algorithm checks for
the guarantees it needs, using `has_all_bitflags()`. Creating a `Graph` requires all of them. See
{ref}`sec_validation`.

### Freezing

`Tables.freeze()` (called when creating a `Graph`) makes each `_datastore` non-writeable, and
prevents attributes being set on the tables, so a `Graph` is immutable. Freezing is permanent:
to edit, use `gig.dump_tables()`, which returns an editable copy (`Tables.copy()`). Every attempt to
edit frozen tables (adding rows, sorting, table algorithms, ...) raises the same error, saying so.

**No separate immutable type ({issue}`136`).** Here, freezing makes one unnecessary: `gig.tables`
is a frozen `Tables`, which refuses every edit with a clear error, so a second class mirroring the
reading parts of `Tables` would add more to keep consistent, for no extra safety. This is likely to
change with a compiled core ({ref}`sec_design_storage_core`), as it did in _tskit_. There, a tree
sequence's tables are held in C memory, so before version 1.0, `ts.tables` returned a copy (slow
for large tree sequences, and not even read-only). Since 1.0, it returns an
`ImmutableTableCollection`, a zero-copy, read-only view of that memory, and `ts.dump_tables()`
returns an editable copy. A compiled core could instead check a frozen flag in each editing method,
as now, but a read-only view type is the natural design. Either way, `gig.dump_tables()` would stay
the way to get editable tables.

## Public contract

These are the parts that code using _giglib_ may rely on, and so that any new implementation
must preserve:

| | Promise |
|---|---|
| Columns | Each column is available as a numpy array (e.g. `tables.iedges.child`), which cannot be edited in place |
| Rows | `table[i]` returns a row object with named fields; `len(table)` is the number of rows |
| Adding rows | `add_row()` returns the new row's ID, and refuses non-integer iedge values |
| Queries | Methods on tables return IDs (e.g. `ids_for_child`, `filter_nodes`); methods on a `Graph` return objects |
| Validity | `ValidFlags`, `has_all_bitflags()`, `set_bitflag()` and `unset_bitflag()` |
| Immutability | A `Graph`, and the tables it wraps, cannot be changed |
| Copying and comparing | `copy()` returns independent, unfrozen tables; `==` compares contents |

Everything else is internal, including `_datastore`, `_data`, `_extra_data`, preallocation and
growth, the index data structures, and the read-only views. Some parameters, such as
`initial_sizes` and `sort(shrink_memory=True)`, are public but only affect performance, so a new
implementation could accept and ignore them.

## Testing

Tests of implementation details (e.g. preallocated sizes, or the contents of the internal
indexes) are kept in `tests/test_implementation.py`. These are expected to be rewritten or
removed if the storage is reimplemented. All other tests should only use the public API, so
that they can check any new implementation. In particular, tests comparing with _tskit_ (e.g.
haplotypes against `ts.alignments()`, and round trips) are a good check of correctness that is
independent of the storage.

## Known limitations

- **No saving or loading.** There is no file format: tables can only be created in memory, or
  converted from _tskit_.
- **Ragged columns are Python lists,** which are slow and memory-hungry for large tables, and
  cannot be exposed as numpy arrays.
- **Adding rows is Python-speed** (about 3 µs per iedge, including optional validation), which
  limits forward simulations ({issue}`86`).
- **No bulk addition of rows.** An `add_rows()` method, which added many rows in one call
  (broadcasting values such as `flags=0`), was dropped when the tables were converted to numpy
  arrays in March 2024, and has not been reimplemented. A skipped test (`test_add_rows` in
  `tests/test_tables.py`) records the intended behaviour.
- **Row and `Graph` objects are created on every access.**
- **The parent index only exists on a `Graph`.**
- **Sequence lengths are not stored**, but derived from the iedges, so a node with no iedges
  has length 0, even if it has mutations.
- **No immutable-tables interface** ({issue}`136`): `gig.tables` returns the (frozen) tables
  themselves.

(sec_design_storage_core)=

## Towards a bespoke core

**Status: Proposed.** The likely endpoint is a bespoke core, probably written in Rust (or
possibly C: see {ref}`below<sec_design_storage_language>`), with a Python API on top. It should
be able to:

- **Append rows fast,** with the incremental index and validity checks described above, since
  forward simulations add rows and run algorithms in alternation.
- **Append many rows at once,** from numpy arrays, growing the storage once.
- **Expose columns to Python without copying,** as read-only numpy arrays.
- **Store ragged columns** as a single data array plus offsets, as _tskit_ does.
- **Support interval queries efficiently,** e.g. finding the iedges above a node that overlap an
  interval ({issue}`71`), and maintain the parent index at the tables level.
- **Save and load,** in a documented file format.
- **Freeze tables,** to give an immutable `Graph`.

(sec_design_storage_language)=

### Language

The choice of language is discussed in {issue}`138`, and has not been made. The current
preference is Rust, mainly because memory allocation bugs are hard to track down in C (as in
_tskit_'s C library). Rust can also expose a C interface, which would make the core usable from
other languages (e.g. R). The issue also agrees that the Python side should be developed further
first, so that the core is designed around algorithms that have settled.

### Suggested path

Keep the public contract above fixed, and replace the storage behind it gradually:

1. **Python stays the reference implementation.** Develop and test new algorithms in Python,
   using the behaviour tests as the oracle for any later port.
2. **Move storage primitives first.** Start with creating tables, appending rows, validating,
   reading columns without copying, and saving and loading, one table at a time. Rewrite
   `tests/test_implementation.py` for the new storage, and check each step with the behaviour
   tests.
3. **Port algorithms only once they are stable,** and only if they need the speed or control
   over memory.
4. **Add a C interface last,** once the core has settled, as a thin layer for use from other
   languages.

### Open questions

- Rust or C ({issue}`138`)? Either way, is it better to build on _tskit_'s C tables (which also
  have Rust bindings) and its `kastore` file format, extending them for iedges and chromosomes,
  or to write a bespoke core from scratch?
- Should the indexes currently built when a `Graph` is created (e.g. the parent index) be
  maintained by the tables instead?
- Should bulk addition (`add_rows()`) be added to the current numpy storage, or wait for the
  core? For nodes and individuals it is simple (grow once, assign whole columns). For iedges,
  validation already works on blocks of new rows (see {ref}`sec_validation_incremental`), but the
  index of iedges for each child is updated one row at a time.
- How should metadata be handled (e.g. schemas, as in _tskit_)?
