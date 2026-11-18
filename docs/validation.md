---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(sec_validation)=

# Validation

_giglib_ checks the validity of a GIG differently from _tskit_. This page explains why, what is
checked, and how to choose between the ways of checking.

## The idea

In _tskit_, a set of tables is either a valid tree sequence or not. Validity is checked all at
once, when the tables are turned into a tree sequence (`tables.tree_sequence()`), and algorithms
generally run on the tree sequence, not on the tables.

_giglib_ instead allows algorithms to run directly on a set of {class}`~giglib.Tables`, e.g. while
a forward simulation is adding rows to them, without the cost of repeatedly freezing them into a
{class}`~giglib.Graph`. The main reason is the need to **locate homologous positions during a
forward simulation**. With no shared coordinate system, choosing a recombination breakpoint means
finding equivalent positions in the two parental genomes, by tracing them up the graph (e.g. with
{meth}`~giglib.Tables.find_mrca_regions`) every time a child is created. In a tree sequence this
problem doesn't arise, as every genome uses the same coordinates. Simulations also need to
simplify their tables regularly, without turning them into a `Graph` (just as _tskit_'s
{meth}`tskit:tskit.TableCollection.simplify` works on tables, not tree sequences). But
algorithms on tables can only give correct answers if the tables satisfy certain properties, such
as parents being older than their children.

So _giglib_ tracks validity as a set of **separate guarantees**, each recorded by a flag (see
{class}`~giglib.ValidFlags`), rather than as a single yes or no. Each guarantee is an
**invariant** of the tables: a property that must keep holding as rows are added, if algorithms
that rely on it are to keep giving correct answers.

- **Each algorithm checks for just the guarantees it needs,** and refuses to run without them.
  For example, {meth}`~giglib.Tables.find_mrca_regions` needs what is required to trace intervals
  up through the iedges ({data}`~giglib.ValidFlags.IEDGES_FOR_TRACING`: parents older than their
  children, valid intervals, and the iedges for each child adjacent and ordered by chromosome),
  but not e.g. the full canonical sort order.
- **A {class}`~giglib.Graph` requires all the guarantees** ({data}`~giglib.ValidFlags.GIG`).
  Creating one checks any guarantees that are not already flagged, and then freezes the tables.
- **Guarantees can be checked as rows are added**, one row at a time, or **later, in bulk**, with
  {meth}`~giglib.Tables.validate`.

The flags currently only concern the iedges table, as that is where the important guarantees
are (see {ref}`sec_validation_other` for the other checks).

## The flags

Each flag records one guarantee about the iedges table:

| Flag | Guarantees | Needs node times? |
|---|---|---|
| `IEDGES_INTEGERS` | All values are integers, and node IDs and coordinates are non-negative (chromosome IDs may be negative, e.g. for circular chromosomes, and the edge ID may be `NULL`) | |
| `IEDGES_INTERVALS` | Each iedge's child and parent intervals have the same length | |
| `IEDGES_CHILD_INTERVAL_POSITIVE` | `child_left < child_right` (only parent intervals can be reversed, for inversions) | |
| `IEDGES_FOR_CHILD_NONOVERLAPPING` | Each position of a child's chromosome has at most one parent | |
| `IEDGES_PARENT_OLDER_THAN_CHILD` | Every parent is older than its child (so the graph is acyclic), and both are nodes in the node table | yes |
| `IEDGES_SAME_PARENT_CHILD_FOR_EDGE` | iedges with the same edge ID have the same parent and child | |
| `IEDGES_FOR_CHILD_ADJACENT` | The iedges for each child are next to each other in the table | |
| `IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC` | ... and ordered by chromosome | |
| `IEDGES_FOR_CHILD_SECONDARY_ORDER_LEFT_ASC` | ... and then by `child_left` | |
| `IEDGES_PRIMARY_ORDER_CHILD_TIME_DESC` | Children are ordered from oldest to youngest | yes |
| `IEDGES_SECONDARY_ORDER_CHILD_ID_ASC` | ... and children of equal time by ID | yes |

The first five define what a GIG *is*. The others define a canonical order of the iedges, which
algorithms rely on for efficiency (e.g. to find the iedges above a node quickly).

Some useful combinations are also defined, such as:

- {data}`~giglib.ValidFlags.GIG` (the same as `IEDGES_ALL`): everything, as needed to create a
  `Graph`.
- {data}`~giglib.ValidFlags.IEDGES_WITHIN_CHILD_SORTED`: the iedges for each child are adjacent,
  and ordered by chromosome and then position. Needed e.g. by `tables.iedges.squash()`
  (added in {pr}`176`).
- {data}`~giglib.ValidFlags.IEDGES_SORTED`: the full canonical order, as produced by
  {meth}`~giglib.Tables.sort`.
- {data}`~giglib.ValidFlags.IEDGES_FOR_TRACING`: what is needed to trace intervals up through
  the iedges of a set of tables (parents older than children, valid intervals, and the iedges for
  each child adjacent and ordered by chromosome).
- {data}`~giglib.ValidFlags.IEDGES_COMBO_NODE_TABLE`: the flags that need the node table.

Some examples of what methods require:

| Method | Requires |
|---|---|
| `tables.iedges.ids_for_child()` | `IEDGES_FOR_CHILD_ADJACENT`, `IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC` |
| `tables.find_mrca_regions()` | `IEDGES_FOR_TRACING` |
| `tables.sample_resolve()` | `IEDGES_FOR_TRACING` |
| `tables.iedges.squash()` ({pr}`176`) | `IEDGES_WITHIN_CHILD_SORTED` |
| `tables.graph()` | `GIG` (everything) |

## How flags are set and unset

```{code-cell}
import giglib as gigl
from giglib import ValidFlags

tables = gigl.Tables()
parent = tables.nodes.add_row(1)
child = tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
print(tables.iedges.flags == ValidFlags.GIG)  # an empty table satisfies everything
```

**Adding a row without validation** removes all the iedge flags, as the table can no longer
guarantee anything:

```{code-cell}
tables.iedges.add_row(child, parent, child_right=100)
print(tables.iedges.flags == 0)
```

**Adding a row with validation** checks the requested guarantees for that row, and keeps them
(if they held before). Guarantees that need node times must be checked using
{meth}`~giglib.Tables.add_iedge_row`, which can see the node table:

```{code-cell}
tables.iedges.clear()
tables.add_iedge_row(child, parent, child_right=100, validate=ValidFlags.GIG)
print(tables.iedges.flags == ValidFlags.GIG)

try:
    tables.add_iedge_row(child, parent, child_left=50, child_right=150, validate=ValidFlags.GIG)
except ValueError as e:
    print("Error:", e)
```

**Bulk validation** with {meth}`~giglib.Tables.validate` checks the whole table, setting the
flag of each guarantee that holds, and raising an error for the first that doesn't. Guarantees
that are already flagged are not checked again, unless `recheck=True`. The exception is the
guarantees that depend on node times (e.g. parents being older than children), which are always
checked again, as the node table can change without the iedges table knowing (e.g. by being
cleared and refilled):

```{code-cell}
tables.iedges.add_row(child, parent, child_left=100, child_right=200)  # removes the flags
tables.validate()  # checks everything, and sets all the flags again
print(tables.iedges.flags == ValidFlags.GIG)
```

Other ways that flags are set:

- {meth}`~giglib.Tables.sort` sorts the iedges, and sets the ordering flags.
- {meth}`~giglib.Tables.from_tree_sequence` sets the flags guaranteed by a valid tree sequence
  ({data}`~giglib.ValidFlags.IEDGES_FROM_TSKIT`), and checks integer coordinates as it adds rows.
- {meth}`~giglib.tables.IEdgeTable.set_bitflag` sets flags *without checking anything*. It is a
  promise, which algorithms then trust (see below). {meth}`~giglib.tables.IEdgeTable.unset_bitflag`
  removes flags.

Algorithms refuse to run if the flags they need are missing, even if the tables happen to be
valid, and the error names the missing flags:

```{code-cell}
other_child = tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
tables.iedges.add_row(other_child, parent, child_right=300)  # flags removed
try:
    tables.find_mrca_regions(child, other_child)
except ValueError as e:
    print("Error:", e)
```

Validating just what the algorithm needs is enough:

```{code-cell}
tables.validate(ValidFlags.IEDGES_FOR_TRACING)
mrcas = tables.find_mrca_regions(child, other_child)
print("MRCA nodes:", list(mrcas.keys()))
```

(sec_validation_choosing)=

## Choosing how to validate

There are three approaches, with different costs and risks:

1. **Validate each row as it is added** (`add_row(..., validate=...)`). This is the slowest
   (adding an iedge takes about 3 µs, including the checks), but reports an error at the exact
   row that breaks a guarantee, which is useful when writing or debugging code.
2. **Add rows without validation, then validate in bulk** with {meth}`~giglib.Tables.validate`,
   just before running an algorithm that needs the flags. Checks are made on whole columns, so
   this is much cheaper per iedge (about 0.2 µs, for all the checks), and still catches any
   mistakes. But each call currently checks the *whole* table, so this only pays off when many
   rows are added between algorithm calls (e.g. building tables, then converting them to a
   `Graph`).
3. **Skip the checks,** with `add_row(..., validate=..., skip_validate=True)` or
   {meth}`~giglib.tables.IEdgeTable.set_bitflag`. The flags are set without any checking, so this
   is the fastest, but it is only safe if the code adding the rows is known to be correct. If that
   code later changes, and breaks a guarantee, nothing will notice: algorithms that trust the
   flags may quietly give wrong answers.

Creating a `Graph` uses approach 2, so is cheap for tables that are already flagged (e.g.
converted from a tree sequence).

### Forward simulations

In a forward simulation, adding rows and running algorithms alternate. For each child,
homologous positions in its parents are located (e.g. with
{meth}`~giglib.Tables.find_mrca_regions`) to choose recombination breakpoints, and then the
child's iedges are added. Adding a row without validation removes the
flags, so the next algorithm call would refuse to run. Approach 2 would then mean checking the
whole table before every algorithm call, which costs more than validating each row. So at present,
simulations need approach 1 (or, once the code is well tested, approach 3).

Two improvements would make bulk validation suitable for simulations (see
{ref}`sec_validation_improvements`):

- **Incremental validation**, checking only the rows added since the last call, so that the cost
  of validating before each algorithm call depends on the number of new rows, not on the size of
  the table.
- **A validated prefix of the table.** Locating homologous positions only reads *older* material:
  the ancestry of the parents, whose iedges were added in earlier generations. So the tables could
  record that rows `0` to `k` are validated, and algorithms that only touch those rows could run
  while newer rows are still unchecked.

### Other algorithms on tables

Simulations also run other algorithms on their tables, e.g. regular simplification. What these
currently need, and the flags they leave on their output (given valid input):

| Method | Requires | Leaves |
|---|---|---|
| {meth}`~giglib.Tables.sample_resolve` | `IEDGES_FOR_TRACING` | only the ordering flags |
| {meth}`~giglib.Tables.decapitate` | (none checked) | only the ordering flags |
| {meth}`~giglib.Tables.remove_non_ancestral_nodes` | `IEDGES_INTEGERS` | all the flags |
| {meth}`~giglib.Tables.sort` | (none) | adds the ordering flags |

So after `sample_resolve()` or `decapitate()`, the tables lose guarantees (e.g. that parents are
older than children) that in fact still hold, as the output is built from valid input. Until these
methods keep the flags that their output satisfies by construction ({issue}`181`), the tables
must be validated again (e.g. with {meth}`~giglib.Tables.validate`) before running algorithms that
need them.

Locating homologous positions efficiently is also the purpose of the proposed "homology table"
({issue}`140`; see {ref}`sec_design_simplification`), which would need the same guarantees.

Until then, approach 1 is recommended while developing, and approach 2 wherever many rows are
added at once. Approach 3 is best avoided unless the speed is really needed.

One thing can *only* be checked as rows are added: whether values are integers. Iedge
coordinates and IDs are stored as integers, so a non-integer value (e.g. `10.5`) is silently
truncated when added. Validating `IEDGES_INTEGERS` in bulk can therefore only check that the
stored node IDs and coordinates are non-negative.

(sec_validation_other)=

## Other checks

Some properties have no flag yet, and are always checked when creating a `Graph`. Flags for
mutations are planned ({issue}`182`):

- The mutations refer to existing nodes, and are grouped by node and ordered by chromosome and
  position (as after {meth}`~giglib.Tables.sort`). See {ref}`sec_design_mutations`.
- Multi-character alleles that cannot be complemented are flagged with a warning, if the GIG has
  inversions (see {ref}`sec_design_mutations`).

(sec_validation_improvements)=

## Possible improvements

- **Incremental bulk validation:** {meth}`~giglib.Tables.validate` could check only the rows
  added since it was last called. Most checks only need the new rows and the row before them (the
  edge ID check is the exception), so this would make approach 2 cost little more than the rows
  being added, even when called before every algorithm in a simulation.
- **A validated prefix:** recording which rows are validated (rather than flags for the whole
  table), so that algorithms that only read older rows can run while newer rows are unchecked
  (see {ref}`sec_validation_choosing`).
- **Retiring `skip_validate`:** if incremental validation makes approach 2 fast enough in
  simulations, the option to set flags without checking could be removed, so that flags can only
  be set by checking (or by an explicit `set_bitflag()`).
- **Refusing non-integer values** when rows are added without validation, so that
  `IEDGES_INTEGERS` can be fully checked in bulk.
- **Tracking changes to the node table:** clearing and refilling the node table can break the
  guarantees that depend on node times. {meth}`~giglib.Tables.validate` (and so creating a
  `Graph`) always checks these again, but algorithms that only look at the flags (e.g.
  {meth}`~giglib.Tables.find_mrca_regions`) do not. Changes to the node table could instead
  remove those flags from the iedges table.
- **Keeping flags through table algorithms:** {meth}`~giglib.Tables.sample_resolve` and
  {meth}`~giglib.Tables.decapitate` (and, later, simplification) could keep the flags that their
  output satisfies by construction, given valid input, rather than only the ordering flags
  ({issue}`181`).
- **More flags:** for mutations ({issue}`182`), e.g. that their times are valid and in order,
  and that they are grouped and ordered by node. And perhaps for "every position of a non-root
  genome has a parent" (see {ref}`sec_design_insertions`).
