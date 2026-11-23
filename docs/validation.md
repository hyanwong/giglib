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
- **Guarantees are checked with {meth}`~giglib.Tables.validate`,** which only checks the rows
  added since it was last called, so it is cheap to call often (see
  {ref}`sec_validation_incremental`). Adding rows makes no checks (apart from refusing
  non-integer values).

The flags currently only concern the iedges table, as that is where the important guarantees
are (see {ref}`sec_validation_other` for the other checks).

## The flags

Each flag records one guarantee about the iedges table:

| Flag | Guarantees | Needs node times? |
|---|---|---|
| `IEDGES_INTEGERS` | Coordinates are non-negative. (All values are integers, as adding a row refuses others. Chromosome IDs may be negative, e.g. for circular chromosomes, and the edge ID may be `NULL`.) | |
| `IEDGES_INTERVALS` | Each iedge's child and parent intervals have the same length | |
| `IEDGES_CHILD_INTERVAL_POSITIVE` | `child_left < child_right` (only parent intervals can be reversed, for inversions) | |
| `IEDGES_FOR_CHILD_NONOVERLAPPING` | Each position of a child's chromosome has at most one parent | |
| `IEDGES_PARENT_OLDER_THAN_CHILD` | Every parent is older than its child (so the graph is acyclic), and both are nodes in the node table (i.e. the node IDs are valid) | yes |
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
  and ordered by chromosome and then position. Needed e.g. by
  {meth}`~giglib.tables.IEdgeTable.squash`.
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
| `tables.iedges.squash()` | `IEDGES_WITHIN_CHILD_SORTED` |
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

**Adding a row** makes no checks (apart from refusing non-integer values), so the table can no
longer guarantee anything, and the iedge flags are removed:

```{code-cell}
tables.iedges.add_row(child, parent, child_right=100)
print(tables.iedges.flags == 0)
```

**Validating** with {meth}`~giglib.Tables.validate` checks the requested guarantees (by default,
all of them), setting the flag of each, and raising an error for the first that doesn't hold:

```{code-cell}
tables.validate()
print(tables.iedges.flags == ValidFlags.GIG)

tables.iedges.add_row(child, parent, child_left=50, child_right=150)  # overlaps the first
try:
    tables.validate()
except ValueError as e:
    print("Error:", e)
```

Validation only checks the rows added since it was last called (see
{ref}`sec_validation_incremental`), so it is cheap to call after adding a few rows, however large
the tables are.

Other ways that flags are set:

- {meth}`~giglib.Tables.sort` sorts the iedges, and sets the ordering flags.
- {meth}`~giglib.Tables.from_tree_sequence` sets the flags guaranteed by a valid tree sequence
  ({data}`~giglib.ValidFlags.IEDGES_FROM_TSKIT`).
- {meth}`~giglib.tables.IEdgeTable.set_bitflag` sets flags for the whole table *without checking
  anything*. It is a promise, which algorithms then trust (see below).
  {meth}`~giglib.tables.IEdgeTable.unset_bitflag` removes flags.

Algorithms refuse to run if the flags they need are missing, even if the tables happen to be
valid, and the error names the missing flags:

```{code-cell}
tables.iedges.clear()
tables.iedges.add_row(child, parent, child_right=100)
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

(sec_validation_incremental)=

## Why only new rows need checking

Validation is incremental: the tables record how many iedges have been checked (the
{ref}`validated rows<sec_validation_validated_rows>`), and for which requirements. When
{meth}`~giglib.Tables.validate` is next called, a requirement that already held for the validated
rows is only checked for the new rows; one that wasn't known to hold is checked for all the rows.

This is enough because adding rows only *appends* to the table: existing rows don't move or
change. Adding a row can certainly make the table invalid (e.g. a new iedge whose child is older
than the previous iedge's child means the iedges are no longer ordered by child time), but **any
new violation must involve a new row**, so only the new rows need checking. That's true of each
kind of requirement:

- **Requirements about a single iedge** (integer coordinates, equal spans,
  `child_left < child_right`, the parent being older than the child): an existing iedge doesn't
  change, so still satisfies them.
- **Requirements comparing each iedge with the previous one** (the ordering requirements, e.g.
  children ordered by time, or each child's iedges by chromosome and position): pairs of existing
  iedges are unchanged, so the only pairs to check are the last existing iedge with the first new
  one, and pairs of new iedges.
- **Requirements over a group of iedges:** the iedges of a child must be adjacent, and must not
  overlap, and iedges with the same edge ID must have the same parent and child. A new violation
  needs a new iedge in the group, so only groups containing new iedges are checked (using the
  index of the iedges for each child, or the edge IDs of the new iedges).

Operations that reorder or change existing iedges, such as {meth}`~giglib.Tables.sort`,
{meth}`~giglib.tables.IEdgeTable.squash` or {meth}`~giglib.tables.IEdgeTable.clear`, set the
flags that they guarantee for all the rows.

**Adding nodes** doesn't break any requirement either: it doesn't change the times of existing
nodes, or which node an existing ID refers to, and the node table has no order of its own. But
the node table *can* be changed in other ways, e.g. by being cleared and refilled, or by
{meth}`~giglib.Tables.change_times`. So the node table records whenever its existing rows
change, and the requirements that depend on node times are then checked again for all the iedges.
Algorithms that rely on these flags (e.g. {meth}`~giglib.Tables.find_mrca_regions`) check them
again in the same way.

The cost of {meth}`~giglib.Tables.validate` is therefore proportional to the number of new rows,
plus a fixed cost of about 50 µs per call. Checking costs about 0.5 µs per iedge when many rows are
checked at once (e.g. when creating a `Graph` from tables built without validation).

(sec_validation_validated_rows)=

## Validated rows

The **validated rows** are the first rows of the iedges table, which were checked when
{meth}`~giglib.Tables.validate` was last called (their number is
{attr}`tables.iedges.num_validated<giglib.tables.IEdgeTable.num_validated>`). Rows added since then
are not validated, so the flags of the whole table ({attr}`~giglib.tables.IEdgeTable.flags`) are
removed until they are. But the flags still hold for the validated rows, and methods that only read
the iedges of particular nodes can use them:

- {meth}`~giglib.tables.IEdgeTable.ids_for_child`, {meth}`~giglib.tables.IEdgeTable.max_pos_as_child`
  and {meth}`~giglib.tables.IEdgeTable.min_pos_as_child` run if all the iedges of the node are
  validated rows.
- {meth}`~giglib.Tables.find_mrca_regions` runs if the iedges of every node that it traces through
  are validated rows. Its error names the first node whose iedges are not.

Methods that use the whole table (e.g. {meth}`~giglib.Tables.sample_resolve`, or creating a
`Graph`) still need all the rows to be validated.

This is safe even if a later row adds to the iedges of a node whose earlier iedges were validated:
the index of the iedges for each child records the last row of each node, so the method then
refuses to read that node's iedges. If the node table has changed since the requirements that
depend on node times were checked, they are checked again for the validated rows only, so
unvalidated rows cause no errors until they are validated.

```{code-cell}
tables = gigl.Tables()
root = tables.nodes.add_row(2)
mum = tables.nodes.add_row(1)
dad = tables.nodes.add_row(1)
tables.iedges.add_row(mum, root, child_right=100)
tables.iedges.add_row(dad, root, child_right=100)
tables.validate()
kid = tables.nodes.add_row(0)
tables.iedges.add_row(kid, mum, child_right=50)  # not yet validated
print(tables.iedges.num_validated, "of", len(tables.iedges), "rows validated")
print("MRCA of mum and dad:", list(tables.find_mrca_regions(mum, dad).keys()))
try:
    tables.find_mrca_regions(kid, dad)
except ValueError as e:
    print("Error:", e)
```

(sec_validation_choosing)=

## Choosing how to validate

There are two approaches:

1. **Add rows, then validate** with {meth}`~giglib.Tables.validate` before running an algorithm
   that needs the flags. This catches any mistakes, and as only the new rows are checked, can be
   done as often as needed: e.g. once all of a child's iedges have been added, or after each row
   (to find out exactly which row breaks a requirement, when writing or debugging code).
2. **Skip the checks** with {meth}`~giglib.tables.IEdgeTable.set_bitflag`. The flags are set
   without any checking, so this avoids the cost of validating, but it is only safe if the code
   adding the rows is known to be correct. If that code later changes, and breaks a guarantee,
   nothing will notice: algorithms that trust the flags may quietly give wrong answers.

Creating a `Graph` uses approach 1, so is cheap for tables that are already validated (e.g.
converted from a tree sequence).

### Forward simulations

In a forward simulation, adding rows and running algorithms alternate. For each child,
homologous positions in its parents are located (e.g. with
{meth}`~giglib.Tables.find_mrca_regions`) to choose recombination breakpoints, and then the
child's iedges are added.

Locating homologous positions only reads *older* material: the ancestry of the parents, whose
iedges were added in earlier generations. So it is enough to validate once per generation (e.g.
before adding the next generation): the parents' iedges are then
{ref}`validated rows<sec_validation_validated_rows>`, and the iedges of the children being added
don't need to be. This only checks each iedge once, and the fixed cost is only paid once per
generation.

### Other algorithms on tables

Simulations also run other algorithms on their tables, e.g. regular simplification. What these
need, and the flags they leave on their output:

| Method | Requires | Leaves |
|---|---|---|
| {meth}`~giglib.Tables.sample_resolve` | `IEDGES_FOR_TRACING` | the input's order-independent flags, plus the ordering flags (from sorting) |
| {meth}`~giglib.Tables.decapitate` | none, unless there are mutations to keep (with the default `discard_old_mutations=False`), when the tables must form a valid GIG | the input's order-independent flags, plus the ordering flags |
| {meth}`~giglib.Tables.remove_non_ancestral_nodes` | `IEDGES_PARENT_OLDER_THAN_CHILD` (so that node IDs are valid) | all the flags |
| {meth}`~giglib.Tables.sort` | (none) | adds the ordering flags |

The "order-independent" flags ({data}`~giglib.ValidFlags.IEDGES_ORDER_INDEPENDENT`) are the
guarantees that still hold if iedges are removed or reordered, or trimmed to part of their
interval (at integer positions, with the child and parent intervals trimmed to corresponding
positions), with node times unchanged. An example is that parents are older than children.
`sample_resolve()` and `decapitate()` build their output from the input iedges in this way, so
they keep any of these flags that the input had ({issue}`181`). Those that depend on node times
are rechecked first if the node table has changed since they were checked.

So, given valid tables, they give valid tables, and algorithms such as
{meth}`~giglib.Tables.find_mrca_regions` can be run straight after simplifying, without
validating again. This relies on the sorting that both methods do by default: with
`sample_resolve(sort=False)`, the ordering flags are not set, so call {meth}`~giglib.Tables.sort`
before running algorithms that need them.

Locating homologous positions efficiently is also the purpose of the proposed "homology table"
({issue}`140`; see {ref}`sec_design_simplification`), which would need the same guarantees.

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

- **Keeping flags through future table algorithms:** like `sample_resolve()` and `decapitate()`,
  node-removing simplification should keep the flags that its output satisfies by construction.
- **More flags:** for mutations ({issue}`182`), e.g. that their times are valid and in order,
  and that they are grouped and ordered by node. And perhaps for "every position of a non-root
  genome has a parent" (see {ref}`sec_design_insertions`).
