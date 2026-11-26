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

(sec_usage)=

# Usage

The fundamental data structures and API provided by _giglib_
intentionally mirror that of [tskit](https://tskit.dev/learn/) (which makes it easy e.g. to initialise a GIG
from an msprime-simulated tree sequence 🎉). Nevertheless, there are a number of terminological
and implementation differences, an incomplete list of which are below.

(sec_usage_overview)=

## Overview

**Building a GIG.** Create an empty set of tables with `gigl.Tables()`, then add genomes with
`tables.nodes.add_row()` and inheritance with `tables.iedges.add_row()`, which gives the
inherited interval in both the child's and the parent's coordinates, so it can record
inversions, duplications and other rearrangements. Genetic variation is stored with
`tables.mutations.add_row()`, and individuals (e.g. diploids) with
`tables.individuals.add_row()`. Put the tables in order with `tables.sort()`, check them with
`tables.validate()`, then freeze them into a read-only `Graph` with `tables.graph()`. A GIG
with no structural variation can also be made from a _tskit_ tree sequence with
`Graph.from_tree_sequence()`, and converted back with `gig.to_tree_sequence()`. For tree
sequences with indels, see {ref}`sec_tools_indels`.

**Using tables while they grow.** Algorithms can run on the tables as well as on a frozen
graph, as long as the tables meet the validity requirements that the algorithm needs (see
{ref}`sec_validation`). Validation only checks the rows added since it was last called, so a
simulator can keep adding generations and keep using the tables, without freezing them each
time.

**Reading sequences.** `gig.haplotype(u)` gives a genome's sequence,
`gig.allele_at(u, position)` a single allele, and `gig.sequence_length(u)` its length (which
can differ between genomes).

**Tracing through the graph.** `gig.iedges_for_child(u)` and `gig.iedges_for_parent(u)` give
the iedges above and below a genome, and `IEdge.transform_position()` (or a
{class}`~giglib.PositionMap`) maps a position between the child's and the parent's coordinates.

**Homology.** A key task when comparing two chromosomes is to find which of their regions are
homologous, e.g. to place crossovers. `find_mrca_regions(u, v)` finds the regions that `u` and
`v` inherit from their most recent common ancestors, and `random_match_pos()` picks a random
homologous pair of positions from them. In a GIG, sequences can be homologous at different
levels (reflecting the fact that e.g. genes within a gene family are homologous to some
degree). To find more distant homology than that defined by the most recent common ancestor,
see {ref}`sec_tools_homology`.

**Trimming.** `sample_resolve()` removes material that doesn't reach the samples, and
`decapitate()` removes everything older than a given time. (A full equivalent of _tskit_'s
`simplify()` doesn't exist yet: see {ref}`sec_design_simplification`.)

**Tools.** The experimental {ref}`giglib.tools <sec_tools>` package has extras built on these:
converting tree sequences with indels, finding and plotting homology, and some simple
examples to show how to build targeted simulators.

(sec_usage_consistency)=

## Conventions

Where there is a conflict, _giglib_ aims for internal consistency rather than
compatibility with _tskit_. In particular:

- **Attributes versus methods** Fixed properties of a GIG are attributes
  (e.g. `gig.nodes`, `gig.iedges`, `gig.time_units`); anything that takes
  parameters or performs a query is a method (e.g. `gig.filter_nodes(flags=..., time=...)`,
  `gig.sample_nodes(time=...)`, `gig.iedges_for_child(u)`).
- **Graphs return objects, tables return IDs** Methods on a `Graph` return objects such as
  `Node` or `IEdge` instances. Collections of objects are returned as sequence-like views
  which support `len()`, indexing and iteration, and whose (read-only) IDs are available via `.ids`,
  e.g. `len(gig.sample_nodes())` or `gig.sample_nodes(time=0).ids`. Methods with the same name
  on a `Tables` object return numpy arrays of IDs instead, e.g. `tables.sample_nodes()`.
- **Same names on both layers** As far as possible, the same query has the same name on `Graph`
  and `Tables` objects, so the object you call it on determines whether you get objects or IDs.
  Queries on a single table are named after what they return, e.g. `gig.iedges_for_child(u)`
  gives `IEdge` objects, and `tables.iedges.ids_for_child(u)` gives iedge IDs.
- **Lenient inputs** Methods on a `Graph` that take a node accept either a node ID or a `Node` object.
- **Nodes first, iedge coordinates by keyword** Methods that add a row take the node(s) first, e.g.
  `tables.mutations.add_row(u, 100, "A")` or `tables.iedges.add_row(child, parent, ...)`. An
  iedge has four coordinates, which are easy to confuse, so they must be given by keyword, e.g.
  `tables.iedges.add_row(1, 0, child_left=0, child_right=100)`. The fields of an iedge row are in
  the same order, starting with `child` and `parent`. This also means that code written for
  _tskit_'s `edges.add_row(left, right, parent, child)` fails, rather than silently creating the
  wrong edges.
- **Positions and chromosomes** A position always comes before the chromosome it refers to, and
  the chromosome is always given by keyword, e.g. `gig.allele_at(u, 100, chromosome=1)`,
  `gig.haplotype(u, chromosome=1, left=0, right=50)` or
  `tables.mutations.add_row(u, 100, "A", chromosome=1)`. Where a method returns a single value or
  sequence for a node (e.g. `allele_at`, `haplotype` or `sequence_length`), `chromosome` defaults
  to `0`. Where a method returns a collection, or a value that can be combined over chromosomes
  (e.g. `iedges_for_child`, `mutations_for_node`, `ids_for_child` or `max_pos`), it defaults to
  `None`, meaning all chromosomes. Named tuples follow the same order, e.g.
  {class}`~giglib.Position` `(position, chromosome)`, {class}`~giglib.Segment`
  `(left, right, chromosome)`, `MutationTableRow(node, position, chromosome, ...)`, and iedges,
  whose `child_chromosome` and `parent_chromosome` fields follow the coordinates. Since the field
  names match the argument names, a `Position` or `Segment` can be passed using `**`, e.g.
  `gig.allele_at(u, **position)` or `gig.haplotype(u, **segment)`. An orientation
  (e.g. whether one genome is inverted relative to another) belongs to a pair of positions, so is
  stored alongside them, rather than in a `Position`.
- **tskit-style names** Some _tskit_ names that do not exist in _giglib_ (e.g. `gig.num_nodes`,
  `gig.samples()`, `gig.node(u)`, `gig.edges()` and `tables.edges`) are provided for convenience,
  but emit a `giglib.TskitNameWarning` suggesting the _giglib_ equivalent (e.g. `len(gig.nodes)`).
  These warnings can be silenced in the usual way, e.g.
  `warnings.simplefilter("ignore", giglib.TskitNameWarning)`.

(sec_usage_tskit_differences)=

## Differences from _tskit_

- **Iedges** Intervals in a GIG as defined by giglib are stored
  in the iedges table, which has a `parent_left` and `child_left` column rather than a simple `left`
  column as in _tskit_ (and similarly for `right`). We use the term `iedge` rather than simply `edge`
  to indicate that this is a denormalised table that contains an interval (`i`) as well as the
  parent and child coordinates that represent an `edge` in the graph. That means a single graph edge
  can be represented by multiple disjoint intervals in the `iedges` table. If the parent
  coordinates are the same as the child's (i.e. the interval is not moved or inverted), they can
  be omitted, and `child_left` defaults to 0, so e.g. `tables.iedges.add_row(1, 0, child_right=100)`
  means that node 1 inherits positions 0 to 100 from node 0, unchanged. If no coordinates are given,
  the iedge covers position 0 only (like the default sequence length of 1 in _msprime_), which is
  handy for building simple trees, e.g. `tables.iedges.add_row(1, 0)`.
- **Tables and Graphs** `Tables` and `Graph` correspond to _tskit_'s `TableCollection` and
  `TreeSequence` (see {ref}`sec_usage_overview`).
- **Chromosomes** The API includes the possibility of having genetic material on different chromosomes.
  This is implemented using two extra columns in the iedges table (see
  https://github.com/hyanwong/giglib/issues/11 for the rationale)
- **Object access** Information stored in GIG tables can be accessed using square brackets, and
  the `len()` function should work, so the canonical usage looks like `gig.nodes[0]`, `len(gig.nodes)`,
  and `[u.id for u in gig.nodes]` rather than the equivalents in _tskit_ (`ts.node(0)`, `ts.num_nodes`,
  and `[u.id for u in ts.nodes()]`.
- **Internal iedge order** The iedges in a GIG are sorted such that all those for a given child are
  adjacent (rather than all edges for a parent being adjacent as in _tskit_). Moreover, iedges are
  sorted in decreasing order of child time, so that edges with more recent children come last. Since
  edges for a given child are adjacent, child intervals for those edges cannot overlap. Edges for a
  given child are ordered by left coordinate, which helps algorithmic efficiency. The internal
  `gig.iedges_for_parent(u)` method returns iedges for a parent in the order expected by tskit.
- **No `.trees()` iterator** See above: the meaning of a "local tree" is unclear in a GIG, so
  implementing the equivalent of the fundamental _tskit_ `.trees()` method is likely to require
  substantial theoretical work.
- **Algorithms on tables** Unlike _tskit_, some fundamental algorithms such as `find_mrca_regions` and
  `sample_resolve` (a basic type of `simplify`) can be run directly on a set of tables.
  For this to work, the tables need to conform to certain validity criteria, recorded as separate
  validity flags, which can be checked as rows are added, or in bulk with `tables.validate()`.
  This allows use of the GIG structure during generation of the tables, without having the substantial
  overhead of continually having to freeze them into an immutable graph. This makes forward simulation
  feasible. See {ref}`sec_validation`.
- **Mutations, but no sites** Each mutation records its position in the coordinates of the node it
  lies above, so there is no sites table, and ancestral states are given by mutations above roots
  (see {ref}`sec_usage_mutations` below).

## Examples

See also {ref}`sec_examples` for longer, runnable examples, such as a simple forward simulation.

As with a _tskit_ tree sequence, a GIG is immutable, so to create one from scratch you
need to make a set of `Tables` and freeze them using the `.graph()` method (which validates
the tables, and builds indexes for fast lookup):

```python
import giglib as gigl

tables = gigl.Tables()
tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
tables.nodes.add_row(1, flags=0)
# Add an inversion
tables.iedges.add_row(0, 1, child_left=0, child_right=1, parent_left=1, parent_right=0)
gig = tables.graph()
assert len(gig.nodes) == 2
assert len(gig.iedges) == 1
assert len(gig.sample_nodes()) == 1
assert list(gig.sample_nodes().ids) == [0]
assert gig.iedges[0].is_inversion()
```

(sec_usage_mutations)=
### Mutations

Mutations are converted when creating a GIG from a tree sequence. Since there is no sites table,
each site's ancestral state becomes an extra mutation (with unknown time) above each root of the
local tree at that site. The alleles of a node are decoded by tracing each position up through
the iedges to the most recent mutation above it, so the haplotypes of the samples are the same
as those from _tskit_:

```python
import msprime
import giglib as gigl

ts = msprime.sim_ancestry(3, sequence_length=50, recombination_rate=0.01, random_seed=2)
ts = msprime.sim_mutations(ts, rate=0.02, random_seed=2)
gig = gigl.Graph.from_tree_sequence(ts)
print(ts.num_sites, "sites and", ts.num_mutations, "mutations in the tree sequence")
print(len(gig.mutations), "mutations in the GIG")  # also has one ancestral state per root per site
for u, alignment in zip(ts.samples(), ts.alignments()):
    assert gig.haplotype(u) == alignment
# 4 sites and 4 mutations in the tree sequence
# 8 mutations in the GIG
```

Positions with no mutations above them have no known allele, so are shown as missing data (`N`).
This matches `ts.alignments()` without a reference sequence (giglib does not yet support
reference sequences).

The coordinates of a mutation are those of the node it lies above, so rearrangements are
handled automatically. Here a child carries a duplicated copy of the first 5 bases of its
parent's genome, so a mutation above the parent is inherited twice, while a mutation above the
child affects only one copy:

```python
tables = gigl.Tables()
root = tables.nodes.add_row(time=2)
parent = tables.nodes.add_row(time=1)
child = tables.nodes.add_row(time=0, flags=gigl.NODE_IS_SAMPLE)
tables.iedges.add_row(parent, root, child_right=10)
tables.iedges.add_row(child, parent, child_right=10)
# a duplicated copy of parent positions 0-5
tables.iedges.add_row(child, parent, child_left=10, child_right=15, parent_left=0, parent_right=5)
for position, base in enumerate("ACGTACGTAC"):
    tables.mutations.add_row(root, position, base)  # ancestral states (with unknown times)
tables.mutations.add_row(parent, 2, "T", time=1.5)
tables.mutations.add_row(child, 13, "G", time=0.5)
tables.sort()
gig = tables.graph()

print(gig.haplotype(root))  # ACGTACGTAC
print(gig.haplotype(parent))  # ACTTACGTAC
print(gig.haplotype(child))  # ACTTACGTACACTGA
print(gig.allele_at(child, 12))  # T
```

The examples above all use chromosome 0, the default. Other chromosomes are given by keyword, e.g.
`tables.mutations.add_row(u, 3, "A", chromosome=1)` or `gig.allele_at(u, 3, chromosome=1)`
(see {ref}`sec_usage_consistency`).

As elsewhere in the API, the `Graph` returns objects, and the tables return IDs:
`gig.mutations_for_node(child)` gives `Mutation` objects, while
`gig.tables.mutations.ids_for_node(child)` gives an array of mutation IDs. Mutation times may be
{data}`~giglib.UNKNOWN_TIME` (use {func}`~giglib.is_unknown_time` or `mutation.is_time_known()`
to check). Where several mutations above a node are at the same position, the order in the table
decides which is the most recent.

Some details to be aware of:

- **Inversions** Alleles inherited through an inversion are read from the other strand, so are
  complemented when decoded (e.g. `A` becomes `T`). Alleles that can't be complemented, such as
  `0` and `1`, are left unchanged. The complements used are in `tables.allele_complements`.
- **Multi-base alleles** A mutation is at a single position, but its derived state can be several
  bases long (e.g. a small insertion). This is allowed, with fewer guarantees, but it is better to
  represent indels using iedges with different parent and child coordinates.
- **Sequence length** A node's length comes from the iedges above and below it, and the mutations
  above it (which give the bases of inserted material, as it has no parent). So a node with no
  iedges or mutations has length 0, and missing data at the end of a genome is not counted: pass
  `right=` to `gig.haplotype()` to decode it.
- **Algorithms** `sample_resolve()` and `remove_non_ancestral_nodes()` keep the mutations needed by
  the remaining nodes. By default, `decapitate()` adds mutations (with unknown times) where needed,
  so that the haplotypes of the remaining nodes are unchanged. With `discard_old_mutations=True`, old
  mutations are simply removed, as in _tskit_, so alleles may become missing:

  ```python
  print(gig.decapitate(1.5).haplotype(1))  # the child is now node 1: ACTTACGTACACTGA
  print(gig.decapitate(1.5, discard_old_mutations=True).haplotype(1))  # NNNNNNNNNNNNNGN
  ```

- **Converting to _tskit_** `gig.to_tree_sequence()` only works for GIGs without rearrangements,
  on a single chromosome, where all roots at each site agree on the ancestral state. It is mainly
  intended for testing.

For the full design, and the reasons behind it, see {ref}`sec_design_mutations`.

(sec_usage_simulations)=
### Simulations

Some simple forward-time simulators are provided in the experimental
{ref}`giglib.tools.simulation <sec_tools_simulation>` module. They are based on a relatively simple extension
of the boilerplate forward-simulation code in the _tskit_ [forward-time simulation tutorial](https://tskit.dev/tutorials/forward_sims.html). The important
difference is that the {meth}`~giglib.Tables.find_mrca_regions` method is used to locate breakpoints
for recombination. Currently only forward simulation code is provided. Nevertheless, it is conceptually
general enough to form the basis of a pangenome simulator (this would however require
substantial effort in fixing reasonable parameters before realistic-looking pangenomes
could be created).

For examples of using the simulation code directly, see
[this comment](https://github.com/hyanwong/giglib/issues/86#issuecomment-1970029273)
or [this one](https://github.com/hyanwong/giglib/issues/82#issuecomment-1972153550) (written
when the simulators were in the test suite, so import them from the modules of
`giglib.tools.simulation` instead, e.g. `giglib.tools.simulation.one_break_inversions`).
