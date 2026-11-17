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

(sec_usage_consistency)=

## API consistency

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
- **Lenient inputs** Methods on a `Graph` that take a node accept either a node ID or a `Node` object.
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
  `gig.samples()`, `gig.node(u)`) are provided for convenience, but emit a `giglib.TskitNameWarning`
  suggesting the _giglib_ equivalent (e.g. `len(gig.nodes)`). These warnings can be silenced in the
  usual way, e.g. `warnings.simplefilter("ignore", giglib.TskitNameWarning)`.

## API differences

- **Iedges** Intervals in a GIG as defined by giglib are stored
  in the iedges table, which has a `parent_left` and `child_left` column rather than a simple `left`
  column as in _tskit_ (and similarly for `right`). We use the term `iedge` rather than simply `edge`
  to indicate that this is a denormalised table that contains an interval (`i`) as well as the
  parent and child coordinates that represent an `edge` in the graph. That means a single graph edge
  can be represented by multiple disjoint intervals in the `iedges` table.
- **Tables and Graphs** giglib provides a `Tables` and `Graph` class, corresponding
  to `TableCollection` and `TreeSequence` classes in _tskit_. Thus to create a GIG from scratch,
  you do `gig = tables.graph()`
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
  For this to work, the tables need to conform to certain validity criteria.
  Substantial additional functionality has been incoporated into the tables objects, so that table rows
  (especially for iedges) can be validated on calling `table.add_row()`, according to certain validity flags
  (see the `ValidFlags` class in `constants.py`).
  This allows use of the GIG structure during generation of the tables, without having the substantial
  overhead of continually having to freeze them into an immutable graph. This makes forward simulation
  feasible.
- **Mutations, but no sites** Each mutation records its position in the coordinates of the node it
  lies above, so there is no sites table, and ancestral states are given by mutations above roots
  (see {ref}`sec_usage_mutations` below).
- **Other stuff** More differences should be noted here!

## Examples

As with a _tskit_ tree sequence, a GIG is immmutable, so to create one from scratch you
need to make a set of `Tables` and freeze them using the `.graph()` method (which gives
opportunity to cache and index important stuff):

```python
import giglib as gigl

tables = gigl.Tables()
tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
tables.nodes.add_row(1, flags=0)
# Add an inversion
tables.iedges.add_row(
    parent=1, child=0, child_left=0, child_right=1, parent_left=1, parent_right=0
)
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
tables.add_iedge_row(0, 10, 0, 10, child=parent, parent=root)
tables.add_iedge_row(0, 10, 0, 10, child=child, parent=parent)
tables.add_iedge_row(10, 15, 0, 5, child=child, parent=parent)  # a duplicated copy of 0-5
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
- **Sequence length** A node's length comes from its iedges, so a node with no iedges has length 0.
  Pass `right=` to `gig.haplotype()` to decode such a node.
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

Examples of code that runs simulations are provided in the test suite. Currently they are based on a relatively simple extension
of the boilerplate forward-simulation code in the _tskit_ [forward-time simulation tutorial](https://tskit.dev/tutorials/forward_sims.html). The important
difference is that the {meth}`~giglib.Tables.find_mrca_regions` method is used to locate breakpoints
for recombination.

For more details the [sim.py file](https://github.com/hyanwong/giglib/blob/main/tests/sim.py)
is a convenience wrapper that links out to other files containing simulation code.
Currently only forward simulation code is provided. Nevertheless, it is conceptually
general enough to form the basis of a pangenome simulator (this would however require
substantial effort in fixing reasonable parameters before realistic-looking pangenomes
could be created).

For a hacky example of how to use the simulation code directly from the test suite, see
[this comment](https://github.com/hyanwong/giglib/issues/86#issuecomment-1970029273)
or [this one](https://github.com/hyanwong/giglib/issues/82#issuecomment-1972153550).
