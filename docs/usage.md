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
- **Nodes by keyword** Where a row refers to more than one node, the nodes are given by keyword,
  e.g. `tables.iedges.add_row(0, 100, child=1, parent=0)`, so they cannot be swapped by mistake.
  (_tskit_'s `edges.add_row(left, right, parent, child)` is easy to get the wrong way round.)
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
  can be represented by multiple disjoint intervals in the `iedges` table. If the parent
  coordinates are the same as the child's (i.e. the interval is not moved or inverted), they can
  be omitted, e.g. `tables.iedges.add_row(0, 100, child=1, parent=0)`.
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
