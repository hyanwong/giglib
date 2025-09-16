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
and implementation differences, an incomplete list of which are below:

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
  and `[u.id for u in ts.nodes()]`. Similarly, we use `gig.sample_ids` (no braces) rather than `ts.samples()`.
- **Internal iedge order** The iedges in a GIG are sorted such that all those for a given child are
  adjacent (rather than all edges for a parent being adjacent as in _tskit_). Moreover, iedges are
  sorted in decreasing order of child time, so that edges with more recent children come last. Since
  edges for a given child are adjacent, child intervals for those edges cannot overlap. Edges for a
  given child are ordered by left coordinate, which helps algorithmic efficiency. The internal
  `gig.iedge_map_sorted_by_parent` array indexes into iedges in the order expected by tskit.
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
assert len(gig.samples) == 1
assert gig.iedges[0].is_inversion()
```

### Simulations

Examples of code that runs simulations are provided in the test suite. Currently they are based on a relatively simple extension
of the boilerplate forward-simulation code in https://tskit.dev/tutorials/forward_sims.html. The important
difference is that the [find_mrca_regions](https://github.com/hyanwong/giglib/blob/3385f5149f7028cb2b5bfd8c236774b926f79de9/giglib/tables.py#L841)
method is used to locate breakpoints for recombination.

For more details the [sim.py file](https://github.com/hyanwong/giglib/blob/main/tests/sim.py)
is a convenience wrapper that links out to other files containing simulation code.
Currently only forward simulation code is provided. Nevertheless, it is conceptually
general enough to form the basis of a pangenome simulator (this would however require
substantial effort in fixing reasonable parameters before realistic-looking pangenomes
could be created).
