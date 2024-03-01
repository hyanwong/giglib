# GeneticInheritanceGraphLibrary

A basic repo for kicking around ideas for the "(Generalised) Genetic Inheritance Graph" structure, which should be able to
capture genetic inheritance with genomic rearrangements, and hence describe inherited structural variation. This is
not meant to be stable software, and the API is subject to change at any time, but for the moment you can
try it out by converting a [succinct tree sequence](https://tskit.dev/tutorials/what_is.html)
to a GIG (although such a gig will not contain structural variation)

```python
import msprime
import GeneticInheritanceGraphLibrary as gigl  # 😁

ts = msprime.sim_ancestry(
    4, sequence_length=100, recombination_rate=0.01, random_seed=1
)
gig = gigl.from_tree_sequence(ts)
print(len(gig.nodes), "nodes in this GIG")
```

## Basic idea

In [`tskit`](https://tskit.dev) we use edge annotations to describe which pieces of DNA are inherited in terms of a left and right coordinate.
It should be possible to extend this to track the L & R in the edge *child*, and the L & R in the edge *parent* separately.
The left and right values in each case refer to the coordinate system of the child and parent respectively.

I'm calling an extended tree-sequence-like structure such as this, a GIG ("Generalised (or Genetic) Inheritance Graph"). For
terminological clarity,
we switch to using the term interval-edge (`iedge`) to refer to what is normally called an `edge` in a *tskit* Tree Sequence.

Note that separating child from parent coordinates brings a host of extra complexities,
and it’s unclear if the efficiency of the tskit approach,
with its edge indexing etc, will port in any meaningful way to this new structure.
Nevertheless, basic operations such as simplification and finding MRCA genomes
have already been (inefficently) implemented.

### Inversions

The easiest example is an inversion. This would be an iedge like

```
{parent: P, child: C, child_left: 100, child_right: 200, parent_left: 200, parent_right: 100}
```

There is a subtle gotcha here, because intervals in a GIG, as in _tskit_, are treated as half-closed
(i.e. do not include the position given by the right coordinate). When we invert an interval, it
therefore does not include the *left* parent coordinate, but does include the *right* parent coordinate.
Any transformed position is thus out by one. Or to put it another way, an inversion specified
by child_left=0, child_right=3, parent_left=3, parent_right=0 transforms the points
0, 1, 2 to 2, 1, 0: although the *interval* 0, 3 is transformed to 0, 3., the *point* 0 is transformed
to position 2, not position 3. See
[here](https://github.com/hyanwong/GeneticInheritanceGraphLibrary/issues/41#issuecomment-1858530867)
for more discussion.

### Duplications

A tandem duplication is represented by two iedges, one for each duplicated region:

```
{parent: P, child: C, child_left: 100, child_right: 200, parent_left: 100, parent_right: 200}
{parent: P, child: C, child_left: 200, child_right: 300, parent_left: 100, parent_right: 200}
```

Or one of the edges could represent a non-adjacent duplication (e.g. corresponding to a transposable element):
```
{parent: P, child: C, child_left: 250, child_right: 350, parent_left: 100, parent_right: 200}
```

### Deletions

A deletion simply occurs when no material from the parent is transmitted to any of its children (and the coordinate system is shrunk)

```
# Deletion of parental region from 200-300
{parent: P, child: C, child_left: 100, child_right: 200, parent_left: 100, parent_right: 200}
{parent: P, child: C, child_left: 200, child_right: 300, parent_left: 300, parent_right: 400}
```

### Viz

The [graphical interpretation](https://github.com/hyanwong/GeneticInheritanceGraphLibrary/issues/2#issuecomment-1684164074)
is something like this, elegantly drawn by [@duncanMR](https://github.com/duncanMR):

<img src="https://github.com/hyanwong/GeneticInheritanceGraph/assets/5416474/0fff67b3-71e7-4ed5-895a-140a06f49940" alt="GIG" width="500"/>

## Meaning of trees
The interpretation of local trees may be rather different from that in tskit. In particular,
if there is a duplication, it could be interpreted as creating two tree tips within a single sample.
This would mean that there is no longer a 1-to-1 mapping from samples to tree tips, which
breaks one of the fundamental principles behind ARGs in general and the _tskit_ representation in particular.

## API differences

The fundamental data structures and API provided by the GeneticInheritanceGraphLibrary as defined in
this GitHub repository intentionally mirror that of _tskit_ (which makes it easy e.g. to initialise a GIG
from an msprime-simulated tree sequence 🎉). Nevertheless, there are a number of terminological
and implementation differences, an incomplete list of which are below:

- **Iedges** As described above, intervals in a GIG as defined in the GeneticInheritanceGraphLibrary are stored
  in the iedges table, which has a `parent_left` and `child_left` column rather than a simple `left`
  column as in _tskit_ (and similarly for `right`).
- **Tables and Graphs** The GeneticInheritanceGraphLibrary has a `Tables` and `Graph` class, corresponding
  to `TableCollection` and `TreeSequence` classes in _tskit_. Thus to create a GIG from scratch,
  you do `gig = tables.graph()`
- **Object access** Information stored in GIG tables can be accessed using square brackets, and
  the `len()` function should work, so the canonical usage looks like `gig.nodes[0]`, `len(gig.nodes)`,
  and `[u.id for u in gig.nodes]` rather than the equivalents in _tskit_ (`ts.node(0)`, `ts.num_nodes`,
  and `[u.id for u in ts.nodes()]`. Similarly, we use `gig.samples` (no braces) rather than `ts.samples()`.
- **Internal iedge order** The iedges in a GIG are sorted such that all those for a given child are
  adjacent (rather than all edges for a parent being adjacent as in _tskit_). Moreover, iedges are
  sorted in decreasing order of child time, so that edges with more recent children come last. Since
  edges for a given child are adjacent, child intervals for those edges cannot overlap. Edges for a
  given child are ordered by left coordinate, which helps algorithmic efficiency. The internal
  `gig.iedge_map_sorted_by_parent` array indexes into iedges in the order expected by tskit.
- **No `.trees()` iterator** See above: the meaning of a "local tree" is unclear in a GIG, so
  implementing the equivalent of the fundamental _tskit_ `.trees()` method is likely to require
  substantial theoretical work.
- **Other stuff** More differences should be noted here!

## Examples

As with a _tskit_ tree sequence, a GIG is immmutable, so to create one from scratch you
need to make a set of `Tables` and freeze them using the `.graph()` method (which gives
opportunity to cache and index important stuff):

```python
import GeneticInheritanceGraphLibrary as gigl

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

Examples of code that runs simulations are provided in the test suite and are a relatively simple extension
of the boilerplate forward-simulation code in https://tskit.dev/tutorials/forward_sims.html. The
[sim.py file](https://github.com/hyanwong/GeneticInheritanceGraphLibrary/blob/main/tests/sim.py)
is a convenience wrapper that links out to other files containing simulation code.
Currently only forward simulation code is provided. Nevertheless, it is conceptually
general enough to form the basis of a pangenome simulator (this would however require
substantial effort in fixing reasonable parameters before realistic-looking pangenomes
could be created).

For a hacky example of how to use the simulation code directly from the test suite, see
https://github.com/hyanwong/GeneticInheritanceGraphLibrary/issues/86#issuecomment-1970029273
or https://github.com/hyanwong/GeneticInheritanceGraphLibrary/issues/82#issuecomment-1972153550.
