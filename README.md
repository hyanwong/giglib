# GeneticInheritanceGraphLibrary

A basic repo for kicking around ideas for the "(General) Genetic Inheritance Graph" structure, which should be able to
capture genetic inheritance with genomic rearrangements, and hence describe inherited structural variation. This is
not meant to be stable software, and the API is subject to change at any time, but for the moment you can
try it out by converting a tree sequence to a GIG (although such a gig will not contain structural variations)

```python
import msprime
import GeneticInheritanceGraphLibrary as gigl

ts = msprime.sim_ancestry(4, sequence_length=100, recombination_rate=0.01, random_seed=1)
gig = gigl.from_tree_sequence(ts)
print(len(gig.nodes), "nodes in this GIG")
```

## Basic idea

In [`tskit`](tskit.dev) we use edge annotations to describe which pieces of DNA are inherited in terms of a left and right coordinate.
It should be possible to extend this to track the L & R in the edge *child*, and the L & R in the edge *parent* separately.
The left and right values in each case refer to the coordinate system of the child and parent respectively.

I'm calling an extended tree-sequence-like structure such as this, a GIG ("General (or Genetic) Inheritance Graph"). For
terminological clarity,
we switch to using the term interval-edge (`iedge`) to refer to what is normally called an `edge` in a *tskit* Tree Sequence.

### Inversions

The easiest example is an inversion. This would be an iedge like

```
{parent: P, child: C, child_left: 100, child_right: 200, parent_left: 200, parent_right: 100}
```

A tandem duplication is represented by two iedges, one for each duplicated region:

```
{parent: P, child: C, child_left: 100, child_right: 200, parent_left: 100, parent_right: 200}
{parent: P, child: C, child_left: 200, child_right: 300, parent_left: 100, parent_right: 200}
```

Or a non-adjacent duplication:
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

This brings with it a load of extra complexities, and it’s unclear if the efficiency of the tskit approach,
with its edge indexing etc, will port in any meaningful way to this new structure.

## Meaning of trees
The interpretation of the trees may be rather different from that in tskit. In particular,
if there is a duplication, it could be interpreted as creating two tree tips within a single sample.
This would mean that there is no longer a 1-to-1 mapping from samples to tree tips, which
breaks one of the fundamental tskit ideas.

## API differences

The API of this GeneticInheritanceGraphLibrary intentionally mirrors that of _tskit_, apart from the
incomplete list of differences below:

- **Iedges** The main difference is that intervals in the GeneticInheritanceGraphLibrary are stored
  in the iedges table, which has a `parent_left` and `child_left` column rather than a simple `left`
  column as in *tskit* (and similarly for `right`).
- **Tables and Graphs** The GeneticInheritanceGraphLibrary has a `Tables` and `Graph` class, corresponding
  to `TableCollection` and `TreeSequence` classes in _tskit_. Thus to create a GIG from scratch,
  you do `gig = tables.graph()`
- **Object access** Information stored in GIG tables can be accessed using square brackets, and
  the `len()` function should work, so the canonical usage looks like `gig.nodes[0]`, `len(gig.nodes)`,
  and `[u.id for u in gig.nodes]` rather than the equivalents in _tskit_ (`ts.node(0)`, `ts.num_nodes`,
  and `[u.id for u in ts.nodes()]`. Similarly, we use `gig.samples` (no braces) rather than `ts.samples()`.
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
tables.iedges.add_row(parent=1, child=0, child_left=0, child_right=1, parent_left=1, parent_right=0)
gig = tables.graph()
assert len(gig.nodes) == 2
assert len(gig.iedges) == 1
assert len(gig.samples) == 1
```
