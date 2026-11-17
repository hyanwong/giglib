# The Genetic Inheritance Graph Library (giglib)

**Genetic ancestry for genomes that rearrange.**

Two kinds of graph are increasingly used to describe genetic variation. [Ancestral recombination graphs](https://tskit.dev/tutorials/args.html) (ARGs) record
how genomes are related through inheritance and recombination, but assume that all genomes line
up base for base. Pangenome graphs capture structural variation between genomes, such as
inversions, duplications, and deletions, but not the genealogy that produced it. A **genetic
inheritance graph** (GIG) combines the two: it records which pieces of DNA each genome
inherited from which ancestor, even when that DNA has been rearranged along the way.

**If you use [tskit](https://tskit.dev), think of giglib as tskit for genomes that rearrange.** A tree
sequence assumes that every genome shares one coordinate system, so position 1000 means the same
thing in every sample. That breaks down as soon as there are inversions, duplications, deletions,
or transposable elements. A GIG drops that assumption: each edge records the inherited interval
in the *child's* coordinates **and** in the *parent's* coordinates. If you know tskit, you already
know most of giglib.

<img src="https://github.com/hyanwong/GeneticInheritanceGraph/assets/5416474/0fff67b3-71e7-4ed5-895a-140a06f49940" alt="A genetic inheritance graph with a duplication, a deletion and an inversion" width="500"/>

*A small GIG with a duplication, a deletion and an inversion. Bands show how each inherited
interval maps from parent to child coordinates. Drawn by [@duncanMR](https://github.com/duncanMR).*

📖 **[Documentation](https://hyanwong.github.io/giglib/)** · This is a proof of concept: the API may change at any time.

## Try it

```bash
python -m pip install git+https://github.com/hyanwong/giglib
```

Start from a tree sequence you already have:

```python
import msprime
import giglib as gigl  # 😁

ts = msprime.sim_ancestry(4, sequence_length=100, recombination_rate=0.01, random_seed=1)
gig = gigl.Graph.from_tree_sequence(ts)
print(len(gig.nodes), "nodes,", len(gig.iedges), "iedges,", len(gig.sample_nodes()), "samples")

ts_again = gig.to_tree_sequence()  # ...and back again
```

Then go somewhere a tree sequence can't follow. Here a child inherits its parent's genome, plus an
extra, inverted copy of the last 50 bases:

```python
tables = gigl.Tables()
parent = tables.nodes.add_row(time=1)
child = tables.nodes.add_row(time=0, flags=gigl.NODE_IS_SAMPLE)
tables.iedges.add_row(child=child, parent=parent, child_left=0, child_right=100, parent_left=0, parent_right=100)
tables.iedges.add_row(child=child, parent=parent, child_left=100, child_right=150, parent_left=100, parent_right=50)
gig = tables.graph()

for ie in gig.iedges_for_child(child):
    print(f"child {ie.child_left}-{ie.child_right} <- parent {ie.parent_left}-{ie.parent_right}",
          "(inverted)" if ie.is_inversion() else "")
# child 0-100 <- parent 0-100
# child 100-150 <- parent 100-50 (inverted)
```

The child's genome is now 150 bases long, while its parent's is 100. giglib can still trace
ancestry through the rearrangement. For example, `gig.find_mrca_regions(u, v)` finds which
regions of two genomes share a common ancestor, and where those regions sit in each genome. That
is enough to build forward simulations with structural variants and recombination: see the
[Usage](https://hyanwong.github.io/giglib/usage.html#simulations) page.

## Design principles

- **No global coordinate system.** Each genome has its own coordinates, and positions are integers
  (base pairs). There is no global sequence length: a node's length comes from the iedges above and
  below it (`gig.sequence_length(u, chromosome)`).
- **One parent per position.** Within a child's chromosome, each position is inherited from at
  most one parent. Iedges are grouped by child, so tracing the ancestry of any point upwards
  through the graph is always unambiguous, however rearranged the genomes are.
- **Multiple chromosomes.** A node can carry several chromosomes, and an iedge can move material
  between them (e.g. translocations).
- **Validate only what you need.** Validity is tracked as a set of separate guarantees
  ([`ValidFlags`](https://hyanwong.github.io/giglib/python-api.html#sec-python-api-tables-validity)),
  such as "parents are older than children" or "iedges for each child are sorted". Rows can be
  checked as they are added, e.g. with `tables.add_iedge_row(..., validate=...)`. Each table
  algorithm checks for just the guarantees it needs, and refuses to run without them. So algorithms
  like `tables.find_mrca_regions()` can run on growing tables during a simulation, without the cost
  of repeatedly freezing them into a `Graph`. Freezing with `tables.graph()` requires all the
  guarantees, and adds indexes for fast lookup.
- **A library, not a simulator.** As with tskit, giglib provides the data structures and
  algorithms that simulators and inference methods can build on.

## Coming from tskit

| tskit | giglib |
|---|---|
| `TableCollection`, `tables.tree_sequence()` | `Tables`, `tables.graph()` |
| `ts.edges()` (`left`, `right`) | `gig.iedges` (`child_left`, `child_right`, `parent_left`, `parent_right`) |
| `ts.node(u)`, `ts.num_nodes` | `gig.nodes[u]`, `len(gig.nodes)` |
| `ts.samples()` | `gig.sample_nodes()` (`Node` objects) or `gig.sample_nodes().ids` (IDs) |
| `ts.tables.nodes.time` | `gig.tables.nodes.time` |
| `ts.simplify()` | `gig.sample_resolve()` (a basic version, so far) |

Where tskit and internal consistency conflict, giglib chooses consistency. A few rules keep the
API predictable:
- `Graph` methods return objects and `Tables` methods return ID arrays, using the same names.
- Collections such as `gig.nodes` support `len()`, indexing and `.ids`.
- Fixed properties are attributes (`gig.nodes`); anything that takes parameters is a method
  (`gig.sample_nodes(time=0)`).

If you type `gig.num_nodes` out of habit, it still works, and a warning tells you the giglib
equivalent.

### What you won't find (yet)

- **No local trees.** Without a shared coordinate system, there is no single "tree at position
  x" across all samples. A duplication also means one sample genome can appear as two tips in the
  same local genealogy, breaking the one-sample-one-tip assumption behind ARGs. So there is no
  `ts.trees()` equivalent: algorithms instead work directly on the graph, by tracing intervals up
  through the iedges.
- **No sites or mutations.** In tskit a site is a position in the shared coordinate system, so a
  global sites table doesn't make sense for a GIG. The plan is to store each mutation's position
  in the coordinates of the node it occurs above, without a separate sites table (see
  [#5](https://github.com/hyanwong/giglib/issues/5)). Until then, tree sequences must be converted
  without their sites and mutations.

For the full story, see [Concepts](https://hyanwong.github.io/giglib/concepts.html) (how
rearrangements are encoded) and [Usage](https://hyanwong.github.io/giglib/usage.html) (all the
differences from tskit).

## Get involved

giglib is an experiment in what an ARG library for structural variation could look like, and
there are plenty of open questions, from how to store mutations to how to define local trees. Ideas,
use cases and pull requests are very welcome: see the [issues](https://github.com/hyanwong/giglib/issues).
