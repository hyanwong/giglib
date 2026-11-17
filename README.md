# The Genetic Inheritance Graph Library (giglib)

**Like [tskit](https://tskit.dev), but for genomes that rearrange.**

A tree sequence assumes that every genome shares one coordinate system, so position 1000 means
the same thing in every sample. That breaks down as soon as there are inversions, duplications,
deletions, or transposable elements. A **genetic inheritance graph** (GIG) drops that assumption:
each edge records the inherited interval in the *child's* coordinates **and** in the *parent's*
coordinates. This lets a GIG store genetic ancestry that includes structural variation.

If you know tskit, you already know most of giglib.

📖 **[Documentation](https://hyanwong.github.io/giglib/)** · This is a proof of concept: the API may change at any time.

## Try it

```bash
python -m pip install git+https://github.com/hyanwong/giglib
```

Start from a tree sequence you already have:

```python
import msprime
import giglib as gigl

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

## Coming from tskit

| tskit | giglib |
|---|---|
| `TableCollection`, `tables.tree_sequence()` | `Tables`, `tables.graph()` |
| `ts.edges()` (`left`, `right`) | `gig.iedges` (`child_left`, `child_right`, `parent_left`, `parent_right`) |
| `ts.node(u)`, `ts.num_nodes` | `gig.nodes[u]`, `len(gig.nodes)` |
| `ts.samples()` | `gig.sample_nodes()` (`Node` objects) or `gig.sample_nodes().ids` (IDs) |
| `ts.tables.nodes.time` | `gig.tables.nodes.time` |
| `ts.simplify()` | `gig.sample_resolve()` (a basic version, so far) |

A few rules keep the API predictable:
- `Graph` methods return objects and `Tables` methods return ID arrays, using the same names.
- Collections such as `gig.nodes` support `len()`, indexing and `.ids`.
- Some algorithms run directly on unfrozen `Tables`, which makes forward simulation practical.

If you type `gig.num_nodes` out of habit, it still works, and a warning tells you the giglib
equivalent.

For the full story, see [Concepts](https://hyanwong.github.io/giglib/concepts.html) (how
rearrangements are encoded) and [Usage](https://hyanwong.github.io/giglib/usage.html) (all the
differences from tskit).

## Get involved

giglib is an experiment in what an ARG library for structural variation could look like, and
there are plenty of open questions, from how to store mutations to how to define local trees. Ideas,
use cases and pull requests are very welcome: see the [issues](https://github.com/hyanwong/giglib/issues).
