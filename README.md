# The Genetic Inheritance Graph Library (giglib)

A library for "(Generalised) Genetic Inheritance Graphs" (GIGs), which extend the idea of an
[ancestral recombination graph](https://tskit.dev/tutorials/args.html) to capture genetic inheritance
with genomic rearrangements, and hence describe inherited structural variation.

This is a proof of concept, not stable software: the API may change at any time.

📖 **Documentation: [https://hyanwong.github.io/giglib/](https://hyanwong.github.io/giglib/)**

## Quick start

```bash
python -m pip install git+https://github.com/hyanwong/giglib
```

You can try it out by converting a [succinct tree sequence](https://tskit.dev/tutorials/what_is.html)
into a GIG (although such a GIG will not contain structural variation):

```python
import msprime
import giglib as gigl  # 😁

ts = msprime.sim_ancestry(4, sequence_length=100, recombination_rate=0.01, random_seed=1)
gig = gigl.Graph.from_tree_sequence(ts)
print(len(gig.nodes), "nodes, of which", len(gig.sample_nodes()), "are samples")
```

## Basic idea

In [_tskit_](https://tskit.dev), each edge records the left and right coordinates of the inherited
DNA. A GIG records these separately for the *child* and the *parent*, each in its own coordinate
system. Rows of this kind are called interval edges, or `iedges`. This lets a GIG represent inversions,
duplications, deletions, and more complex rearrangements: see
[Concepts](https://hyanwong.github.io/giglib/concepts.html) for examples.

<img src="https://github.com/hyanwong/GeneticInheritanceGraph/assets/5416474/0fff67b3-71e7-4ed5-895a-140a06f49940" alt="A schematic GIG" width="500"/>

(Schematic drawn by [@duncanMR](https://github.com/duncanMR)).

## API principles

The design is inspired by _tskit_, but where the two conflict, giglib aims for internal
consistency rather than compatibility with _tskit_:

- **Tables and graphs** A GIG is created by freezing a mutable `Tables` object into an immutable
  `Graph`, using `gig = tables.graph()`. Some algorithms can be run directly on the tables, which
  makes forward simulation feasible.
- **Graphs return objects, tables return IDs** For example, `gig.sample_nodes()` returns `Node`
  objects, while `tables.sample_nodes()` returns a numpy array of IDs. Where possible, a query
  has the same name on both.
- **Views** Collections such as `gig.nodes` or `gig.iedges_for_child(u)` support `len()`,
  indexing, and iteration, and expose their IDs via `.ids`.
- **Attributes versus methods** Fixed properties are attributes (`gig.nodes`, `gig.time_units`).
  Anything that takes parameters is a method (`gig.filter_nodes(flags=..., time=...)`).
- **tskit names** Common _tskit_ names such as `gig.num_nodes` still work, but they emit a warning
  suggesting the giglib equivalent (here, `len(gig.nodes)`).

See [Usage](https://hyanwong.github.io/giglib/usage.html) for more detail, including the full
list of differences from _tskit_ and pointers to the example simulation code.

## Contributing

Ideas and discussion happen in the [issues](https://github.com/hyanwong/giglib/issues).
Pull requests are welcome.
