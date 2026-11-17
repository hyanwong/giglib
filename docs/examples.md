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

(sec_examples)=

# Examples

These examples are run when the documentation is built, so they always work with the current
version of _giglib_.

(sec_examples_wright_fisher)=

## A simple Wright-Fisher simulation

This is a minimal forward-time Wright-Fisher simulation, similar to the simplest example in the
_tskit_ [forward simulation tutorial](https://tskit.dev/tutorials/forward_sims.html). There are
`N` (haploid) genomes in each of `G` generations. Each genome in a new generation inherits from
a randomly chosen parent in the previous generation.

```{code-cell}
import numpy as np
import giglib as gigl


def sim_wright_fisher(N, G, random_seed=None):
    rng = np.random.default_rng(random_seed)
    tables = gigl.Tables()
    parent_ids = [tables.nodes.add_row(G, flags=gigl.NODE_IS_SAMPLE) for _ in range(N)]

    for g in reversed(range(G)):
        child_ids = [tables.nodes.add_row(g, flags=gigl.NODE_IS_SAMPLE) for _ in range(N)]
        for child in child_ids:
            tables.iedges.add_row(child, rng.choice(parent_ids))
        parent_ids = child_ids

    tables.sort()
    return tables.graph()


gig = sim_wright_fisher(N=4, G=8, random_seed=2)
print(len(gig.nodes), "nodes and", len(gig.iedges), "iedges")
```

Since no coordinates are given to `tables.iedges.add_row(child, parent)`, each iedge covers
position 0 only (as for the default sequence length of 1 in _msprime_). Every node is marked as a
sample, so that all of them are kept.

Each position in a genome has at most one parent, so we can trace the lineage of a genome back
through the generations by following the iedges above it:

```{code-cell}
u = gig.sample_nodes(time=0)[0]
lineage = [u.id]
while len(iedges := gig.iedges_for_child(lineage[-1])) > 0:
    lineage.append(iedges[0].parent)
print("Lineage of node", u.id, ":", lineage)
```

There are no rearrangements, so the GIG can be converted to a tree sequence. Here we simplify it
to the final generation, and draw the genealogy:

```{code-cell}
ts = gig.to_tree_sequence().simplify(gig.sample_nodes(time=0).ids)
print(ts.draw_text())
```

(sec_examples_recombination)=

## Adding recombination

To simulate a genome of length `L` with recombination, each child inherits the left part of its
genome from one parent, and the right part from another, using `child_left` and `child_right`.
As the parent coordinates are not given, they default to the child's, so these are ordinary
recombinations, without any structural variation.

```{code-cell}
def sim_wright_fisher_recombination(N, G, L, random_seed=None):
    rng = np.random.default_rng(random_seed)
    tables = gigl.Tables()
    parent_ids = [tables.nodes.add_row(G, flags=gigl.NODE_IS_SAMPLE) for _ in range(N)]

    for g in reversed(range(G)):
        child_ids = [tables.nodes.add_row(g, flags=gigl.NODE_IS_SAMPLE) for _ in range(N)]
        for child in child_ids:
            left_parent, right_parent = rng.choice(parent_ids, size=2, replace=False)
            breakpoint = rng.integers(1, L)
            tables.iedges.add_row(child, left_parent, child_right=breakpoint)
            tables.iedges.add_row(child, right_parent, child_left=breakpoint, child_right=L)
        parent_ids = child_ids

    tables.sort()
    return tables.graph()


gig = sim_wright_fisher_recombination(N=4, G=20, L=100, random_seed=3)
child = gig.sample_nodes(time=0)[0]
for ie in gig.iedges_for_child(child):
    print(f"Positions {ie.child_left}-{ie.child_right} of node {child.id} come from node {ie.parent}")
```

{meth}`~giglib.Graph.find_mrca_regions` finds the most recent common ancestors of two genomes, and
the regions that each one shares:

```{code-cell}
u, v = gig.sample_nodes(time=0).ids[:2]
mrcas = gig.find_mrca_regions(u, v)
for mrca, regions in sorted(mrcas.items()):
    intervals = ", ".join(f"{r.left}-{r.right}" for r in sorted(regions))
    print(f"Node {mrca} (time {gig.nodes[mrca].time:g}) is the MRCA over {intervals}")
```

Again, this can be converted to a tree sequence, which now has several local trees:

```{code-cell}
ts = gig.to_tree_sequence().simplify(gig.sample_nodes(time=0).ids)
print(ts.num_trees, "trees. The first one is:")
print(ts.first().draw_text())
```

Simulations that also include structural variants, such as inversions, need to place
recombination breakpoints at equivalent positions in the two parents, which is what
{meth}`~giglib.Tables.find_mrca_regions` is for. See {ref}`sec_usage_simulations` for examples
of such simulations.
