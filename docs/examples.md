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

{meth}`~giglib.Graph.find_homology` finds the homology between two genomes: the segments of each
that trace back to the same segment of a common ancestor, and their most recent common ancestor
(MRCA):

```{code-cell}
u, v = gig.sample_nodes(time=0).ids[:2]
for h in gig.find_homology(u, v):
    print(f"Node {h.mrca} (time {gig.nodes[h.mrca].time:g}) is the MRCA over {h.u.left}-{h.u.right}")
```

Again, this can be converted to a tree sequence, which now has several local trees:

```{code-cell}
ts = gig.to_tree_sequence().simplify(gig.sample_nodes(time=0).ids)
print(ts.num_trees, "trees. The first one is:")
print(ts.first().draw_text())
```

Simulations that also include structural variants, such as inversions, need to place
recombination breakpoints at equivalent positions in the two parents, which is what
{meth}`~giglib.Tables.find_homology` is for. See {ref}`sec_usage_simulations` for examples
of such simulations.

(sec_examples_homology)=

## Finding homology

{meth}`~giglib.Tables.find_homology` finds which segments of two genomes descend from the same
segment of a common ancestor (see {ref}`finding homology <sec_usage_homology>`). Here is a small
GIG to try it on. An ancestor `A` has two copies of a region of an older ancestor `R` (a tandem
duplication). Genome `u` inherits both copies, and genome `v` only the first, with its second half
inverted. In the diagram, each genome is a horizontal bar, and each iedge is a band joining an
interval of a child (below) to the interval of its parent (above) that it inherits; a band that
crosses itself is an inversion:

```{code-cell}
:tags: [hide-input]

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

import giglib as gigl

tables = gigl.Tables()
R = tables.nodes.add_row(10)
A = tables.nodes.add_row(5)
u = tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
v = tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
tables.iedges.add_row(A, R, child_left=0, child_right=10, parent_left=0, parent_right=10)
tables.iedges.add_row(A, R, child_left=10, child_right=20, parent_left=0, parent_right=10)
tables.iedges.add_row(u, A, child_right=20)  # both copies
tables.iedges.add_row(v, A, child_right=5)  # the first half of the first copy...
tables.iedges.add_row(v, A, child_left=5, child_right=10, parent_left=10, parent_right=5)  # ...and the second, inverted
tables.sort()
tables.validate()

# Where to draw each genome: (y, x offset, label)
rows = {R: (2, 10, "R"), A: (1, 5, "A"), v: (0, 0, "v"), u: (0, 15, "u")}


def draw(ax, highlight=None):
    for node, (y, x, label) in rows.items():
        length = max(ie.child_right for ie in tables.iedges if ie.child == node) if node != R else 10
        ax.add_patch(Rectangle((x, y - 0.1), length, 0.2, color="0.6"))
        ax.text(x - 0.5, y, label, ha="right", va="center", fontsize=12)
        ax.text(x, y - 0.18, "0", ha="center", va="top", fontsize=8)
        ax.text(x + length, y - 0.18, str(length), ha="center", va="top", fontsize=8)
    for ie in tables.iedges:
        (cy, cx, _), (py, px, _) = rows[ie.child], rows[ie.parent]
        corners = [
            (cx + ie.child_left, cy + 0.1),
            (cx + ie.child_right, cy + 0.1),
            (px + ie.parent_right, py - 0.1),
            (px + ie.parent_left, py - 0.1),
        ]
        ax.add_patch(Polygon(corners, color="tab:blue", alpha=0.25))
    if highlight is not None:
        # Mark the segments of u, v and their MRCA in a homology record
        for node, segment in [(u, highlight.u), (v, highlight.v), (highlight.mrca, highlight.mrca_segment)]:
            y, x, _ = rows[node]
            left, right = sorted([segment.left, segment.right])
            ax.add_patch(Rectangle((x + left, y - 0.1), right - left, 0.2, color="tab:orange"))
    ax.set_xlim(-2, 37)
    ax.set_ylim(-0.5, 2.4)
    ax.axis("off")


fig, ax = plt.subplots(figsize=(8, 3))
draw(ax)
plt.show()
```

The result is a list of {class}`~giglib.Homology` records, sorted by position in `u`. Each record
says that a segment of `u` and a segment of `v` descend from the same segment (`mrca_segment`) of
their most recent common ancestor (`mrca`):

```{code-cell}
for h in tables.find_homology(u, v):
    print(h)
```

Positions in `u` run from `u.left` to `u.right`, and correspond, in order, to positions running
from `v.left` towards `v.right` (and likewise in the MRCA). So the first record says that
positions 0 to 5 of `u` are homologous to positions 0 to 5 of `v`, through positions 0 to 5 of
`A`. In the second, `v.left` is greater than `v.right`: `v`'s segment runs backwards, because of
the inversion, so positions 5 to 10 of `u` match positions 9 down to 5 of `v`. This record is
highlighted here:

```{code-cell}
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(8, 3))
draw(ax, highlight=tables.find_homology(u, v)[1])
plt.show()
```

Each record has some properties that make it easier to use:

```{code-cell}
h = tables.find_homology(u, v)[1]
print("Number of positions:", h.length)
print("Reversed (e.g. by an inversion):", h.is_reversed)
print("The map from u to v:", h.map)
print("Position 5 of u is homologous to position", h.map.position(5), "of v")
print("Time of the MRCA:", tables.nodes.time[h.mrca])
```

{class}`~giglib.Segment` records can be passed as keyword arguments, e.g. to get the sequence of
a segment (if the GIG has mutations) with `gig.haplotype(u, **h.u)`.

### Which homology is found

So far, there is no homology for `u`'s second copy of the region (positions 10 to 20). By default
(`mode="first_mrca"`), each lineage stops being traced at the first MRCA it reaches. The lineage
of `v`'s copy reaches `A`, where it meets that of `u`'s first copy, and stops. So it isn't there
to meet the lineage of `u`'s second copy, which only meets it further back, in `R`. The other
modes find this homology too:

```{code-cell}
names = {R: "R", A: "A"}
for mode in ["first_mrca", "closest_homologues", "all"]:
    print(f"{mode}:")
    for h in tables.find_homology(u, v, mode=mode):
        mrca = names[h.mrca]
        print(f"  u {h.u.left}-{h.u.right} ~ v {h.v.left}-{h.v.right}, through {mrca} (time {tables.nodes.time[h.mrca]:g})")
```

Here, `"closest_homologues"` and `"all"` give the same result, as the homology through `R` is the
closest that `u`'s second copy has. They would differ if `v` also had the second copy: then each
copy's closest homologue would be the same copy in the other genome, through `A`, and only
`"all"` would also find the older homology between the two copies, through `R`.

The homology between two genomes is easiest to see in a dot plot (see
{ref}`the homology tools <sec_tools_homology>`), coloured by the time of the MRCA:

```{code-cell}
from giglib.tools import homology

ax = homology.plot(tables, u, v)
ax.set_xlabel("position in u")
ax.set_ylabel("position in v")
plt.show()
```
