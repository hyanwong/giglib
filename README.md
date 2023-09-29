# GeneticInheritanceGraph

A basic repo for kicking around ideas for the "(General) Genetic Inheritance Graph" structure, which should be able to capture genetic inheritance with
genomic rearrangements, and hence describe inherited structural variation.

## Basic idea

In [`tskit`](tskit.dev) we use edge annotations to describe which pieces of DNA are inherited in terms of a left and right coordinate.
It should be possible to extend this to track the L & R in the edge *child*, and the L & R in the edge *parent* separately.
The left and right values in each case refer to the coordinate system of the child and parent respectively.

I'm calling an extended tree-sequence-like structure such as this, a GIG ("General (or Genetic) Inheritance Graph").

### Inversions

The easiest example is an inversion. This would be an edge like

```
{parent: P, child: C, child_lft: 100, child_rgt: 200, parent_lft: 200, parent_rgt: 100}
```

A tandem duplication is represented by two edges, one for each duplicated region:

```
{parent: P, child: C, child_lft: 100, child_rgt: 200, parent_lft: 100, parent_rgt: 200}
{parent: P, child: C, child_lft: 200, child_rgt: 300, parent_lft: 100, parent_rgt: 200}
```

Or a non-adjacent duplication:
```
{parent: P, child: C, child_lft: 250, child_rgt: 350, parent_lft: 100, parent_rgt: 200}
```

### Deletions

A deletion simply occurs when no material from the parent is transmitted to any of its children (and the coordinate system is shrunk)

```
# Deletion of parental region from 200-300
{parent: P, child: C, child_lft: 100, child_rgt: 200, parent_lft: 100, parent_rgt: 200}
{parent: P, child: C, child_lft: 200, child_rgt: 300, parent_lft: 300, parent_rgt: 400}
```

This brings with it a load of extra complexities, and it’s unclear if the efficiency of the tskit approach,
with its edge indexing etc, will port in any meaningful way to this new structure.

## Meaning of trees
The interpretation of the trees may be rather different from that in tskit. In particular,
if there is a duplication, it could be interpreted as creating two tree tips within a single sample.
This would mean that there is no longer a 1-to-1 mapping from samples to tree tips, which
breaks one of the fundamental tskit ideas.
