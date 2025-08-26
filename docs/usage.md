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

In [`tskit`](https://tskit.dev) we use edge annotations to describe which pieces of DNA are inherited in terms of a left and right coordinate.
It should be possible to extend this to track the L & R in the edge *child*, and the L & R in the edge *parent* separately.
The left and right values in each case refer to the coordinate system of the child and parent respectively.

We call an extended tree-sequence-like structure such as this, a GIG ("Generalised (or Genetic) Inheritance Graph"). For
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
[here](https://github.com/hyanwong/giglib/issues/41#issuecomment-1858530867)
for more discussion.

### Duplications

A tandem duplication is represented by two iedges, one for each duplicated region:

```
{parent: P, child: C, child_left: 100, child_right: 200, parent_left: 100, parent_right: 200}
{parent: P, child: C, child_left: 200, child_right: 300, parent_left: 100, parent_right: 200}
```

Or one of the iedges could represent a non-adjacent duplication (e.g. corresponding to a transposable element):
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


:::{todo}
Add examples of MRCA finding, sample resolving, simplification and (when mutations have been added) haplotype decoding.
:::
