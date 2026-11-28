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

(sec_tools_simulation_single_origin)=

# Genomes from a single base pair

```{note}
Stages 1 to 3 of this simulator are written: neutral and clonal, then selection on genome length,
then crossovers placed by homology. The working plan, and progress, are in {issue}`221`.
```

This forward-time simulator starts from a single genome, one base pair long, and grows a
population of haploid genomes from it, whose lengths change by tandem duplication and deletion.
Every position of every genome then descends from that one base, so all positions are homologous
to each other at some level: the whole genome is one gene family. The aims are to:

- **test the {ref}`homology tools <sec_tools_homology>`** on the hardest case for them, where
  everything is homologous, and recent homology must be picked out from older homology between
  copies (see the {ref}`homology design note <sec_design_homology>`);
- **measure how they scale** with genome length, the number of duplications and deletions, and
  the depth of the graph;
- **check itself**, as genome lengths and the homology between a child and its parent follow
  directly from the recorded events, and the spread of genome lengths can be predicted.

It isn't meant to be a realistic model of genome evolution.

## The model

The population grows from the single genome to `N` haploid genomes (doubling each generation),
then stays at that size (Wright-Fisher). Each child copies its parent, and occasionally has a
tandem duplication or a deletion, whose lengths are drawn from a geometric distribution (with mean
`mean_length`), and redrawn if longer than the genome (or, for a deletion, if it would remove the
genome's last position). Capping the length instead would make deleting all but one position of a
short genome common, so many genomes would collapse to a single position. A duplication of positions `[s, s + L)` of a parent of length `n` is recorded by three
iedges, which give the child's positions in terms of the parent's:

| Child positions | Parent positions |
|---|---|
| `[0, s + L)` | `[0, s + L)` |
| `[s + L, s + 2L)` | `[s, s + L)` (the copy) |
| `[s + 2L, n + L)` | `[s + L, n)` |

**Choosing the rates.** While genomes are short, duplications and deletions are limited by the
length of the genome, so a deletion usually removes most of it, while a duplication at most
doubles it. With equal rates of duplication and deletion, genomes therefore stay very short. For
them to grow, duplications need to be commoner than deletions. Then the mean length grows (by
about `mean_length` times the difference in rates, per generation, once genomes are long), but
the spread of lengths within the population settles down, as any two genomes share a fairly
recent common ancestor (on average `N` generations ago). Each generation adds a variance of about
`Vm = (duplication_rate + deletion_rate) × E[L²]` to a lineage's length (where
`E[L²] = 2 m² - m` for a geometric length with mean `m`), so the standard deviation of lengths in
the population is about `√(N Vm)`.

```{code-cell}
from giglib.tools.simulation.single_origin import sim_single_origin

N, generations = 100, 1000
duplication_rate, deletion_rate, mean_length = 0.05, 0.02, 10
gig = sim_single_origin(
    N,
    generations,
    duplication_rate=duplication_rate,
    deletion_rate=deletion_rate,
    mean_length=mean_length,
    random_seed=1,
)
print(len(gig.nodes), "genomes, joined by", len(gig.iedges), "iedges")
```

Here are the lengths of the genomes in each generation, and their standard deviation, compared
with the prediction. The lengths drift a lot, and the standard deviation fluctuates around the
prediction, as it depends on the random genealogy of the population.

```{code-cell}
:tags: [hide-input]

import matplotlib.pyplot as plt
import numpy as np

times = gig.tables.nodes.time
lengths = np.array([gig.sequence_length(n) for n in range(len(gig.nodes))])
lengths[0] = 1  # the single base at the start (which has no iedges above it)
generation = generations - np.arange(generations, -1, -1)
in_generation = [lengths[times == generations - g] for g in generation]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].fill_between(generation, [x.min() for x in in_generation], [x.max() for x in in_generation], alpha=0.3, label="range")
axes[0].plot(generation, [x.mean() for x in in_generation], label="mean")
axes[0].set_xlabel("generation")
axes[0].set_ylabel("genome length")
axes[0].legend()
Vm = (duplication_rate + deletion_rate) * (2 * mean_length**2 - mean_length)
axes[1].plot(generation, [x.std() for x in in_generation], label="in the simulation")
axes[1].axhline(np.sqrt(N * Vm), linestyle="--", color="black", label="predicted")
axes[1].set_xlabel("generation")
axes[1].set_ylabel("standard deviation of genome length")
axes[1].legend()
plt.show()
```

## Homology

As everything is homologous, every pair of positions, one in genome `u` and one in genome `v`, has
a time to its most recent common ancestor (TMRCA). What matters is each position's *closest*
homologues (those with the most recent common ancestor, usually allelic), rather than its
*remote* homology through older ancestors, e.g. with other copies of a region (see
{ref}`finding homology <sec_usage_homology>`). So most of the plots here use
{meth}`~giglib.Tables.find_homology` with `mode="closest_homologues"`, whose output is usually
much smaller than all the homology (`mode="all"`), which covers every pair of positions. (It can
still cover many pairs, if many copies are equally close: see below.)

As soon as a genome replicates, its two children share every position at the split, so the dot
plot of any two genomes has a strong leading diagonal (see
{ref}`homology in the concepts <sec_concepts_homology>`): their allelic homology, with steps where
later duplications or deletions shifted one genome. This is why two genomes usually have a single
long chain of homology: replication copies the whole genome at once, so the allelic homology
starts as one unbroken line, and chains join up the pieces that later duplications and deletions
break it into. Duplications made after the two lineages split give lines off the diagonal with the
*same* TMRCA as the diagonal (both copies are equally close to the single copy in the other
genome), and those made before it give older, lighter lines.

Each generation adds `N` genomes, most of which aren't needed to describe the ancestry of the
last generation. So before finding homology between genomes of the last generation, we
{meth}`~giglib.Graph.simplify` the graph, which keeps the homology between them unchanged (see
{ref}`sec_design_simplification`), but makes finding it much faster:

```{code-cell}
simplified = gig.simplify()  # the samples (the last generation) are nodes 0 to N - 1
print(len(simplified.nodes), "genomes, joined by", len(simplified.iedges), "iedges")
```

Here are the closest homologues of two pairs of genomes in the last generation, and of the most
recent genome with a duplication against its parent (in the full graph, as its parent is not
kept by simplification):

```{code-cell}
from giglib.tools import homology

# The most recent genome with a duplication, i.e. that is longer than its parent: nodes are
# numbered in the order they were made, so the most recent have the highest IDs
for child in reversed(range(1, len(gig.nodes))):
    parent = gig.iedges_for_child(child)[0].parent
    if gig.sequence_length(child) > gig.sequence_length(parent):
        break

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (u, v) in zip(axes[:2], [(0, 1), (2, 3)]):
    homology.plot(simplified.tables, u, v, mode="closest_homologues", ax=ax)
homology.plot(gig.tables, child, parent, mode="closest_homologues", ax=axes[2])
axes[2].set_title("a genome against its parent")
plt.show()
```

As every position descends from the single base, *all* the homology between two genomes covers
every pair of positions. So rather than a dot plot, it is clearer as a matrix, with a cell for
each pair of positions, coloured by the time of their MRCA. The recent allelic homology runs
along the diagonal (with its copies, made since the lineages split), within blocks of older
homology between copies made earlier, nested like the clades of a tree:

```{code-cell}
:tags: [hide-input]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (u, v) in zip(axes, [(0, 1), (2, 3)]):
    # The time of the MRCA of each pair of positions, from all the homology
    tmrca = np.full((simplified.sequence_length(v), simplified.sequence_length(u)), np.nan)
    for h in simplified.find_homology(u, v, mode="all"):
        for x in range(h.u.left, h.u.right):
            tmrca[h.map.position(x), x] = simplified.nodes[h.mrca].time
    image = ax.imshow(tmrca, origin="lower", cmap="viridis")
    fig.colorbar(image, ax=ax, label="TMRCA", shrink=0.8)
    ax.set_xlabel(f"position in genome {u}")
    ax.set_ylabel(f"position in genome {v}")
plt.show()
```

The same idea applies within a single genome ({issue}`135`). A dot plot of a genome against itself
has each position's homology with itself on the diagonal (at the time of the genome, 0), and its
paralogous homology off the diagonal, symmetric about it. Two copies of a region meet in the genome
that was duplicated, so each line off the diagonal is coloured by when that duplication happened.
The closest homologues of each position are its nearest paralogues; all the homology adds the
older copies, back to the single base:

```{code-cell}
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, mode in zip(axes, ["closest_homologues", "all"]):
    homology.plot(simplified.tables, 0, 0, mode=mode, ax=ax)
    ax.set_title(mode)
plt.show()
```

## Selection on genome length

With `target_length` and `fitness_width`, there is stabilising selection on genome length: a
genome of length `n` has fitness `exp(-(n - T)² / (2 Vs))`, where `T` is the target length and
`Vs` is `fitness_width` squared, and parents are chosen in proportion to their fitness. In a large
population, mutation (here, duplications and deletions) and selection balance when the variance of
lengths `V` satisfies `V² = Vm (V + Vs)`. Drift also limits the spread (to about `N Vm`, as above),
so the spread is a little less than this predicts. As duplications are commoner than deletions,
mutation pushes lengths up, by about `mean_length × (duplication_rate - deletion_rate)` per
generation, so the mean settles a little above the target, where selection pulls lengths down as
much (by about `V / Vs` times the distance from the target).

```{code-cell}
target_length, fitness_width = 200, 50
selected = sim_single_origin(
    N,
    generations,
    duplication_rate=duplication_rate,
    deletion_rate=deletion_rate,
    mean_length=mean_length,
    target_length=target_length,
    fitness_width=fitness_width,
    random_seed=1,
)
```

```{code-cell}
:tags: [hide-input]

times = selected.tables.nodes.time
lengths = np.array([selected.sequence_length(n) for n in range(len(selected.nodes))])
lengths[0] = 1
in_generation = [lengths[times == generations - g] for g in generation]
Vs = fitness_width**2
V = (Vm + np.sqrt(Vm**2 + 4 * Vm * Vs)) / 2  # the balance of mutation and selection

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].fill_between(generation, [x.min() for x in in_generation], [x.max() for x in in_generation], alpha=0.3, label="range")
axes[0].plot(generation, [x.mean() for x in in_generation], label="mean")
axes[0].axhline(target_length, linestyle="--", color="black", label="target")
axes[0].set_xlabel("generation")
axes[0].set_ylabel("genome length")
axes[0].legend()
axes[1].plot(generation, [x.std() for x in in_generation], label="in the simulation")
axes[1].axhline(np.sqrt(V), linestyle="--", color="black", label="predicted (mutation and selection)")
axes[1].set_xlabel("generation")
axes[1].set_ylabel("standard deviation of genome length")
axes[1].legend()
plt.show()
```

Once lengths reach the target, the genomes stop growing, but duplications and deletions go on
replacing each other, so the homology between two genomes is broken up much as before:

```{code-cell}
simplified_selected = selected.simplify()
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (u, v) in zip(axes, [(0, 1), (2, 3)]):
    homology.plot(simplified_selected.tables, u, v, mode="closest_homologues", ax=ax)
plt.show()
```

## Crossovers

With `recombination_rate`, a child can have two parents, and inherits the start of one and the end
of the other, joined at a crossover. As in the
{ref}`microsatellite simulator <sec_tools_simulation_microsatellites_forward>`, a crossover is
only placed in a long *chain* of homology between the parents (at least `min_chain_length`
positions, with gaps of at most `max_gap`), as a stand-in for allelic homology. Where a chain
pairs copies that aren't allelic (e.g. where one parent lost a copy), the crossover is unequal,
and changes the number of copies. Genomes need to be longer than `min_chain_length` for
crossovers to happen, so here selection holds them near a length of 200.

Each crossover finds the homology between two parents, tracing them back through the graph. So
the simulation can simplify the graph every `simplify_interval` generations, with respect to the
current generation, and at the end. As simplification doesn't change the homology between the
genomes it keeps, this gives the same genomes as not simplifying (which is tested), but here it
takes about 7 seconds rather than 25:

```{code-cell}
recombinant = sim_single_origin(
    30,
    300,
    duplication_rate=duplication_rate,
    deletion_rate=deletion_rate,
    mean_length=mean_length,
    target_length=200,
    fitness_width=50,
    recombination_rate=0.5,
    min_chain_length=50,
    max_gap=10,
    simplify_interval=10,
    random_seed=1,
)
print(len(recombinant.nodes), "genomes, joined by", len(recombinant.iedges), "iedges")
```

The homology between two genomes is now a mosaic: where they inherit from parents that share a
recent common ancestor, it is recent, and elsewhere older:

```{code-cell}
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (u, v) in zip(axes, [(0, 1), (2, 3)]):
    homology.plot(recombinant.tables, u, v, mode="closest_homologues", ax=ax)
plt.show()
```

## Stages

1. **Neutral, clonal, with duplications and deletions** (done). The outputs are the
   distribution of genome lengths, compared with the prediction, and homology plots between
   genomes, between a child and its parent, and of a genome with itself.
2. **Selection on genome size** (done), with the same outputs.
3. **Crossovers placed by homology** (done). A recombinant child has two parents, and a crossover
   placed in a long chain of homology between them, as in the microsatellite simulator, so
   unequal crossing over, between copies that aren't allelic, follows naturally. Other rules
   (see {ref}`sec_design_recombination`), and a separate rate of unequal crossing over
   ({issue}`83`), could be tried, as could choosing breakpoints by rejection sampling
   ({issue}`102`), which would be faster.
4. **Evaluation cases.** The time and memory used per generation, and by the homology tools for
   each pair of genomes (including the number of homology records), as genomes get longer and the
   graph deeper.

## What stage 1 has shown

- **Homology can be all at one MRCA.** If two genomes' common ancestor lived while genomes were
  still very short, every position of each is a copy of the same few positions, made after their
  lineages split. Then every position of one is a closest homologue of every position of the
  other, through the same MRCA, and a dot plot is a solid block. To see structure, the population
  needs to be small enough (or the simulation long enough) that lineages split after the genomes
  have grown. It also means many records of homology (one for each pair of copies), which is a
  reason to consider a grouped form of the records (see the open questions of the
  {ref}`homology design note <sec_design_homology>`).
- **Finding homology needed to be faster.** At first, finding the closest homologues of two
  genomes here took minutes, as every lineage was traced back to the single base at the start,
  and so did finding all the homology. Stopping once every position has found homology
  ({pr}`230`), and giglib's own operations on intervals ({pr}`234`), brought these down to a
  fraction of a second, and a second or two.
- **Simplification.** Each generation adds `N` nodes, most of which are not needed to describe
  the ancestry of the samples. {meth}`~giglib.Tables.simplify` removes them (see
  {ref}`sec_design_simplification`), without changing the homology between the samples. Here,
  it takes a fraction of a second to reduce about 100,000 genomes to a few hundred, after which
  finding all the homology between two genomes takes a second or two, rather than about a
  minute.

```{eval-rst}
.. automodule:: giglib.tools.simulation.single_origin
  :members:
```
