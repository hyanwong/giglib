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
Stage 1 of this simulator (neutral and clonal) is written; the later stages are planned. The
working plan, and progress, are in {issue}`221`.
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
`mean_length`), but no longer than the genome (and a deletion never removes a genome's last
position). A duplication of positions `[s, s + L)` of a parent of length `n` is recorded by three
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

N, generations = 20, 400
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
with the prediction. With only 20 genomes, the lengths drift a lot: the population sometimes
collapses to short genomes, when a lineage with a short genome happens to take over.

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
{ref}`finding homology <sec_usage_homology>`). So the plots here use
{meth}`~giglib.Tables.find_homology` with `mode="closest_homologues"`, whose output, unlike all
the homology (`mode="all"`), doesn't grow with the square of genome length.

As soon as a genome replicates, its two children share every position at the split, so the dot
plot of any two genomes has a strong leading diagonal (see
{ref}`homology in the concepts <sec_concepts_homology>`): their allelic homology, with steps where
later duplications or deletions shifted one genome. This is why two genomes usually have a single
long chain of homology: replication copies the whole genome at once, so the allelic homology
starts as one unbroken line, and chains join up the pieces that later duplications and deletions
break it into. Duplications made after the two lineages split give lines off the diagonal with the
*same* TMRCA as the diagonal (both copies are equally close to the single copy in the other
genome), and those made before it give older, lighter lines.

Here are the closest homologues of two pairs of genomes in the last generation, and of the most
recent genome with a duplication against its parent:

```{code-cell}
from giglib.tools import homology

samples = gig.sample_nodes().ids
# The most recent genome with a duplication, i.e. that is longer than its parent: nodes are
# numbered in the order they were made, so the most recent have the highest IDs
for child in reversed(range(1, len(gig.nodes))):
    parent = gig.iedges_for_child(child)[0].parent
    if gig.sequence_length(child) > gig.sequence_length(parent):
        break

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (u, v) in zip(axes, [(samples[0], samples[1]), (samples[2], samples[3]), (child, parent)]):
    homology.plot(gig.tables, u, v, mode="closest_homologues", ax=ax)
axes[2].set_title("a genome against its parent")
plt.show()
```

The same idea applies within a single genome ({issue}`135`): a dot plot of a genome against itself
would have the genome's homology with itself on the diagonal, and its duplications off the
diagonal, symmetric about it, coloured by when they happened. This needs
{meth}`~giglib.Tables.find_homology` to allow `u` and `v` to be the same genome, which it doesn't
yet.

## Stages

1. **Neutral, clonal, with duplications and deletions** (done). The outputs are the
   distribution of genome lengths, compared with the prediction, and homology plots between
   genomes, and between a child and its parent.
2. **Selection on genome size,** with the same outputs.
3. **Crossovers placed by homology.** Each child has two parents, and a crossover is placed in a
   long chain of closest homology between them, as in the microsatellite simulator. The rule for
   which chains count as allelic will be chosen after looking at the plots from the first two
   stages (see {ref}`sec_design_recombination`). Unequal crossing over, between copies that
   aren't allelic, then follows naturally.
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
- **Finding the closest homologues needed a stopping rule.** Tracing every lineage to the single
  base at the start took minutes; stopping once every position has found homology ({pr}`230`)
  takes seconds.
- **No simplification yet.** Each generation adds `N` nodes, which are never removed, as a full
  equivalent of _tskit_'s `simplify()` doesn't exist yet (see {ref}`sec_design_simplification`).

```{eval-rst}
.. automodule:: giglib.tools.simulation.single_origin
  :members:
```
