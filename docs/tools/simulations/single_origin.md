(sec_tools_simulation_single_origin)=

# Genomes from a single base pair (planned)

```{note}
This simulator is planned, but not yet written. This page describes what it will do. The
working plan, and progress, are in {issue}`221`.
```

This forward-time simulator starts from a single genome, one base pair long, and grows a
population of haploid genomes from it, whose lengths change by tandem duplication and deletion.
Every position of every genome then descends from that one base, so all positions are homologous
to each other at some level: the whole genome is one gene family. The aims are to:

- **test the homology tools** ({ref}`sec_tools_homology`) on the hardest case for them, where
  everything is homologous, and recent homology must be picked out from older homology between
  copies (see {ref}`sec_design_homology`);
- **measure how they scale** with genome length, the number of duplications and deletions, and
  the depth of the graph;
- **check itself**, as genome lengths and the homology between a child and its parent follow
  directly from the recorded events, and the spread of genome lengths can be predicted.

It isn't meant to be a realistic model of genome evolution.

## The model

The population grows from the single genome to `N` haploid genomes (doubling each generation),
then stays at that size (Wright-Fisher). Each child copies its parent, and occasionally has a
tandem duplication or a deletion, whose lengths are drawn from the same distribution (e.g.
geometric, with a mean of 20 positions), but no longer than the genome (and a deletion never
removes a genome's last position). A duplication of positions `[s, s + L)` of a parent of length
`n` is recorded by three iedges, which give the child's positions in terms of the parent's:

| Child positions | Parent positions |
|---|---|
| `[0, s + L)` | `[0, s + L)` |
| `[s + L, s + 2L)` | `[s, s + L)` (the copy) |
| `[s + 2L, n + L)` | `[s + L, n)` |

Without selection, the mean genome length wanders at random, but the spread of lengths in the
population settles down, as any two genomes share a fairly recent common ancestor (on average
`N` generations ago). With selection towards a target length (fitness falling away from it, e.g.
from 1000 positions), the lengths stay around the target, with a spread that depends on the
population size, the rates and lengths of duplications and deletions, and the strength of
selection.

## Homology

As everything is homologous, every pair of positions, one in genome `u` and one in genome `v`, has
a time to its most recent common ancestor (TMRCA). What matters is each position's *closest*
homologues (those with the most recent common ancestor, usually allelic), rather than its
*remote* homology through older ancestors, e.g. with other copies of a region (see
{ref}`sec_usage_homology`). So the simulator will use
{meth}`~giglib.Tables.find_homology` with `mode="closest_homologues"`, whose output, unlike all
the homology (`mode="all"`), doesn't grow with the square of genome length.

As soon as a genome replicates, its two children share every position at the split, so the dot
plot of any two genomes has a strong leading diagonal (see {ref}`sec_concepts_homology`): their
allelic homology, with steps where later duplications or deletions shifted one genome. This is
why two genomes usually have a single long chain of homology: replication copies the whole genome
at once, so the allelic homology starts as one unbroken line, and chains join up the pieces that
later duplications and deletions break it into. Duplications made after the two lineages split
give lines off the diagonal with the *same* TMRCA as the diagonal (both copies are equally close
to the single copy in the other genome), and those made before it give older, lighter lines.

The same idea applies within a single genome ({issue}`135`): a dot plot of a genome against itself
would have the genome's homology with itself on the diagonal, and its duplications off the
diagonal, symmetric about it, coloured by when they happened. This needs
{meth}`~giglib.Tables.find_homology` to allow `u` and `v` to be the same genome, which it doesn't
yet.

## Stages

1. **Neutral, clonal, with duplications and deletions.** The outputs are the distribution of
   genome lengths, compared with the prediction, and homology plots between random pairs of
   genomes, between a child and its parent, and of single genomes against themselves.
2. **Selection on genome size,** with the same outputs.
3. **Crossovers placed by homology.** Each child has two parents, and a crossover is placed in a
   long chain of closest homology between them, as in the microsatellite simulator. The rule for
   which chains count as allelic will be chosen after looking at the plots from the first two
   stages (see {ref}`sec_design_recombination`). Unequal crossing over, between copies that
   aren't allelic, then follows naturally.
4. **Evaluation cases.** The time and memory used per generation, and by the homology tools for
   each pair of genomes (including the number of homology records), as genomes get longer and the
   graph deeper.

## Expected difficulties

- **The amount of homology.** All homology between two genomes of length `n` covers up to about
  `n²` pairs of positions, so measurements need to use closest homology, or a limit on the age
  of the homology (the `time_cutoff` of {meth}`~giglib.Tables.find_homology`).
- **No simplification yet.** Each generation adds `N` nodes, which are never removed, as a full
  equivalent of _tskit_'s `simplify()` doesn't exist yet (see {ref}`sec_design_simplification`).
  Long runs will show how much it is needed.
