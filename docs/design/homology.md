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

(sec_design_homology)=

# Homology

**Status: Proposed.** Homology, as recorded by a GIG, and the mechanisms _giglib_ can provide for
finding it. How simulators can use it to restrict recombination is in the companion note,
{ref}`sec_design_recombination`. Chains are a *heuristic*: a testable starting point, not a
definition of which homology is allelic ({issue}`202`). Related: {issue}`111` (levels of homology),
{issue}`140` (the homology table), {issue}`210` (maps to the roots), {issue}`102` (homology of a
single position), {ref}`sec_design_simplification` (duplication nodes). Testing: {issue}`202`
(simulations with microsatellites and a centromere) and {issue}`221` (genomes from a single base
pair).

## Summary

- **Homology needs no reference.** Two positions, in two genomes, are *homologous* if they trace
  back to the same position in a common ancestor. A GIG records exactly this, so homology is
  defined without any reference genome or shared coordinate system, as _giglib_ requires (genomes
  may differ greatly in length, in the number of copies of a region, and even in their number of
  chromosomes).
- **Length variation breaks homology into pieces,** e.g. at microsatellites or centromeric repeats
  whose lengths differ between the two genomes. So allelic homology is measured over **chains** of
  pieces, which may be separated by small shifts, rather than over single pieces.
- **Policy belongs to simulators, mechanism can be in giglib.** The *rules* for recombination
  (where crossovers may happen, thresholds and rates) are choices for each simulator, not part of
  _giglib_'s model. But _giglib_ can provide reference-free *mechanisms* that make the common
  rules easy and fast to implement: finding the homology between two genomes, chaining it, and
  caches or indexes such as the homology table.

## Homology in a GIG

Each iedge maps an interval of a child's genome onto an interval of its parent's genome (a
{class}`~giglib.PositionMap`: a shift, reversed for an inversion). Tracing a position up through
the iedges gives its position in each ancestor. Two positions, in genomes `u` and `v`, are
homologous if they reach the same position in some common ancestor. The position at which this
first happens is in their most recent common ancestor (MRCA) at that position.
{meth}`~giglib.Tables.find_homology` finds the regions of `u` and `v` that are homologous in
this way, and where they are in each genome: with `mode="all"`, all of them; by default
(`mode="first_mrca"`), it stops tracing each lineage at its first MRCA, so it can miss older
homology of duplicated regions (see {ref}`finding homology <sec_usage_homology>`).

This is homology by descent, which is reference-free. It is not the same as similarity of sequence,
which is what the cell's recombination machinery detects. But old homology has had more time to
accumulate mutations, so the time to the MRCA (TMRCA) is a reasonable proxy for divergence, if
needed.

## The homology map between two genomes

Composing the maps from `u` up to an MRCA, and back down to `v`, gives a single
{class}`~giglib.PositionMap` from positions in `u` to homologous positions in `v`. Along the
genome of `u`, this map is piecewise: it changes wherever `u` and `v` differ in length (an
insertion or deletion between them), and wherever the homology comes from a different place (e.g.
a TE copy). We call each maximal piece with a single map a **homology block**: an interval of `u`,
the homologous interval of `v`, and the map between them. A block can span several MRCA regions
(different ancestors at different positions), as long as the map is the same.

A block has a length, an orientation (whether the map is reversed), and a pair of chromosomes. The
records found by {meth}`~giglib.Tables.find_homology` each have a single map, so blocks can be
found by merging records that are adjacent in `u` and have the same map. As a block can span several
MRCAs, it has no single MRCA or time to the MRCA (TMRCA): it keeps the MRCA of each of its
sub-intervals (as a list of intervals of `u`, each with its MRCA node and time), so that a rule
that depends on the TMRCA can use its value at a particular position.

## Why single blocks aren't enough

A first idea is to allow a crossover only well inside a *long* block. That separates TE copies
(their blocks are only as long as the TE) from allelic positions (whose blocks are long). But it
fails in an important case: regions whose length often changes from one generation to the next,
such as microsatellites and centromeric repeats. Wherever `u` and `v` carry different numbers of
repeats, the map between them shifts, so the block ends. In a genome with many such regions,
allelic homology is chopped into short blocks, and a rule of "long blocks only" would wrongly
suppress ordinary crossovers.

The shifts are small, though: a few repeat units. And the blocks either side of a shift still line
up, in the same order. That is what chains capture.

## Chains

A **chain** is a series of homology blocks that are in the same order in both genomes, in the same
orientation, on the same pair of chromosomes, and separated by gaps of at most `g` positions in
each genome. Chains are a standard idea in comparing whole genomes (e.g. the "chains" used to align
genome assemblies), used here for homology by descent rather than similarity.

The clearest way to see this is a **dot plot**: plot each position of `u` against its homologous
position in `v`. Homology appears as line segments, one for each block. The plots on this page are
drawn by _giglib_'s own {func}`giglib.tools.homology.plot`, from small GIGs (show the code to see
how), and colour each segment of homology by its time to the most recent common ancestor (TMRCA):

```{code-cell}
:tags: [hide-input]

import matplotlib.pyplot as plt

import giglib as gigl
from giglib.tools import homology

# A small GIG: an old ancestor R, with a TE at its positions 100-105, and a recent common
# ancestor P of genomes u and v, which inherits R's positions 0-90. u has 3 extra
# microsatellite repeats at 30-32 (with no parent), and a TE copied from R at 80-85. v has
# a TE copied from R at 10-15, and 2 extra repeats at 58-59.
tables = gigl.Tables()
R = tables.nodes.add_row(10)
P = tables.nodes.add_row(1)
u = tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
v = tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
tables.iedges.add_row(P, R, child_right=91)
for child, pieces in [
    (u, [(0, 30, P, 0), (33, 80, P, 30), (80, 86, R, 100), (86, 100, P, 77)]),
    (v, [(0, 10, P, 0), (10, 16, R, 100), (16, 58, P, 10), (60, 99, P, 52)]),
]:
    for cl, cr, parent, pl in pieces:
        tables.iedges.add_row(child, parent, child_left=cl, child_right=cr, parent_left=pl, parent_right=pl + cr - cl)
tables.sort()
tables.validate()

ax = homology.plot(tables, u, v)
ax.annotate("TE in v", (10, 13), xytext=(22, 6), arrowprops={"arrowstyle": "->"})
ax.annotate("extra repeats in u", (31.5, 37), xytext=(36, 26), arrowprops={"arrowstyle": "->"})
ax.annotate("extra repeats in v", (55, 58.5), xytext=(58, 48), arrowprops={"arrowstyle": "->"})
ax.annotate("TE in u", (83, 85), xytext=(85, 72), arrowprops={"arrowstyle": "->"})
ax.set_xlabel("position in genome u")
ax.set_ylabel("position in genome v")
plt.show()
```

Here, `u` and `v` share a recent common ancestor, and each has gained a copy of a TE from a much
older one.

In this example, `u` and `v` are two homologous chromosomes, and the maximum gap `g` is at least
as long as the TE (6 positions):

- **Allelic homology** runs along the diagonal, from bottom left to top right, but is broken into
  five blocks, with a recent TMRCA (dark blue). Each break is a small shift: a horizontal step where `u` has material that
  `v` lacks (extra repeats, or a TE inserted in `u`), or a vertical step where `v` has material
  that `u` lacks. Joining the blocks across these small steps gives one long **chain**, covering
  almost the whole chromosome.
- **The two TE copies** (one inserted in `u`, one in `v`, from the same family) are homologous to
  each other, so they appear as a short segment, far from the diagonal, with an old TMRCA
  (yellow). It can't join the chain: reaching it from the allelic block before it in `u` would mean
  going *backwards* in `v`, and reaching the allelic block after it from the TE segment would need
  a jump far larger than `g`. So it
  is a chain on its own, only as long as the TE.

Each block is a point of the chain where a crossover could happen, but the chain's length is what
suggests that the homology is allelic. A crossover is then allowed well inside a long chain (the allelic
one), but not in a short one (the TE copies), however many shifts the long chain has. In the same way:

- **Different chromosomes, or chromosome arms,** give homology on a different pair of chromosomes,
  or far off the diagonal, so it can't join an allelic chain.
- **An inversion** (between `u` and `v`) gives a block in the opposite orientation, a segment
  running from top left to bottom right. It breaks the chain. But breaking the chain doesn't, on
  its own, suppress crossovers within the inversion: if the inversion is longer than `L`, it is a
  long (reversed) chain of its own. So suppressing crossovers within heterozygous inversions (as
  happens in practice, where they would give unbalanced gametes) needs an explicit rule: see
  {ref}`sec_design_recombination_rules`.
- **A large insertion or deletion** between `u` and `v` gives a step longer than `g`, so also breaks
  the chain. Crossovers near large heterozygous structural variants are indeed reduced in practice.
  The threshold `g` sets what counts as "large".

### Finding chains

Given the blocks between `u` and `v`, sorted by their position in `u`:

1. Link each block to the next block that is on the same pair of chromosomes, in the same
   orientation, and starts at most `g` positions after it ends, in both genomes (going forwards in
   `v` for a block in the same orientation, or backwards for a reversed one).
2. A chain is a maximal run of linked blocks. Its length is the total length of its blocks.

When there are several candidates for the next block (e.g. near a tandem duplication, where both
copies are close by), there are several possible chains through the blocks. As in genome
alignment, the natural choice is the chain with the most homology, found by a short dynamic
programming over the candidate links (though the alternatives are equally valid *candidate*
pairings, among which a simulator could also choose at random).

Only blocks within `g` of each other can be linked, so a search limited to those can be close to
linear in the number of blocks, i.e. in the number of differences in length between the two
genomes. A simple implementation that compares every pair of blocks is quadratic.

## Copy-number variation

Copy-number variants (CNVs) are regions present in different numbers of copies in different
genomes, e.g. tandem copies of a gene. All the copies of a region are homologous to each other (they
descend from one ancestral copy), so every copy in `u` is homologous to every copy in `v`. In a dot
plot, this gives a small grid of parallel segments. Here, region B is present in 3 tandem copies in
`u`, but 2 in `v`: it was duplicated in a common ancestor of `u` and `v`, and one of the copies was
duplicated again in `u`'s lineage:

```{code-cell}
:tags: [hide-input]

# An old ancestor R has A (0-30), B (30-40), C (40-80). B was duplicated in the common
# ancestor P of u and v (A, B1, B2, C), and v inherits P's genome. In u's lineage, P's B2 was
# duplicated again (A, B1, B2, B3, C).
tables = gigl.Tables()
R = tables.nodes.add_row(10)
P = tables.nodes.add_row(1)
u = tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
v = tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
for child, pieces in [
    (P, [(0, 40, R, 0), (40, 50, R, 30), (50, 90, R, 40)]),
    (v, [(0, 90, P, 0)]),
    (u, [(0, 50, P, 0), (50, 60, P, 40), (60, 100, P, 50)]),
]:
    for cl, cr, parent, pl in pieces:
        tables.iedges.add_row(child, parent, child_left=cl, child_right=cr, parent_left=pl, parent_right=pl + cr - cl)
tables.sort()
tables.validate()

ax = homology.plot(tables, u, v)
for x in (30, 40, 50, 60):
    ax.axvline(x, color="0.85", lw=0.5, zorder=0)
for y in (30, 40, 50):
    ax.axhline(y, color="0.85", lw=0.5, zorder=0)
ax.set_xlabel("position in genome u (A, B1, B2, B3, C)")
ax.set_ylabel("position in genome v (A, B1, B2, C)")
plt.show()
```

- **The allelic chain is ambiguous within the CNV.** Here, `v`'s second copy has equally recent
  homology with `u`'s second and third copies (the two recent segments that overlap in `v`). The
  chain with the most homology pairs `u`'s first two copies with `v`'s two copies, and treats
  `u`'s third copy as a gap (if it is no longer than `g`). But pairing `u`'s third copy with `v`'s
  second instead, with the gap before it, has just as much homology. This is the usual ambiguity of aligning CNVs, and is real: nothing in
  either genome says which copies are "the same".
- **Unequal crossovers change copy number.** A crossover between copies that are *not* paired in
  the chosen chain (e.g. the older segments, between the copies made in the common ancestor) is
  unequal crossing over, and gives one product with more copies and
  one with fewer. This is how CNVs grow and shrink, by non-allelic homologous recombination, so a
  simulator should be able to allow it, e.g. at a rate relative to allelic crossovers, between
  copies that lie close to the allelic chain (parallel segments near the diagonal).
- **Larger CNVs break the chain.** If the extra copies are longer than `g`, the chain breaks at the
  CNV, so crossovers are suppressed near it, apart from any unequal crossovers that the simulator
  allows.

So CNVs need no special representation: they are copies made by iedges, and their homology shows
up as parallel blocks. What a simulator needs to decide is how to choose among the equally good
pairings of copies for allelic crossovers, and at what rate to allow unequal ones.

## The homology table, as a cache

The *homology table* of {issue}`140` maps each genome directly to an ancestor, by composing the
iedges along the way. This **merges changes across generations**: e.g. a microsatellite that
expanded, contracted and expanded again over many generations becomes a single net shift, or none
at all if it returned to its ancestral length (and the pieces either side then merge). So the
number of pieces in a genome's map depends on how many places it differs from the ancestor, not on
how many generations it went through, and it saves tracing through every intermediate ancestor.

But it **can't merge differences between individuals**. The map from `u` to `v` is `u`'s map to the
ancestor, followed by the inverse of `v`'s. It has a piece boundary wherever the two genomes differ
in length, e.g. where they carry different numbers of repeats at a microsatellite, because that is
a real difference between them. So the homology table makes the blocks between `u` and `v` cheap to
find, and as few as possible, while chaining is still what decides which homology is allelic. The
homology table is an optional cache for speed, not part of the definition of homology.

(sec_design_homology_root_maps)=

**Maps to the roots are enough for blocks.** Homology blocks don't depend on *which* ancestor gives
the homology, only on *where* positions map to. As each position traces up to exactly one root
position, the blocks between `u` and `v` are where their maps to the roots reach the same root
position. These maps are what the most aggressive simplification (removing every node between the
genomes and their roots) would give, and in a forward simulation they can be kept up to date at
each birth, by composing the parents' maps through the child's iedges. But they find homology of
every age, so queries that report the TMRCA need the coalescence nodes as well ({issue}`210`).

## What giglib would provide

The line to draw is between **policy** and **mechanism**. Policy is any choice that depends on the
biology being modelled: whether crossovers need long chains, the values of `L` and `g`, the rates
of allelic and unequal crossovers and of gene conversion, and how to choose among equally good
pairings of copies. These are options of a simulator. Mechanism is anything that answers a question
about the graph itself, whatever the policy, and that many simulators would need. It can be
provided by _giglib_, as long as it stays reference-free:

- **The homology between two genomes:** a list of blocks, each with its intervals in `u` and `v`,
  its chromosomes, its {class}`~giglib.PositionMap`, and the MRCA (node and time) of each of its
  sub-intervals. This is largely {meth}`~giglib.Tables.find_homology`, with adjacent regions
  that share a map merged (keeping their MRCAs).
- **The homologous positions of a single position:** given a position in `u`, its homologous
  positions in `v` (with their orientations), found by tracing up from both genomes until they
  meet. This is what rejection sampling of breakpoints ({issue}`102`) needs, and is much cheaper
  than finding all the homology between two genomes.
- **Chaining:** joining blocks into chains, given a maximum gap `g` (a parameter, not a policy), and
  the chain containing a given block. These are generic operations on homology, so belong with it.
- **Caches and indexes that make these fast:** e.g. the homology table ({issue}`140`), which maps
  genomes directly to an ancestor, and indexes of the iedges above and below each node (some of
  which already exist). These would be optional, and must not change the answers, only how quickly
  they are found.

At first, these could be developed in `giglib.tools` (e.g. alongside the simulators of
`giglib.tools.simulation`), and moved into the core once they have settled, and once it is clear
which of them are needed for efficiency (e.g. caches that need to be kept up to date as tables
change are better maintained by the tables themselves).

## Open questions

- **The limits of homology by descent:** copies with similar sequences but no recorded shared
  ancestry (e.g. independent insertions of similar sequence) are invisible to it, and a long chain
  can hide substantial divergence if its gaps add up. Should rules also use the total size of the
  gaps, or a model of sequence divergence?
- **Chains across chromosomes,** after fusions or fissions: should blocks on different chromosome
  pairs be linked if they are adjacent in one genome?
- **Speed:** finding all the blocks between two parents at every meiosis needs a full MRCA search.
  Rejection sampling ({issue}`102`) could instead pick a position in one parent, find its
  homologous positions in the other, and check only the chain around it. The homology table
  ({issue}`140`) would make each search cheaper.
- **Flat or grouped homology records:** {meth}`~giglib.Tables.find_homology` returns one record
  for each pair of homologous segments. If a segment of a common ancestor has `n` copies in each
  genome that share it as their MRCA (e.g. a tandem array that expanded independently in both
  lineages), that is `n²` records, rather than the `2n` segments of a grouped form (the ancestral
  segment with its copies in each genome), which the tracing builds internally anyway. This
  seems unlikely to matter at the numbers of copies in current simulations (tens), but should be
  measured (e.g. in the evaluation cases of {issue}`221`). If needed, a grouped output could be
  added alongside the flat one, as each record keeps its `mrca_segment`. (In mode `"all"`,
  grouping would save less, as pairs of copies can have different MRCAs.)

Open questions about how simulators use homology are in {ref}`sec_design_recombination`.
