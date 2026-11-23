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

# Homology and recombination

**Status: Proposed.** Chains are a *heuristic*: a testable starting point for restricting
recombination in plausible ways, not a definition of which homology is allelic, and the rules for
recombination here are simulator approximations, to be tested ({issue}`202`). Related: {issue}`121` (transposable element
simulations), {issue}`83` (stopping "weird" crossovers), {issue}`186` (gene conversion versus
crossovers), {issue}`140` (the homology table), {issue}`102` and {issue}`86` (faster breakpoints),
{ref}`sec_design_simplification` (duplication nodes). Testing: {issue}`202` (simulations with
microsatellites and a centromere, before TE simulations).

## Summary

- **Homology needs no reference.** Two positions, in two genomes, are *homologous* if they trace
  back to the same position in a common ancestor. A GIG records exactly this, so homology is
  defined without any reference genome or shared coordinate system, as _giglib_ requires (genomes
  may differ greatly in length, in the number of copies of a region, and even in their number of
  chromosomes).
- **But homologous does not mean "can cross over".** Copies of a transposable element (TE), or the
  two copies of a duplicated region, are homologous too. A crossover between them is non-allelic
  (ectopic) recombination, which creates rearranged chromosomes, so is rare. Gene conversion
  between them is more common.
- **Allelic homology is long, and colinear.** Two positions at the same place on homologous
  chromosomes are surrounded by more homologous material, in the same order and orientation. Two
  copies of a TE at different places are not: their flanking DNA is unrelated. This makes long,
  colinear homology a good proxy for allelic homology, without a reference (though not a perfect
  one: e.g. long segmental duplications are long and colinear too).
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
{meth}`~giglib.Tables.find_mrca_regions` finds all the regions of `u` and `v` that are homologous
in this way, and where they are in each genome.

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
MRCA regions found by {meth}`~giglib.Tables.find_mrca_regions` are already grouped by map, so
blocks can be found by merging adjacent regions with the same map. As a block can span several
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
position in `v`. Homology appears as line segments, one for each block:

```{code-cell}
:tags: [remove-input]

import matplotlib.pyplot as plt

# Homology blocks between genomes u and v: (u_start, u_end, v_start), all in the same
# orientation. The allelic blocks are separated by small shifts: v has a TE insertion
# at 10-15, u has 3 extra microsatellite repeats at 30-32, v has 2 extra at 58-59, and
# u has a TE insertion at 80-85.
allelic = [(0, 10, 0), (10, 30, 16), (33, 55, 36), (55, 80, 60), (86, 100, 85)]
# The two TE copies are homologous to each other, but not allelic
paralogous = [(80, 86, 10)]

fig, ax = plt.subplots(figsize=(6, 6))
for i, (a, b, c) in enumerate(allelic):
    ax.plot([a, b], [c, c + b - a], color="tab:blue", lw=4, solid_capstyle="butt",
            label="homology blocks in one chain" if i == 0 else None)
for (a, b, c), (d, e, f) in zip(allelic, allelic[1:]):
    ax.plot([b, d], [c + b - a, f], color="tab:blue", lw=1, ls=":",
            label="gaps of at most g between them" if a == 0 else None)
for a, b, c in paralogous:
    ax.plot([a, b], [c, c + b - a], color="tab:red", lw=4, solid_capstyle="butt",
            label="TE copies: homologous, not allelic")
ax.annotate("TE in v", (10, 13), xytext=(22, 6), arrowprops={"arrowstyle": "->"})
ax.annotate("extra repeats in u", (31.5, 37), xytext=(36, 26), arrowprops={"arrowstyle": "->"})
ax.annotate("extra repeats in v", (55, 58.5), xytext=(58, 48), arrowprops={"arrowstyle": "->"})
ax.annotate("TE in u", (83, 85), xytext=(85, 72), arrowprops={"arrowstyle": "->"})
ax.set_xlabel("position in genome u")
ax.set_ylabel("position in genome v")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect("equal")
ax.legend(loc="upper left")
plt.show()
```

In this example, `u` and `v` are two homologous chromosomes, and the maximum gap `g` is at least
as long as the TE (6 positions):

- **Allelic homology** runs along the diagonal, from bottom left to top right, but is broken into
  five blocks (blue). Each break is a small shift: a horizontal step where `u` has material that
  `v` lacks (extra repeats, or a TE inserted in `u`), or a vertical step where `v` has material
  that `u` lacks. Joining the blocks across these small steps gives one long **chain**, covering
  almost the whole chromosome.
- **The two TE copies** (one inserted in `u`, one in `v`, from the same family) are homologous to
  each other, so they appear as a short segment (red), far from the diagonal. It can't join the
  chain: reaching it from the blue block before it in `u` would mean going *backwards* in `v`, and
  reaching the blue block after it from the red one would need a jump far larger than `g`. So it
  is a chain on its own, only as long as the TE.

Each block is a point of the chain where a crossover could happen, but the chain's length is what
suggests that the homology is allelic. A crossover is then allowed well inside a long chain (blue),
but not in a short one (red), however many shifts the long chain has. In the same way:

- **Different chromosomes, or chromosome arms,** give homology on a different pair of chromosomes,
  or far off the diagonal, so it can't join an allelic chain.
- **An inversion** (between `u` and `v`) gives a block in the opposite orientation, a segment
  running from top left to bottom right. It breaks the chain. But breaking the chain doesn't, on
  its own, suppress crossovers within the inversion: if the inversion is longer than `L`, it is a
  long (reversed) chain of its own. So suppressing crossovers within heterozygous inversions (as
  happens in practice, where they would give unbalanced gametes) needs an explicit rule: see
  {ref}`sec_design_homology_rules`.
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
`u`, but 2 in `v`:

```{code-cell}
:tags: [remove-input]

import matplotlib.pyplot as plt

# u = A (0-30), B1 (30-40), B2 (40-50), B3 (50-60), C (60-100)
# v = A (0-30), B1 (30-40), B2 (40-50), C (50-90)
u_copies = [30, 40, 50]
v_copies = [30, 40]
chain = [(0, 30, 0), (30, 40, 30), (40, 50, 40), (60, 100, 50)]  # (u_start, u_end, v_start)

fig, ax = plt.subplots(figsize=(6, 6))
for i, a in enumerate(u_copies):
    for j, c in enumerate(v_copies):
        if (a, a + 10, c) not in chain:
            ax.plot([a, a + 10], [c, c + 10], color="tab:orange", lw=4, solid_capstyle="butt",
                    label="other pairings of copies" if (i, j) == (0, 1) else None)
for i, (a, b, c) in enumerate(chain):
    ax.plot([a, b], [c, c + b - a], color="tab:blue", lw=4, solid_capstyle="butt",
            label="the chain with the most homology" if i == 0 else None)
ax.plot([50, 60], [50, 50], color="tab:blue", lw=1, ls=":", label="a gap (the extra copy in u)")
for y in (30, 40, 50):
    ax.axhline(y, color="0.85", lw=0.5, zorder=0)
for x in (30, 40, 50, 60):
    ax.axvline(x, color="0.85", lw=0.5, zorder=0)
ax.set_xlabel("position in genome u (A, B1, B2, B3, C)")
ax.set_ylabel("position in genome v (A, B1, B2, C)")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect("equal")
ax.legend(loc="upper left")
plt.show()
```

- **The allelic chain is ambiguous within the CNV.** Here, the chain with the most homology pairs
  `u`'s first two copies with `v`'s two copies, and treats `u`'s third copy as a gap (if it is no
  longer than `g`). But pairing `u`'s last two copies with `v`'s, with the gap at the start, has
  just as much homology. This is the usual ambiguity of aligning CNVs, and is real: nothing in
  either genome says which copies are "the same".
- **Unequal crossovers change copy number.** A crossover between copies that are *not* paired in
  the chosen chain (orange) is unequal crossing over, and gives one product with more copies and
  one with fewer. This is how CNVs grow and shrink, by non-allelic homologous recombination, so a
  simulator should be able to allow it, e.g. at a rate relative to allelic crossovers, between
  copies that lie close to the allelic chain (parallel segments near the diagonal).
- **Larger CNVs break the chain.** If the extra copies are longer than `g`, the chain breaks at the
  CNV, so crossovers are suppressed near it, apart from any unequal crossovers that the simulator
  allows.

So CNVs need no special representation: they are copies made by iedges, and their homology shows
up as parallel blocks. What a simulator needs to decide is how to choose among the equally good
pairings of copies for allelic crossovers, and at what rate to allow unequal ones.

(sec_design_homology_rules)=

## Rules for recombination

These would be options of a simulator, not part of _giglib_:

- **Crossovers** are placed only in chains at least `L` positions long, and at least some margin
  from the ends of the chain (as a crossover needs homology either side of it). Within such a
  chain, the crossover position in `u` gives the position in `v` through the block that contains
  it. Positions within the gaps of a chain can't carry a crossover, as they have no homologous
  position in the other genome.
- **Crossovers need the same orientation:** a chain in the opposite orientation to the main
  allelic chain (e.g. a heterozygous inversion) is excluded, however long it is, so crossovers
  within heterozygous inversions are suppressed (the existing simulators already skip reversed
  homology). This is an approximation, not a consequence of the chains: a long reversed chain is
  still long homology. Modelling the crossovers that do happen within inversion loops, and their unbalanced
  products, is an open question.
- **Gene conversion** copies a short tract from one homologous region into another, so only needs
  a block at least as long as the tract. It can use any block, including paralogous ones (e.g.
  between TE copies, or within the repeats of a centromere), which is non-allelic gene conversion.
- **Rates** could depend on the length of a chain (e.g. non-allelic recombination between long
  segmental duplications), and on the TMRCA at the position concerned (as a proxy for divergence),
  rather than using fixed thresholds.

For example, a TE simulation ({issue}`121`) could use a chain length `L` much longer than a TE, a
gap `g` at least as long as a TE (so that a TE inserted in only one of the two genomes doesn't
break the allelic chain), and allow gene conversion between TE copies. Crossovers would then
happen only between allelic positions, without the simulator needing to record which material is
part of a TE.

(sec_design_homology_annotations)=

## Annotations inherited from ancestors (e.g. centromeres)

Some rules apply to *regions*: e.g. crossovers are suppressed within a centromere, but gene
conversion is allowed there ({issue}`186`). Without a reference, a simulator needs a way to know
whether a position of a genome is in such a region, even after many generations of insertions,
deletions and copying have changed its coordinates.

The answer is to define the region on **ancestors**, and let descendants inherit it through
homology, as they inherit alleles. In a simulation, the natural ancestors are the founder genomes
(the roots): e.g. a position is *centromeric* if it traces back to a centromeric position in a
root. This stays reference-free: the annotation is attached to genomes in the graph, not to an
external reference, and different founders could even have different annotations. It also answers
the question in {issue}`186` of whether suppression of recombination is inherited: it is. This is
a modelling assumption, suited to features that follow the sequence: a simulator could also
suppress recombination using masks or rates that depend on the current genotype or context
(e.g. sex, or a centromere that moves).

- **Copied material** inherits the annotation of its source (e.g. a new copy of a centromeric repeat
  unit is centromeric).
- **New material** (inserted positions with no parent, see {ref}`sec_design_insertions`) has no
  ancestor to inherit from, so gets a default, or an annotation given by the simulator when it
  creates the insertion.
- **Finding the annotation** of a position means tracing it up to a root, which is the same
  tracing as for finding homology, so the same caches (e.g. the homology table) make it fast.
- **Annotations could also be rates** rather than yes or no, e.g. a recombination rate map defined on
  the founders, and carried by every genome descended from them, however rearranged.

An annotation like this is a *policy* input, given by the simulator. Tracing a position to its
ancestors, and looking up what is there, is *mechanism*, which _giglib_ could provide.

## Some cases

| Case | Homology between `u` and `v` | Effect |
|---|---|---|
| Microsatellites, centromeric repeats that change length | Many small shifts | Allelic chains stay long; crossovers are unaffected |
| TE copies at different places | Short blocks, far off the diagonal | No crossovers between copies; gene conversion allowed |
| Large segmental duplications | Long paralogous blocks | Non-allelic crossovers possible, if `L` allows them (e.g. at a lower rate) |
| Copy-number variants (e.g. tandem copies) | A grid of parallel blocks; the allelic pairing of copies is ambiguous | Allelic crossovers through one pairing; unequal crossovers (other pairings) change copy number |
| Heterozygous inversion | A reversed block | Breaks the chain; crossovers within it are excluded by requiring the same orientation as the allelic chain |
| Large heterozygous insertion or deletion | A step longer than `g` | Breaks the chain, so fewer crossovers nearby |
| Different chromosome numbers (e.g. after a fusion) | One chromosome's chain continues on two chromosomes of the other | Crossovers within each part of the chain; the rule for linking blocks across chromosomes needs deciding |
| A centromere (or other region where crossovers are suppressed) | Allelic chains through it, as elsewhere | Crossovers suppressed by an annotation inherited from the founders; gene conversion allowed |
| Bacterial genomes of very different sizes | Local blocks, possibly few long chains | Recombination is mostly gene conversion (replacing a homologous tract), so needs only blocks |

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

## What giglib would provide

The line to draw is between **policy** and **mechanism**. Policy is any choice that depends on the
biology being modelled: whether crossovers need long chains, the values of `L` and `g`, the rates
of allelic and unequal crossovers and of gene conversion, and how to choose among equally good
pairings of copies. These are options of a simulator. Mechanism is anything that answers a question
about the graph itself, whatever the policy, and that many simulators would need. It can be
provided by _giglib_, as long as it stays reference-free:

- **The homology between two genomes:** a list of blocks, each with its intervals in `u` and `v`,
  its chromosomes, its {class}`~giglib.PositionMap`, and the MRCA (node and time) of each of its
  sub-intervals. This is largely {meth}`~giglib.Tables.find_mrca_regions`, with adjacent regions
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
- **Separate models for each kind of event:** the mechanism could yield candidate pairs of
  homologous regions, from which a simulator chooses crossovers, gene conversions or non-allelic
  events, each with its own rules for tracts and products.

- **Thresholds or rates:** fixed `L` and `g`, or rates that depend smoothly on chain length, gap
  sizes and TMRCA? What are sensible defaults?
- **The ends of chains:** how much homology is needed either side of a crossover?
- **Choosing among overlapping chains** near tandem duplications and CNVs: is "most homology" the
  right choice, should one of the equally good pairings be chosen at random, and does it need a
  dynamic programming in practice?
- **Unequal crossovers:** at what rate, and between which copies (e.g. only those within some
  distance of the allelic chain)?
- **Chains across chromosomes,** after fusions or fissions: should blocks on different chromosome
  pairs be linked if they are adjacent in one genome?
- **Inversions:** suppress crossovers within heterozygous inversions (as now), or model them?
- **Speed:** finding all the blocks between two parents at every meiosis needs a full MRCA search.
  Rejection sampling ({issue}`102`) could instead pick a position in one parent, find its
  homologous positions in the other, and check only the chain around it. The homology table
  ({issue}`140`) would make each search cheaper.
