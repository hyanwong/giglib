(sec_design_recombination)=

# Recombination

**Status: Proposed.** How simulators can use homology (see the {ref}`homology design note <sec_design_homology>`) to restrict
recombination in plausible ways. The rules here are simulator approximations, to be tested
({issue}`202`), and are *policy* for each simulator to choose, built on the *mechanism* that
_giglib_ provides. Related: {issue}`121` (transposable element simulations), {issue}`83` (stopping
"weird" crossovers), {issue}`186` (gene conversion versus crossovers), {issue}`102` and {issue}`86`
(faster breakpoints).

## Summary

- **Homologous does not mean "can cross over".** Copies of a transposable element (TE), or the
  two copies of a duplicated region, are homologous too. A crossover between them is non-allelic
  (ectopic) recombination, which creates rearranged chromosomes, so is rare. Gene conversion
  between them is more common.
- **Allelic homology is long, and colinear.** Two positions at the same place on homologous
  chromosomes are surrounded by more homologous material, in the same order and orientation. Two
  copies of a TE at different places are not: their flanking DNA is unrelated. This makes long,
  colinear homology a good proxy for allelic homology, without a reference (though not a perfect
  one: e.g. long segmental duplications are long and colinear too).
- **Long, colinear homology is found by chaining** homology blocks across small gaps (see
  {ref}`the homology design note <sec_design_homology>`), so rules for crossovers can be based on chains.
- **Policy belongs to simulators, mechanism can be in giglib.** The *rules* for recombination
  (where crossovers may happen, thresholds and rates) are choices for each simulator, not part of
  _giglib_'s model.

(sec_design_recombination_rules)=

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

(sec_design_recombination_annotations)=

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

## Open questions

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
- **Inversions:** suppress crossovers within heterozygous inversions (as now), or model them?
