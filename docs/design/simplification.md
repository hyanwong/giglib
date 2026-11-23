(sec_design_simplification)=

# Simplification

**Status: Proposed.** Nothing here is implemented yet, apart from
{meth}`~giglib.Tables.sample_resolve`, which does not remove nodes. Several choices are left open,
to be decided by performance measurements on forward simulations
(see {ref}`sec_design_simplification_evaluation`).
Related: {issue}`140` (a "homology table" for locating recombination breakpoints),
{ref}`sec_design_mutations` (which requires simplification to keep duplication nodes),
{ref}`sec_design_storage` (cached indexes), {issue}`158` (reference lengths).

This note also records **where the discussion has reached** (see
{ref}`sec_design_simplification_discussion`), so that work can resume from it.

## Summary

- **Simplification removes nodes.** {meth}`~giglib.Tables.sample_resolve` only removes the
  parts of iedges that don't lead to samples. Simplification would also remove nodes that carry
  no information about the genealogy of the samples, such as unary nodes, as in _tskit_'s
  `simplify()`.
- **Removing a node means composing iedges.** An iedge maps an interval of a child's genome to
  an interval of its parent's genome, by a shift, optionally reversed. Two such maps in a row
  give another one, so the iedges through a removed node can be replaced by iedges that skip
  it.
- **One core, several uses.** The same operation, with different rules for which nodes to keep,
  gives both simplification, and the more aggressive "homology table" proposed in
  {issue}`140`. The homology table is a cache used for finding equivalent positions in different
  genomes (e.g. to place recombination breakpoints).
- **Coalescence and duplication nodes are distinguished.** Both are places where copies of an
  ancestral region split. But only coalescences tell us about demography, and duplications must
  be kept to know which copy of a duplicated region is which.
- **Many choices depend on speed and memory,** and should be settled by measurements on forward
  simulations with structural variants.

## Removing a node: composing iedges

Suppose a child `c` inherits an interval of node `u`, and `u` inherits that interval from its
parent `p`:

```
c ──(iedge A: c's [a, b) -> u's [a', b'))──> u ──(iedge B: u's [a', b') -> p's [a", b"))──> p
```

Each iedge maps positions by a shift, reversed if the iedge is an inversion. Composing the two
gives a single map from `c`'s `[a, b)` to `p`'s `[a", b")`, which is an inversion if exactly one
of A and B is. (With {class}`~giglib.PositionMap`, this is
`PositionMap.from_iedge(A).then(PositionMap.from_iedge(B))`.) So `u` can be removed from this path by replacing A and B with one iedge from
`c` to `p`. In general, the parent interval of A overlaps several iedges above `u`, so A is
split at their boundaries, giving one composed iedge per piece. Chromosomes compose in the same
way: the composed iedge goes from A's child chromosome to B's parent chromosome.

This is exact: the haplotypes of the remaining nodes are unchanged, as long as any mutations
above `u` are moved down (see {ref}`sec_design_simplification_mutations`).

Composing leaves many pieces that abut in both the child and the parent (e.g. where `u` had
several iedges above it, all from the same parent, or where both pieces of a split come from
the same ancestor). These can be merged with {meth}`~giglib.tables.IEdgeTable.squash`, as
_tskit_'s `simplify()` does with edges.

## One core, different rules for which nodes to keep

| | Keeps | Removes |
|---|---|---|
| **Simplification** | samples, coalescence nodes, duplication nodes | everything else, e.g. unary nodes |
| **Homology table** ({issue}`140`) | samples, duplication nodes, and something at the top (see below) | unary nodes *and* ordinary coalescence nodes |

So the homology table is a more aggressive simplification, kept alongside the real iedges. It
skips the intermediate ancestors that make tracing between two genomes slow, while keeping the
duplication nodes needed to know which copy of a duplicated region is which. This suggests
writing a single core routine that composes iedges, taking a rule for which nodes (or regions of
nodes) to keep, and building both on top of it.

## Coalescence and duplication

Take a region of an ancestral genome, and follow its copies down to the samples. Any two copies
are either in **different** sample genomes, or in the **same** sample genome, at different
positions (paralogues). Going back in time, the lineages of the two copies eventually merge.

### Defining a duplication node

A tempting rule is to classify each merge by the iedges that the copies pass through: through
two iedges of the *same* child into one parent would be a duplication, and through *different*
children a coalescence. This fails for duplications created by **unequal crossing-over**
(non-allelic homologous recombination, NAHR). There, a child inherits one copy from each parent,
and the copies only merge further back in time, through different children, which looks like a
coalescence. A definition that works in both cases is:

> The **duplication node** of two copies is the oldest node whose genome contains both copies,
> at different positions.

- **Tandem duplication within one parent:** the child contains both copies, and its parent only
  one, so the child is the duplication node.
- **Unequal crossing-over:** the child contains both copies, and each of its parents only one, so
  again the child is the duplication node.

What happens to the two lineages above the duplication node depends on how the duplication
arose:

- **Tandem duplication:** the copies merge immediately, in the parent, as both are copies of one
  interval of the parent's genome. This merge *is* the duplication event: it is not the
  coalescence of two independently inherited lineages, so it carries no information about
  demography.
- **Unequal crossing-over:** the copies come from different parents, and merge further back
  in time, where the two (allelic) lineages coalesce. That merge is an ordinary coalescence.

Whether a split in the genealogy of a region is a "speciation" (orthology) or a "duplication"
(paralogy) depends on whether a duplication node lies between it and the copies: the same
distinction as in gene-tree / species-tree reconciliation.

### Why distinguish them

- **Demography.** Looking backwards in time, only coalescences carry information about
  demography (e.g. population sizes). Duplication nodes mark mutational events in a genome's
  structure.
- **Which copy is which.** Below a duplication node, the copies are at different positions in
  that node's genome, so the node records the identity of each paralogue. Without it, two copies
  in a sample might trace back to the same ancestral region, with no record of which is which
  (see point 2 in the first comment on {issue}`140`).
- **Timing.** Strictly, duplication nodes are not needed to represent the genealogy. If a
  duplication node is unary, removing it just leaves a descendant with two iedges that overlap
  in one ancestor. But that loses when (and in which genome) the duplication happened.

So both simplification and the homology table should keep duplication nodes. This is also what
the mutations design assumes ({ref}`sec_design_mutations`).

### Finding duplication nodes

For a tandem duplication, the duplication node is easy to find: it is a child with two iedges
whose parent intervals overlap in the same parent. For unequal crossing-over, it can only be
found by tracking pairs of copies up the graph, to the oldest genome that contains both: the
expensive part. Options:

1. **Only detect the easy case** (overlapping iedges from one parent) in a first version.
2. **Track pairs of copies** during simplification, when ancestral material is passed up the
   graph anyway.
3. **Record duplication nodes when they are created,** e.g. a simulator flags a node as a
   duplication node (a new node flag, such as `NODE_IS_DUPLICATION`, or a column), so they don't
   need detecting at all. Simplification would then keep flagged nodes, and could check or
   update the flags.

Option 3 is cheapest, but only works for GIGs whose creator knows where duplications happened.

## Granularity: whole nodes or regions of nodes

_tskit_'s `simplify()` keeps a node if it is a coalescence *somewhere*, but bypasses it in regions
where it is unary. The alternative is to keep or remove whole nodes. Bypassing unary regions
gives fewer, longer iedges, but more composing, and more splitting of iedges at the boundaries
of regions. Which is faster to compute, and gives smaller tables, needs measuring.

(sec_design_simplification_mutations)=

## Mutations

Mutations are above nodes, not on iedges. If simplification only removes nodes (or regions of
nodes) that are unary, a mutation above a removed region of a node `u` has exactly one descendant
copy, in the nearest descendant `d` that is kept. So it can move to the node `d`, at the
corresponding position. This is found by tracing the *original* iedges from `d` up to `u`: one
iedge if `d` is a child of `u`, or several if a chain of unary nodes is removed. The composed
iedge cannot be used for this, as it maps `d`'s coordinates to those of the node above `u`, not
of `u` itself. The mutation's derived state is complemented if that path includes an odd number
of inversions. Its time stays valid: it was between the times of `u` and of `u`'s parent, `d` is
younger than `u`, and after composing iedges, `d`'s parent in that region is `u`'s parent (or an
older kept ancestor). A region with several copies is a coalescence or a duplication, so its node
is kept, and its mutations stay above it.

The homology table does not affect mutations, which stay above their nodes in the real graph.
The time of a mutation is only constrained by the real iedges (it must be between the times of its
node and of that node's parent), not by the homology table's iedges, which skip intermediate
ancestors, and so would allow a much wider range of times.

## The homology table

Proposed in {issue}`140`. It would be used by a "homologous location" method, e.g.
`gig.homologous_location(source_node, source_position, target_node)`, returning the equivalent
position (and orientation) in the target genome. It could also speed up other methods that trace
positions up the graph, such as {meth}`~giglib.Graph.ancestral_allele_at` (which only needs to
reach the root of a position), and {meth}`~giglib.Graph.allele_at` (although that also needs the
mutations on the skipped nodes). Open questions:

- **Several equivalent positions.** A position can have more than one equivalent in the same
  target genome. For example, after a tandem duplication of a parent's positions 10 to 20 to make
  a child's positions 10 to 20 and 20 to 30, position 15 of the parent is equivalent to both 15
  and 25 in the child. Or a position may have none (e.g. after a deletion). So the method must
  either return *all* the equivalent positions (each with a chromosome and orientation, possibly
  none), take a way of choosing between them (e.g. to choose a recombination breakpoint at
  random), or report that the answer is ambiguous. This must be decided before implementing it.
  Returning all of them is the most general, and the others can be built on it.

- **What is at the top?** The deepest roots, a chosen reference genome (which ties in with the
  reference sequences in {ref}`sec_design_mutations`, and with {issue}`158`), or the local MRCA
  of each region?
- **Where does it live?** In memory only, and rebuilt when needed? Or part of the tables, saved
  with them and covered by validity flags? The storage note ({ref}`sec_design_storage`) has the
  same question for the parent index.
- **Keeping it up to date.** In a forward simulation, new iedges are added every generation. Can
  the table be updated incrementally (new nodes are composed onto the existing table), or must it
  be rebuilt when simplifying? At least, it needs a flag saying whether it is stale.
- **Defragmentation.** Over a long simulation, intervals may be split into many pieces that could
  be merged again, e.g. once a deletion is fixed (see the second comment on {issue}`140`). How
  often should this be done, and is it equivalent to choosing a new reference?

## Related questions (out of scope here)

- **Multiple counting.** Duplicated regions can share the same coalescence nodes above their
  duplication node, so naively counting coalescences across the genome (e.g. to infer
  demography) would count those coalescences more than once. This is a question for the theory
  and the statistics computed from a GIG, rather than for the design of the data structure, and
  probably needs a place of its own.

(sec_design_simplification_evaluation)=

## Evaluation

Many of the choices above depend on speed and memory, so should be decided by measurement, using
the `evaluation/` folder at the root of the repository (modelled on
[tsdate's](https://github.com/tskit-dev/tsdate/tree/main/evaluation)): scripts that are run by
hand, are not part of the package or the unit tests, record their results with the giglib
version and commit, and have their own requirements. Simplification would add cases to
`evaluation/performance.py` (see `evaluation/README.md`).

### A forward simulator

The evaluation needs forward simulations with recombination and structural variants: tandem
duplications, deletions, inversions, and unequal crossing-over (to create duplications that are
hard to detect). The Wright-Fisher simulators in `giglib.tools.simulation` are a starting
point, but only include inversions. A simulator could live in `evaluation/` (if only used there), or in
{ref}`giglib.tools <sec_tools>` (if also useful to users). It would record the true duplication
nodes, so that detection methods can be checked against them.

### What to measure

For simplification:

- **Time and peak memory** of simplifying, against population size, number of generations,
  genome length, and the rates of recombination and of each kind of structural variant.
- **Size of the output** (numbers of nodes and iedges), for whole-node versus region
  granularity.
- **Cost of finding duplication nodes:** detecting the easy case only, tracking pairs of copies,
  and using flags set by the simulator.
- **How often to simplify** in a forward simulation (every generation, or every *k*
  generations), as in _tskit_'s forward simulation tutorial.

For the homology table:

- **Time to find homologous locations** (e.g. recombination breakpoints) with and without the
  table, and against the depth of the graph.
- **Cost of maintaining the table**, per generation (incrementally) or per simplification
  (rebuilding), and its size compared with the iedges table.
- **Cost and benefit of defragmenting,** and how often to do it.

### Checks of correctness

These belong in the unit tests (or slower validation scripts), not in the performance scripts:

- Haplotypes of the remaining samples are unchanged by simplification.
- With no structural variants, the result matches _tskit_'s `simplify()` (after converting with
  `to_tree_sequence()`).
- Homologous locations found with the homology table match those found by tracing the full
  iedges.
- Detected duplication nodes match those recorded by the simulator.

(sec_design_simplification_discussion)=

## Discussion so far

Recorded so that the work can be picked up again later. The discussion happened in a
conversation about the giglib API, mutations, and storage (November 2026), and in {issue}`140`.

**Agreed, or leaned towards:**

- Write this design note first, and implement simplification (which removes nodes) after it.
  Simplification and the homology table of {issue}`140` are bound together, as both compose
  iedges, with different rules for which nodes to keep.
- Coalescence and duplication nodes should be distinguished. The two main reasons are knowing
  which copy of a duplicated region is which, and that only coalescences inform demography
  (raised by Yan Wong).
- The definition of a duplication node above ("the oldest node whose genome contains both
  copies, at different positions") was proposed so that duplications created by unequal
  crossing-over are handled. It has not been challenged, but has not been formally agreed.
- Simplification keeps duplication nodes, as the mutations design already says.
- Many choices should be **decided by efficiency**, measured with a forward simulator (see
  {ref}`sec_design_simplification_evaluation`), rather than settled in advance.
- The multiple-counting question (duplicated regions sharing coalescences) is theory rather than
  design, and needs a place of its own.

**Not yet discussed in detail:**

- What is at the top of the homology table, and where it lives.
- Whether duplication nodes should be flagged (e.g. by a simulator), detected, or both.
- The exact algorithm. A natural starting point is _tskit_'s simplify algorithm (Kelleher et al.
  2018), which passes "ancestral material" up the graph from the samples, extended so that each
  piece of material carries a map (shift and orientation) to its position in each descendant
  sample, rather than assuming shared coordinates.

**Suggested next steps:**

1. Settle the open questions that don't need measuring (e.g. the definition of a duplication
   node), and update this note.
2. Add simplification cases to `evaluation/performance.py`.
3. Write a forward simulator with recombination and structural variants, including tandem
   duplications and unequal crossing-over, that records the true duplication nodes.
4. Implement a first simplification, composing iedges and keeping samples, coalescences and
   (easily detected) duplication nodes, with correctness tests (haplotypes unchanged; agreement
   with _tskit_ when there are no structural variants).
5. Measure, then make the choices left open here. Then build the homology table on the same
   core.

**Starting points in the code:**

- {meth}`~giglib.tables.IEdgeTable.squash`: merges adjacent iedges, which simplification will
  need after composing iedges.
- {meth}`~giglib.Tables.sample_resolve` (in `giglib/tables.py`): passes intervals of ancestral
  material up the graph, oldest node last, using the `portion` library to merge intervals. It
  does not remove nodes, but its structure (a stack of nodes ordered by time) is close to what
  simplification needs.
- {meth}`~giglib.Tables.remove_non_ancestral_nodes`: removes nodes, and renumbers the nodes,
  iedges, indexes and mutations. Simplification will need the same renumbering.
- {class}`~giglib.PositionMap`: the map from one genome's coordinates to another's, made by an
  iedge (a shift, reversed for an inversion). All the code that maps positions or intervals
  through iedges uses it. Maps along a chain of iedges combine into one with
  {meth}`~giglib.PositionMap.then`, which is what composing iedges needs.
- {meth}`~giglib.Tables.find_mrca_regions`: traces two genomes up to their MRCAs, which is the
  operation the homology table aims to speed up.
- `Graph._known_alleles()`: traces positions up through iedges, combining their position maps
  as it goes, as composing iedges will.
- {meth}`~giglib.Tables.decapitate`: moves the effect of removed mutations onto remaining nodes,
  which is similar to moving mutations down when removing a node.
- The Wright-Fisher simulators in `giglib.tools.simulation` (with inversions only), and the examples in
  `docs/examples.md`.

## Open questions

- Whole-node or region granularity (decided by measurement)?
- How to find duplication nodes, and should they be flagged?
- Should simplification optionally keep unary nodes, or other nodes (e.g. those with
  mutations), as _tskit_'s `simplify()` can?
- What is at the top of the homology table, and where does it live?
- What should `homologous_location()` return when a position has several (or no) equivalents?
- Should `sample_resolve()` become an option of `simplify()`?
