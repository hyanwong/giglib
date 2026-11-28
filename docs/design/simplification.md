(sec_design_simplification)=

# Simplification

**Status: Implemented** ({pr}`241`). {meth}`~giglib.Tables.simplify` keeps the branch points of
the position tree, region by region, with the algorithm below, and the ends of lineages. Options
to keep more nodes (by flags, and non-coalescent regions) are proposed in {issue}`244`. The
measurements are in {pr}`247` (see {ref}`sec_design_simplification_evaluation`).
Related: {ref}`sec_design_homology_table` (a cache for locating homologous positions, built by
the same core), {ref}`sec_design_mutations` (which requires simplification to keep duplication nodes),
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
  gives both simplification, and the more aggressive homology table
  ({ref}`sec_design_homology_table`), a cache for finding equivalent positions in different
  genomes (e.g. to place recombination breakpoints).
- **Simplification keeps the branch points of the position tree.** Each position of a genome has
  a single parent position, so the ancestry of the samples' positions is a tree. Simplification
  keeps the nodes (in the regions) where lineages of sample positions merge, whether from
  different children (a split) or through two iedges of the same child (a copy), and bypasses
  them elsewhere. So {meth}`~giglib.Tables.find_homology` gives the same results after
  simplifying, which is the main test of correctness.
- **Coalescence and duplication are distinguished.** Both are places where copies of an
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

(sec_design_simplification_branch_points)=

## What simplification keeps: the branch points of the position tree

Each position of a genome inherits from a single position of a single parent, so following a
position back in time gives a single lineage. The lineages of all the samples' positions therefore
form a tree, the *position tree* (or a forest, if there are several roots), whose leaves are the
positions of the samples (see {issue}`111`). A **branch point** of this tree is a position of an
ancestral node where two or more lineages of sample positions merge, either

- from **different children** (a split, as at a coalescence in a tree sequence), or
- through **two iedges of the same child**, both inheriting that position (a copy: the parent of a
  tandem duplication, for example).

**Simplification keeps exactly the branch points**, together with the samples. Like _tskit_'s
`simplify()`, it works region by region: a node is kept in the regions where it is a branch point,
and bypassed (by composing iedges) where it is not. A node that is a branch point nowhere is
removed. The ends of lineages that never merge are also kept (as with _tskit_'s
`keep_input_roots=True`): a GIG has no separate record of a genome's length, which is known
only from its iedges and mutations, so without an iedge for each sample position a sample with
missing data at the end of its genome would become shorter.

This definition gives the main test of correctness. The MRCA of any two homologous positions, as
reported by {meth}`~giglib.Tables.find_homology`, is where their lineages merge, so it is a branch
point. So after simplifying, {meth}`~giglib.Tables.find_homology` must give the same results, in
every mode (after relabelling the nodes). And the haplotypes of the samples must be unchanged, once
mutations are moved (see {ref}`sec_design_simplification_mutations`).

**The branch point of a duplication is not the "duplication node" below.** For a tandem
duplication, the copies merge in the *parent* (both copies inherit the same positions of it), so the
parent is the branch point, and is kept. The child, which is the oldest genome containing both
copies (the duplication node, below), is not a branch point: removing it loses only *when* (in
which genome) the duplication happened, not any homology. This agrees with the mutations design,
which asks for "a node with a duplication beneath it" to be kept ({ref}`sec_design_mutations`).
So keeping duplication nodes is an option, for when the timing of duplications matters, e.g. for
nodes that a simulator has flagged as duplications.

## One core, different rules for which nodes to keep

| | Keeps | Removes |
|---|---|---|
| **Simplification** | samples, and the branch points of the position tree (where lineages split, and where they are copied) | everything else, e.g. unary nodes |
| **Homology table** ({ref}`sec_design_homology_table`) | samples, the branch points where lineages are copied, and duplication nodes | unary nodes *and* ordinary coalescence nodes |

So the homology table is a more aggressive simplification, kept alongside the real iedges. This
suggests a single core routine that composes iedges, taking a rule for which nodes (or regions of
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

So both simplification and the homology table keep the branch points where copies merge (see
{ref}`sec_design_simplification_branch_points`). Keeping the duplication nodes themselves, which
record when the duplications happened, is an option for simplification, but the homology table
needs them, to tell the copies apart (see {ref}`sec_design_homology_table`).

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

## The algorithm

A generalisation of _tskit_'s simplify algorithm (Kelleher et al. 2018), which passes segments of
"ancestral material" up the graph from the samples, using maps rather than shared coordinates:

1. Visit the nodes from the youngest to the oldest. Each node has a list of segments of ancestral
   material, passed up from its children: an interval of the node's genome, the output node that
   it comes from, and the {class}`~giglib.PositionMap` from the node's coordinates to that output
   node's. A sample starts with a segment covering its whole genome, mapping to itself.
2. Cut the node's genome into pieces covered by the same segments, with a single sweep (as
   {meth}`~giglib.Tables.find_homology` does, with `giglib.intervals.pieces`). Where a piece is
   covered by two or more segments, the node is a branch point: it is kept (as an output node),
   with an output iedge from each segment's output node to it, and passes up one segment, mapping
   to itself. Where a piece is covered by one segment, the node is bypassed there, and that
   segment is passed up, its map composed with each iedge above.
3. Move each mutation above a bypassed region down to the output node of the segment covering
   it, at the mapped position (complementing its derived state if the map is reversed). Remove the
   mutations above regions with no ancestral material.
4. Merge output iedges that abut ({meth}`~giglib.tables.IEdgeTable.squash`), renumber the nodes,
   and return the map from the old node IDs to the new ones.

## Granularity: whole nodes or regions of nodes

_tskit_'s `simplify()` keeps a node if it is a coalescence *somewhere*, but bypasses it in regions
where it is unary. The alternative is to keep or remove whole nodes. Bypassing unary regions
gives fewer, longer iedges, but more composing, and more splitting of iedges at the boundaries
of regions. Which is faster to compute, and gives smaller tables, needs measuring. The definition
above (keeping branch points) is region by region, so that is the default; keeping whole nodes
could be an option.

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


- **Time and peak memory** of simplifying, against population size, number of generations,
  genome length, and the rates of recombination and of each kind of structural variant.
- **Size of the output** (numbers of nodes and iedges), for whole-node versus region
  granularity.
- **Cost of finding duplication nodes:** detecting the easy case only, tracking pairs of copies,
  and using flags set by the simulator.
- **How often to simplify** in a forward simulation (every generation, or every *k*
  generations), as in _tskit_'s forward simulation tutorial.

### Checks of correctness

These belong in the unit tests (or slower validation scripts), not in the performance scripts:

- {meth}`~giglib.Tables.find_homology` gives the same results after simplifying, in every mode
  (after relabelling the nodes), on graphs with structural variants (e.g. from the simulators of
  tandem duplications, microsatellites, and genomes from a single base pair).
- Haplotypes of the remaining samples are unchanged by simplification, including missing data,
  except that a sample whose genome was only known to be longer from descendants that are removed
  becomes shorter (losing only missing data: a GIG has no record of genome lengths, see
  {issue}`158`).
- Simplifying twice gives the same as simplifying once.
- With no structural variants, the result matches _tskit_'s `simplify(keep_input_roots=True)`
  (after converting with `to_tree_sequence()`).
- If duplication nodes are kept, those detected match those recorded by the simulator.

(sec_design_simplification_discussion)=

## Discussion so far

Recorded so that the work can be picked up again later. The discussion happened in a
conversation about the giglib API, mutations, and storage (November 2026), and in {issue}`140`.

**Agreed, or leaned towards:**

- Simplification and the homology table ({ref}`sec_design_homology_table`) are bound together,
  as both compose iedges, with different rules for which nodes to keep.
- Coalescence and duplication nodes should be distinguished. The two main reasons are knowing
  which copy of a duplicated region is which, and that only coalescences inform demography
  (raised by Yan Wong).
- The definition of a duplication node above ("the oldest node whose genome contains both
  copies, at different positions") was proposed so that duplications created by unequal
  crossing-over are handled. It has not been challenged, but has not been formally agreed.
- Simplification keeps the branch points of the position tree, region by region, so that
  {meth}`~giglib.Tables.find_homology` gives the same results afterwards (proposed in November
  2026, after the homology work of {issue}`111`; see
  {ref}`sec_design_simplification_branch_points`). For a duplication, the branch point is the
  node with the duplication beneath it, as the mutations design says. Keeping the duplication
  nodes themselves (which record when duplications happened) is an option.
- Many choices should be **decided by efficiency**, measured with a forward simulator (see
  {ref}`sec_design_simplification_evaluation`), rather than settled in advance.
- The multiple-counting question (duplicated regions sharing coalescences) is theory rather than
  design, and needs a place of its own.

**Not yet discussed in detail:**

- Whether duplication nodes, if kept, should be flagged (e.g. by a simulator), detected, or
  both.

**Done since:** simplification, with the checks of correctness ({pr}`241`, which also keeps the
ends of lineages, so that samples don't lose missing data at the ends of their genomes), and
evaluation cases showing that it makes finding homology 2.5 to 40 times faster ({pr}`247`).
**Next:** the options in {issue}`244`, and the homology table, if measurements show it is needed.

## Open questions

- Should keeping whole nodes (rather than regions) be an option, if measurements favour it?
- Should keeping duplication nodes be an option, and if so, how are they found (flagged by a
  simulator, or detected)? {issue}`244` proposes keeping nodes with given flags.
- Should simplification optionally keep non-coalescent regions of part-coalescent nodes, or
  other nodes (e.g. those with mutations), as _tskit_'s `simplify()` can ({issue}`244`)?
- Should `sample_resolve()` become an option of `simplify()`?
