(sec_design_insertions)=

# Inserted material

**Status: Proposed.** Discussion: the alignment tool in {issue}`139` (and its first PR, {pr}`175`),
which raised the question. Related: {issue}`158` (sequence lengths), {issue}`126` (pseudo-roots),
{ref}`sec_design_mutations`, {ref}`sec_design_simplification`.

## Summary

- **The question:** should every position of a genome that has ancestors (a non-root node) have
  a parent? At the moment, nothing requires it, so a node can have positions with no iedge above
  them. These are effectively "new" material, with no recorded ancestry.
- **Inserted material comes from somewhere.** Insertions are either copies of material from
  elsewhere in the graph (duplications, transpositions, unequal crossing-over), which should be
  represented by an iedge from that source, or genuinely new bases.
- **The origin is often unknown.** In real data, it is often impossible to tell where a small
  insertion came from, or even exactly where it is. So "origin unknown" needs a representation,
  which can be refined if the origin is later identified.
- **Proposal (to be discussed):** represent inserted material of unknown or new origin by a
  **source node** for each insertion event: a root with an iedge into the node that gained the
  insertion, with the inserted bases (if known) as mutations above the source node. Then every
  position of a non-root genome has a parent, and every base traces back to some root.

## Kinds of insertion

1. **Copies of material elsewhere in the graph.** Tandem duplications (including microsatellite
   expansions and polymerase slippage), dispersed duplications, transpositions, and duplications
   created by unequal crossing-over. The inserted bases have a source in an ancestor, so the
   insertion should be an iedge from that source, possibly from a different position, chromosome
   or parent. This records the homology, which is needed e.g. to find recombination breakpoints
   ({issue}`140`) and to tell paralogues apart ({ref}`sec_design_simplification`).
2. **Genuinely new bases.** Untemplated insertions (e.g. at repaired double-strand breaks), and
   foreign DNA (e.g. viral insertions, or horizontal transfer from outside the sampled
   population). These have no ancestor in the graph.
3. **Of unknown origin.** Observed data (e.g. insertions called against a reference, or the
   multi-character alleles of {pr}`175`) usually say *what* was inserted, but not *where it came
   from*. A short insertion could be any of the above: e.g. `AT` inserted into `ATATAT` could be a
   tandem copy of a neighbouring `AT`, a copy from elsewhere, or new.

### Where an insertion is can also be ambiguous

Within a repeat, an insertion (or deletion) has several equivalent positions: inserting `AT`
anywhere in `ATATAT` gives the same sequence. VCF handles this by "left-aligning" indels to a
canonical position. For a GIG, the choice changes which bases are considered homologous, though
not the sequences. Any conversion from data needs a rule (e.g. left-alignment, as in VCF), and
should document it.

## Options for new material, or material of unknown origin

a. **Positions with no parent**, inside a non-root node. This is what nothing currently forbids,
   and what the first version of the alignment tool ({pr}`175`) does.
b. **A source node for each insertion event**: a new root, with an iedge into the node that gained
   the insertion (covering just the inserted positions), and the inserted bases as mutations
   above the source node (no mutation where a base is unknown).
c. **A single shared source node** (e.g. one "new material" node per GIG), with each insertion an
   iedge from a different part of its genome.
d. **For foreign DNA**, a donor node representing the source genome (e.g. a virus), if the donor
   sequence is known. This is like (b), but the source node could be shared between insertions
   from the same donor, recording that they are homologous.

### Consequences

| | (a) no parent | (b) source node per event | (c) one shared source node |
|---|---|---|---|
| Every non-root position has a parent | no | yes | yes |
| Sequence lengths from iedges ({issue}`158`) | wrong if an insertion is at the end of a genome | correct | correct |
| Homology of separate insertions | none | none (separate roots) | implies they are all related, which is misleading |
| Number of extra nodes | 0 | one per insertion event | 1 |
| Refining the origin later | add an iedge | replace the source node's iedge by one from the real source | replace one iedge |

Further considerations:

- **Time of a source node.** The insertion happened on the branch above the node that gained it,
  so the source node could be given the time of that node's parent (or of the insertion, if
  known). It must be older than the node that gained the insertion.
- **Marking source nodes.** They could be identified by a node flag (e.g.
  `NODE_IS_INSERTION_SOURCE`), or by metadata, so that conversions and algorithms can treat them
  specially (e.g. turning them back into insertions when converting to a tree sequence).
- **A validity flag.** If every non-root position must have a parent, this could become a
  validity guarantee (a new `ValidFlags` flag), checked when iedges are added, or when creating a
  `Graph`. A node whose positions all lack parents is a root, so the guarantee would be about
  nodes that have *some* parents.
- **Simplification.** Source nodes are unary roots. Simplification should probably keep them (as
  they record that material is new or of unknown origin), or record the same information in
  another way. See {ref}`sec_design_simplification`.
- **Mutations.** With (b) and (c), the inserted bases are ancestral states of the source node, so
  are found by {meth}`~giglib.Graph.ancestral_allele_at`, like any other ancestral state.
- **Pseudo-roots ({issue}`126`)** are a different idea: a cached ancestral *sequence* at a node,
  to avoid tracing further up the graph. A source node is a real root, for material whose ancestry
  is unknown.

## Conversions

- **From tree sequences with alignment columns** ({pr}`175`): an insertion is a mutation from `-`
  to some bases above a node. With (b), each such mutation would create a source node, with an
  iedge into the node below the mutation, covering the inserted positions.
- **To tree sequences:** with (b) or (c), source nodes would be turned back into insertions:
  their iedges become new alignment columns, with ancestral state `-`, and a mutation above the
  node that gained the insertion.

## Recommendation (for discussion)

Option (b): a source node for each insertion event of unknown or new origin, marked by a node
flag, given the time of the parent of the node that gained the insertion, with the inserted bases
(if known) as mutations above it. Insertions of known origin are iedges from their source. If
agreed, the alignment tool ({pr}`175`) would be changed to create source nodes, and a validity
flag could later require every non-root position to have a parent.

## Open questions

- Is a source node per event right, or should some be shared (e.g. for a known foreign donor)?
- Flag or metadata to mark source nodes, and what time should they have?
- Should "every non-root position has a parent" become a validity requirement, and if so, when is
  it checked?
- What should simplification do with source nodes?
- What rule should place insertions that are ambiguous within repeats (e.g. left-alignment)?
