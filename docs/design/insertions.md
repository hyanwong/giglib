(sec_design_insertions)=

# Inserted material

**Status: Decided** (see {ref}`sec_design_insertions_decision`), apart from the remaining open
questions. Discussion: {issue}`178`, and the alignment tool ({issue}`139`, {pr}`175`, {issue}`194`),
which raised the question. Related: {issue}`158` (sequence lengths), {issue}`126` (pseudo-roots),
{ref}`sec_design_mutations`, {ref}`sec_design_simplification`.

## Summary

- **The question:** should every position of a genome that has ancestors (a non-root node) have
  a parent? Or can a node have positions with no iedge above them, which are effectively "new"
  material, with no recorded ancestry?
- **Inserted material comes from somewhere.** Insertions are either copies of material from
  elsewhere in the graph (duplications, transpositions, unequal crossing-over), which should be
  represented by an iedge from that source, or genuinely new bases.
- **The origin is often unknown.** In real data, it is often impossible to tell where a small
  insertion came from, or even exactly where it is. So "origin unknown" needs a representation,
  which can be refined if the origin is later identified.
- **Decision:** inserted material of new or unknown origin is a **local root**: positions with no
  parent, inside a node that has parents elsewhere. The inserted bases are given by mutations
  above that node, as ancestral states are given by mutations above a root, with `N` for a base
  that is unknown. There are no extra "source" nodes.

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

a. **Positions with no parent** (local roots), inside a non-root node. This is what nothing
   forbids, and what the alignment tool ({pr}`175`) does. **This is the option chosen** (see
   {ref}`sec_design_insertions_decision`).
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
| Sequence lengths from iedges ({issue}`158`) | wrong if an insertion is at the end of a genome (but correct if lengths also use mutations: see below) | correct | correct |
| Homology of separate insertions | none | none (separate roots) | implies they are all related, which is misleading |
| Number of extra nodes | 0 | one per insertion event | 1 |
| Refining the origin later | add an iedge | replace the source node's iedge by one from the real source | replace one iedge |

Further considerations (written when source nodes were proposed):

- **Time of a source node.** The insertion happened on the branch above the node that gained it,
  so a source node could be given the time of that node's parent (or of the insertion, if
  known).
- **Marking source nodes.** They would need a node flag or metadata, so that conversions and
  algorithms could treat them specially.
- **Simplification.** Source nodes would be unary roots, which simplification would need to keep.
- **Pseudo-roots ({issue}`126`)** are a different idea: a cached ancestral *sequence* at a node,
  to avoid tracing further up the graph.

## Conversions

- **From tree sequences with alignment columns** ({pr}`175`): an insertion is a mutation from `-`
  to some bases above a node, so the inserted positions of that node have no parent, and their
  bases are mutations above it (`N` if unknown).
- **To tree sequences** ({issue}`194`): each inserted position (with no parent) gets a new
  alignment column, with ancestral state `-`, and a mutation above the node that gained the
  insertion.

(sec_design_insertions_decision)=

## Decision

Inserted material of new or unknown origin is represented by **local roots**: positions of a node
that have no parent, although the node has parents at other positions (option a). Insertions of
known origin (duplications, transpositions, unequal crossing-over, or foreign DNA from a known
donor, as in option d) are iedges from their source, as before.

**Why not source nodes (option b)?** Local roots need nothing new:

- Decoding alleles, and {meth}`~giglib.Graph.ancestral_allele_at`, already trace a position up
  until it has no parent, and then look for a mutation at that node. So inserted bases are decoded
  like any other ancestral state: "the ancestral state is the mutation above the root" holds
  position by position.
- There are no extra nodes, node flags or conventions about the times of source nodes, and the
  origin is easily refined later, by adding an iedge from the real source.
- Simplification needs no special case. If the node that gained an insertion is removed (where it
  is unary), the inserted positions become local roots of its child, and their mutations are moved
  down to it. If several descendants inherit the insertion, the node is a coalescence there, so is
  kept.
- Converting to a tree sequence ({issue}`194`) gives each position with no parent a new column,
  which is exactly an inserted position.

**Conventions,** to keep this unambiguous:

1. **Every inserted position has a mutation** above the node that gained it, with derived state
   `N` if the base is unknown (`N` is treated as an allele, e.g. by the alignment tool). So a
   position with no parent and no mutation is *missing data* (as for an isolated sample in
   _tskit_), not an inserted base of unknown sequence: the two would otherwise look the same,
   although only inserted bases are part of the genome (e.g. for statistics).
2. **Sequence lengths count mutations as well as iedges.** A node's length on a chromosome is the
   largest position given by the iedges above or below it, or by mutations above it. With
   convention 1, this counts an insertion at the end of a genome, which iedges alone would not.

**What is lost** compared with source nodes:

- **Checking for accidental gaps.** Code that should never create insertions (e.g. a simulator
  without them) can no longer rely on "every non-root position has a parent" to catch a bug that
  leaves positions without a parent. An optional validity flag could restore this check (see the
  open questions).
- **The time of the insertion** is not recorded, only bounded: it happened on the branch above
  the node that gained it. If it is known, it can be given as the time of the mutations for the
  inserted bases (which have no parent, so are only bounded below by the node's time).
- **Homology between separate insertions** is not recorded, as with source nodes: two independent
  insertions of the same sequence are unrelated local roots, which is correct. Insertions that are
  homologous because they share a known donor need iedges from a donor node.

## Open questions

- **An optional validity flag** for code that never creates insertions, e.g. requiring the iedges
  of each child to cover each of its chromosomes without gaps, starting from position 0. This is
  cheap to check incrementally once the iedges are sorted.
- **Checking convention 1:** should validation (or creating a `Graph`) check that every inserted
  position has a mutation? This needs the positions with no parent of each node, which can be
  found from the sorted iedges.
- **Placing insertions that are ambiguous within repeats:** what rule should conversions from data
  use (e.g. left-alignment, as in VCF)?
