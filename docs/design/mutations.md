(sec_design_mutations)=

# Mutations

**Status: Accepted, mostly implemented.** Implemented: the mutations table and the
`gig.mutations` and `gig.mutations_for_node()` accessors, conversion to and from _tskit_, allele
complements, `gig.allele_at()` and `gig.haplotype()`, and handling of mutations in
`sample_resolve()`, `remove_non_ancestral_nodes()` and `decapitate()`. Not yet implemented:
reference sequences, variants, and simplification that removes nodes.
Discussion: {issue}`5`. Related: {issue}`140` (portable reference sequences),
{issue}`126` (pseudo-roots), {issue}`135` (structural change without recombination),
{issue}`141` (a position class).

## Summary

- Mutations are stored in a mutations table, with **no sites table**. Each mutation records its
  position in the coordinates of the node it lies above.
- Each mutation replaces the base at a **single position**, and **never changes the coordinate
  system**. Its derived state is usually a single character. Indels, like other structural
  changes, are preferably represented by new nodes with iedges whose parent and child coordinates
  differ. Small indels can instead be represented by multi-character (or empty) derived states,
  with weaker guarantees.
- There is **no per-site ancestral state**. Ancestral states come from mutations above root nodes
  (and, later, from shared reference sequences).
- Mutation **times may be unknown**. Mutations at the same node and position must be in time
  order, and the table order defines the order of mutations with equal or unknown times.
- Across an inversion, alleles are **complemented** (multi-character alleles are
  reverse-complemented) using a user-configurable map of complements. Alleles that cannot be
  complemented using the map are treated as strand-symmetric. This only affects how alleles are
  decoded, not how ancestry is encoded.
- Genomes with no mutation above them at a position, including isolated nodes, have **unknown
  (missing)** state there.

## Why no sites table?

In _tskit_, a site is a position in the coordinate system shared by all genomes, and a mutation
belongs to a site. A GIG has no shared coordinate system (see {ref}`sec_concepts`): position
1000 in one genome need not correspond to position 1000 in another. So a global table of sites
doesn't make sense, and each mutation instead stores its own position, in its own node's
coordinates.

Whether two mutations are "at the same site" therefore depends on a frame of reference. They are
at the same site *relative to an ancestor* if their positions trace back to the same position in
that ancestor's genome. Site identity, and the mutation "parent" of each mutation, are therefore
*derived* quantities. They can be computed (e.g. when a {class}`~giglib.Graph` is created), but
are not stored in the tables.

## The mutations table

| Column | Meaning |
|---|---|
| `node` | The node the mutation lies above |
| `position` | The (integer) position, in `node`'s coordinates for that chromosome |
| `chromosome` | The chromosome, in `node`'s genome |
| `time` | The time of the mutation, or unknown (see below) |
| `derived_state` | The allele produced by the mutation |
| `metadata` | Arbitrary metadata |

A mutation above node `u` at position `x` occurred on the branch between `u` and its parent at
that position. As in _tskit_, each position in a child is inherited from at most one parent, so
this branch is well-defined (and if `u` has no parent at `x`, the mutation lies above a root).
Unlike _tskit_, the position is in `u`'s coordinates, which may differ from those of its parent
and its children.

### Times

- If known, a mutation's time must be at least the time of `node`, and strictly less than the time
  of `node`'s parent at that position (there is no upper limit for a mutation above a root).
- Times may be unknown, using a special value analogous to _tskit_'s `UNKNOWN_TIME`. Unlike
  _tskit_, known and unknown times may be mixed at the same node and position. Known times must
  agree with the table order (oldest first), and mutations with unknown times take their place in
  the order from the table. (_tskit_ forbids mixing because it would make the order of mutations
  at a site uncheckable. In _giglib_, the table order is authoritative, so this isn't a problem.)
  For example, a mutation giving the ancestral state above a root usually has an unknown time, and
  comes before any other mutations at that node and position.

### Order

Mutations for the same node are adjacent in the table, mirroring the way that iedges for the
same child are adjacent. Within a node, mutations are sorted by chromosome, then by position.
At the same node, chromosome and position, they are sorted with the oldest first. If times are
equal or unknown, the table order defines the order in which the mutations occurred.

Grouping mutations by node means that when an interval is traced rootwards through a node, the
mutations within it can be found with a binary search on position.

(sec_design_mutations_one_position)=

### One position per mutation

Each mutation is at a single position, and its derived state *replaces the base at that
position*. Mutations never change the coordinate system: if they did, the positions recorded on
iedges below them would no longer be correct.

#### Indels: use iedges where possible

The preferred way to represent an insertion or deletion, as for any structural change, is with a
new node, whose iedges have **different parent and child coordinates** (see {ref}`sec_concepts`
and {issue}`135`). For example, a child `C` that has lost positions 10–12 of its parent `P`
(with sequence length 100) is

```
{parent: P, child: C, child_left: 0,  child_right: 10, parent_left: 0,  parent_right: 10}
{parent: P, child: C, child_left: 10, child_right: 97, parent_left: 13, parent_right: 100}
```

An insertion is the reverse: positions in the child that are not inherited from any parent.
Since they have no ancestry above `C`, their bases are given by mutations above `C`, in the same
way as ancestral states are given by mutations above a root (see below).

Coding indels this way keeps coordinates meaningful: every genome's coordinates correspond
exactly to the bases in its sequence, and homologous positions in different genomes can be found
by tracing through the iedges.

#### Indels as alleles (less preferable)

Sometimes adding a node for each small indel is impractical. So a derived state may be any
string, not just a single character, allowing small indels to be represented without changing
the coordinate system:

- **Insertion**: `A` → `ATTG` means that bases `TTG` are inserted after the `A` at that position.
- **Deletion**: `A` → `` (the empty string), or `A` → `-`, means that the base at that position is
  deleted.

A change to several adjacent bases is represented as one mutation per position, all at the same
time. In particular, a deletion of several bases is a run of consecutive deletion mutations. This
doesn't record that they were a single event, but means that a mutation can never be split by an
iedge boundary, so is always inherited (or not) as a whole. If needed, the mutations from a single
event could be linked using metadata.

This differs from VCF, where a deletion such as `ATGC` → `A` is a single record, with the REF
allele specifying how many bases it covers, and the leading `A` included only because VCF cannot
represent an empty allele. In _giglib_, mutations do not store the previous allele, and empty
alleles are allowed, so the same deletion at position *x* is three mutations at the same time:
`` at *x*+1, *x*+2 and *x*+3 (the `A` at *x* is unchanged, so needs no mutation). When importing
from VCF, such a record should be split into one mutation per deleted base. When exporting to
VCF, runs of adjacent deletion mutations with the same time can be merged back into a single
record, adding the anchoring base.

Indels coded as alleles come with **weaker guarantees**. Coordinates no longer correspond to
offsets in the actual DNA sequence: a haplotype can be longer or shorter than the node's sequence
length, and haplotypes from different genomes cannot be compared base by base using coordinates.
They are therefore best kept small and isolated, and coding indels using iedges is preferable.

## Ancestral states

There is no per-site ancestral state. Instead, the state at a position is defined by the
mutations above it. To specify the ancestral state, place a mutation above the root at that
position. The state of any genome at a position with no mutations above it is unknown (missing
data). In particular, there is no option to treat missing data as having an ancestral state: a
node with no iedges at a position isn't connected to any other genome there, so there is no
ancestral state that could apply to it.

This means that if older ancestors are added later, the "ancestral" mutation can be moved
upwards, rather than an ancestral state being fixed for a site.

An ancestral state therefore belongs to a position in the *root's* genome, not to a position in a
sample's genome, and must be placed using the root's coordinates. Like any mutation, it is
inherited by every copy of that position in the root's descendants. So if a descendant has a
duplication, a single mutation above the root gives the ancestral allele at two (or more)
positions in that descendant. This is the correct behaviour, since both copies derive from the
same ancestral base: the positions are the same site relative to the root, even though later
mutations below the duplication may make them differ.

### Later: reference sequences

Storing a full sequence on each root node would be expensive. A human-sized genome is about
3 GB, and in a GIG the root changes along the genome, so the ancestral sequence would be split
across many root nodes. Instead, the plan is to store a small number of shared reference
sequences once, together with a mapping table with rows like

```
(node, chromosome, left, right, reference, ref_left, orientation)
```

meaning "this interval of `node`'s genome corresponds to the reference from `ref_left`, in this
orientation". Only a few nodes (e.g. roots or pseudo-roots, {issue}`126`) need rows, so its size
depends on the number of breakpoints, not on the sequence length. The state at a position is then
found by tracing rootwards until reaching either a mutation, or a node with a mapping into a
reference sequence.

The same mapping table is the structure proposed in {issue}`140` for choosing recombination
breakpoints, and it gives a reference frame for genotypes (see
{ref}`sec_design_mutations_future`), so it should be designed once, for all three purposes.

The first implementation will only support ancestral states given by mutations above roots.
Nothing in the mutations table should prevent reference sequences being added later.

## Looking up alleles

The allele of a genome at a position will be returned by

```python
gig.allele_at(u, position, chromosome=0)
```

This traces the position rootwards through the iedges (transforming the position at each one),
and returns the derived state of the first mutation found, complemented if necessary (see below).
If no mutation is found, it returns `None` (unknown). Similarly,

```python
gig.haplotype(u, chromosome=0, *, left=0, right=None, missing_data_character="N")
```

returns the alleles at a range of positions as a string (see {issue}`113`), with missing data
shown as `N`. For a sample in a tree sequence converted from _tskit_, this matches the
corresponding output of {meth}`tskit:tskit.TreeSequence.alignments`, with two caveats:

- `right` must be given as the tree sequence's length. By default, `haplotype()` stops at the
  end of the node's genome as implied by its iedges, which may be shorter (e.g. if a sample's
  edges end before the end of the tree sequence).
- The tree sequence must not have a reference sequence. Reference sequences are not converted,
  so where `alignments()` would use reference bases (at positions without sites),
  `haplotype()` returns missing data.

Natural extensions are `gig.alleles_at(u, positions)`
for many positions at once, and versions accepting a position object ({issue}`141`).

(sec_design_mutations_strands)=

## Strands and inversions

A genome's coordinates read one strand of the DNA from left to right. An inversion flips a
segment over, so reading the child from left to right through the inverted segment means reading
the *other* strand of the parent, backwards. A base that the parent reads as `A` is therefore
read by the child as `T`: what is preserved is the base *pair*, not the base.

Since a mutation's position is in its own node's coordinates, its derived state is also defined
on its own node's (forward) strand. When `allele_at` traces a position rootwards, it tracks the
number of inversions it passes through. If it finds a mutation after an odd number of inversions,
it reports the *complement* of that mutation's derived state (or, for a multi-character derived
state, its *reverse complement*). In other words, complementing an allele is part of
transforming coordinates across an inversion, just like reflecting the position.

Because a derived state replaces whole bases, reverse-complementing gives the right answer for
insertions too. If a parent's `A` becomes `ATTG` (the insertion `TTG` after the `A`), a child that
inherits that position through an inversion reads `CAAT`, i.e. the inserted bases now come
*before* the complement of the original base, exactly as reading the other strand backwards
requires.

### The complement map

Not every allele can be complemented: `0` and `1` (as used by default in _msprime_) have no
complement, and a deletion (`-`) is the same on both strands. Which alleles are complemented is
therefore defined by a map on the {class}`~giglib.Tables`:

```python
tables.allele_complements = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N", "-": "-", ...}
```

When an allele is traced across an inversion, the rules are applied in this order:

1. If the whole allele is a key in the map, it is replaced by the corresponding value. This
   allows the complement of any allele to be specified explicitly.
2. Otherwise, if the allele has more than one character, and every character is in the map, it
   is reverse-complemented: each character is complemented using the map, and the order of the
   characters is reversed. (The empty string, representing a deletion, is unchanged.)
3. Otherwise, the allele is left unchanged: it is treated as strand-symmetric.

The default map contains the IUPAC nucleotide codes (including ambiguity codes, e.g. `R` ↔ `Y`)
plus `-` ↔ `-`. If `A`, `C`, `G` and `T` are used as abstract labels, set the map to `{}` to turn
complementing off.

Note that rule 3 also applies to single-character alleles that are not in the map (e.g. `X`, or
`0` and `1`): these are left unchanged without any warning, so not every allele inherited through
an inversion is complemented.

When a {class}`~giglib.Graph` is created, a warning is issued if it contains at least one
inversion *and* any multi-character allele that falls under rule 3 (i.e. that is not in the map,
and contains characters that are not in the map). The warning lists those alleles, explains that
they will be treated as strand-symmetric, and suggests adding them to the map (mapping an allele
to itself if it really is symmetric). No warning is issued for graphs without inversions, such as
those converted from _tskit_.

### Strand handling only affects decoding

The complement map only affects how alleles are *reported*. It is used when decoding alleles
(by `allele_at`, and later by haplotype extraction and the equivalent of `variants()`), and for
the one-off check that generates the warning above. It has no effect on how ancestry or mutations
are encoded or processed. Storing, sorting and validating the tables, simplification,
`find_mrca_regions`, and calculating mutation parents are all purely structural, and never
examine allele states. So this machinery adds no cost to the encoding of ancestry.

Even when decoding, tracking orientation is essentially free: tracing a position through an
inversion already requires reflecting it, so whether an odd number of inversions has been
crossed is known at no extra cost. Complementing is then a single lookup for the one allele
that is reported.

### Example

Consider a parent `P` and child `C`, where `C` inherits positions 0–10 of `P` inverted
(an iedge with `child_left=0, child_right=10, parent_left=10, parent_right=0`), and a mutation
above `P` at position 2:

| Derived state of the mutation | `allele_at(P, 2)` | `allele_at(C, 7)` |
|---|---|---|
| `A` | `A` | `T` |
| `ATTG` (insertion) | `ATTG` | `CAAT` |
| `-` or `` (deletion) | `-` or `` | `-` or `` |
| `1` | `1` | `1` |

(Position 2 in `P` corresponds to position 7 in `C`, because an inversion of the interval
\[0, 10) maps position `x` to `10 - x - 1`.)

Strand-specific mutation processes are handled automatically. For example, a C→T mutation
recorded on one node's strand is correctly seen as G→A by a descendant that inherits it through
an inversion.

## Simplification

- A node with a duplication beneath it represents a split in the local genealogy, so
  simplification should always keep such nodes. (A possible later optimisation is to push
  duplications up to the next coalescent node.) This also avoids having to turn one mutation
  above a duplicated node into several mutations, one for each duplicated copy.
- Mutations in regions that are not inherited by any sample are removed, as in _tskit_.
  The current {meth}`~giglib.Tables.sample_resolve` does this, and never removes nodes, so
  other mutations stay above the same nodes. Mutations above sample nodes are always kept.
  Simplification that removes (e.g. unary) nodes is not yet implemented: it would need to
  move mutations above a removed node down to the corresponding positions in its child. See
  {ref}`sec_design_simplification`.

## Decapitation

{meth}`~giglib.Tables.decapitate` removes nodes at least as old as a given time, and the iedges
above the remaining nodes. As in _tskit_, it removes mutations at least as old as that time, as
well as mutations above removed nodes. But because ancestral states are themselves mutations
above roots, removing them would turn known alleles into missing data. So by default
(`discard_old_mutations=False`), wherever a remaining node loses its parent or some of its own
(old) mutations, mutations with unknown time are added above it, recording the alleles it
inherited there. These include the effect of any removed mutations on the node's own branch,
and come before the node's other mutations at the same position. Any of those other mutations
that came before a removed one are overwritten by it, so they are removed too. The haplotypes of
all remaining nodes are therefore unchanged. Only positions with mutations above them are
examined, so the time taken does not depend on the lengths of the iedges. With
`discard_old_mutations=True`, old mutations are simply removed, as in _tskit_, so haplotypes may
gain missing data.

## Converting to and from _tskit_

- **From _tskit_**: each mutation keeps its node, and its position is the position of its site
  (a tree sequence has no rearrangements, so the coordinates are the same everywhere). Unknown
  times stay unknown. Each site's ancestral state becomes a mutation with unknown time above each
  root of the local tree at that position, placed before any other mutations at that root and
  position. Isolated nodes (roots with no children) get no such mutation, so are left as missing
  data. This matches _tskit_'s default treatment of isolated samples as missing.
- **To _tskit_**: this is mainly for testing, so only the simplest cases are supported.
  Mutations at the same position are grouped into sites. At each site, the first mutation above
  each (non-isolated) root of the local tree gives the site's ancestral state: every root must
  have such a mutation, and they must all agree, otherwise an error is raised. All other
  mutations become _tskit_ mutations, and must either all have known times or all have unknown
  times. The iedges must also be convertible (i.e. there are no rearrangements), and all iedges
  and mutations must be on the same chromosome. Converting a tree sequence to a GIG and back
  gives the same sites and mutations (although if times are unknown, mutations at a site may be
  reordered).

(sec_design_mutations_future)=

## Not yet: variants and genotypes

The equivalent of _tskit_'s `variants()` will need a reference frame: a reference sequence, or
the genome of a particular node. Each position in the reference is mapped down to every sample.
A sample can then have zero, one or several corresponding positions (deleted, present once, or
duplicated), so genotypes may have more than one value per sample. Alleles from samples that
inherit a position in opposite orientations can be reported on the reference strand, using the
`orientation` column of the reference mapping table.

This is not part of the first implementation, but the design above should not prevent it.

## API summary

Following the {ref}`API conventions<sec_usage_consistency>`:

| | `Graph` (objects) | `Tables` (IDs and arrays) |
|---|---|---|
| All mutations | `gig.mutations` (view of `Mutation` objects) | `tables.mutations` (the table) |
| Mutations above a node | `gig.mutations_for_node(u, chromosome=None)` | `tables.mutations.ids_for_node(u, chromosome=None)` |
| Allele lookup | `gig.allele_at(u, position, chromosome=0)` | |
| Haplotypes | `gig.haplotype(u, chromosome=0, *, left=0, right=None)` | |
| Complement map | | `tables.allele_complements` |

(sec_design_mutations_open)=

## Open questions

- **Metadata.** The mutations table has a `metadata` column, but in general _giglib_ does not
  yet implement metadata properly (e.g. schemas and encoding, as in _tskit_). It should do so
  eventually, for all tables.
