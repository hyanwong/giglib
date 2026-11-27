(sec_design_homology_table)=

# The homology table

**Status: Proposed.** Not implemented, and only needed if measurements show that tracing through
the graph limits large simulations. Discussion: {issue}`140` (which also covers {issue}`210`).
Related: {ref}`sec_design_simplification` (the operation that would build it),
{ref}`sec_design_homology`, {ref}`sec_design_recombination`, {ref}`sec_design_storage` (cached
indexes), {ref}`sec_design_mutations` (reference sequences).

## Purpose

Many questions about homology don't need to know *when* two positions shared an ancestor, only
*which* positions are homologous, and in which orientation. Two examples are choosing where a
crossover goes, and finding homology blocks ({func}`giglib.tools.homology.homology_blocks`, which
the microsatellite simulator calls at every meiosis). At the moment they trace both genomes up
through the graph, at a cost that grows with its depth and with how fragmented it is.

The *homology table* would answer them faster. It is a set of iedges that map each genome
directly up to a few kept ancestors, by composing the iedges in between, as
{meth}`~giglib.Tables.simplify` does (see {ref}`sec_design_simplification`). This **merges changes
across generations**. For example, a microsatellite that expanded, contracted and expanded again
becomes a single net shift, or none at all if it returned to its ancestral length. So the number of
pieces in a genome's map depends on how many places it differs from its kept ancestors, not on how
many generations lie between them. It **can't merge differences between individuals**. The map
from `u` to `v` has a piece boundary wherever the two genomes differ in length, as that is a real
difference between them. Chaining blocks is still what decides which homology is allelic.

## An internal cache, not data

The table is derived entirely from the iedges, so it is a **cache**, like the indexes of the
iedges above and below each node:

- it is owned by the tables, never saved, and rebuilt or updated when needed;
- it is invalidated when the iedges change (e.g. by a version counter, as the node table's version
  already invalidates the node-time validity flags);
- it must never change an answer, only how quickly it is found;
- its public face is the queries that use it (e.g. `homologous_location()`, placing crossovers,
  {func}`~giglib.tools.homology.homology_blocks`, {meth}`~giglib.Graph.ancestral_allele_at`), not
  the table itself.

It stays reference-free: its tops are whichever nodes are kept, not a chosen reference genome.
That is the difference from the **reference mapping** proposed in the mutations design
({ref}`sec_design_mutations`), which maps nodes onto shared reference sequences to give ancestral
states. That mapping is data, with a meaning of its own, which users see and save. The two can
share the format of their rows (an interval of a node, an interval of its target, and an
orientation) and the code that composes maps, but not their status.

## Which nodes to keep

The table is built by the same core as {meth}`~giglib.Tables.simplify` (composing iedges, region
by region), with a different rule for which nodes to keep. Simplification keeps every branch point
of the position tree, of two kinds (see {ref}`sec_design_simplification_branch_points`):
*splits*, where lineages from different children merge, and *copies*, where two iedges of one child
inherit the same position (the parent of a duplication). There is a range of options:

| Keeps | Can tell allelic from paralogous? | Pieces per genome |
|---|---|---|
| Only the roots: maps to the roots ({issue}`210`) | No | Fewest |
| Copies, and duplication nodes; bypasses splits | Yes | One per region, plus about one per breakpoint of a structural variant in the genome's ancestry |
| Splits and copies: simplification itself | Yes (and gives MRCAs) | Many, as each lineage passes through every coalescence |

**Maps to the roots** are enough to find homology blocks, which depend only on *where* positions
map to. But they find homology of every age. If everything descends from one ancestral position,
every copy of a repeat maps to the same root position. Then allelic and paralogous pairs can't be
told apart, which is what choosing a crossover needs.

**Keeping copies and duplication nodes** fixes this. If only the copy branch point P (the parent)
is kept, then below a duplication both copies map to the same positions of P. If the duplication
node D (the oldest genome with both copies, see
{ref}`sec_design_simplification_branch_points`) is also kept, the copies are distinct positions
of D. So copy 1 in `u` and copy 1 in `v` meet at D (allelic), while copy 1 in `u` and copy 2 in
`v` only meet at P (paralogous). Inversions and deletions need no kept nodes, as they just change
the map, or remove positions. Detecting duplication nodes is an open question, but a simulator can
flag them, and pass the flag to the core as the nodes to keep (as proposed in {issue}`244`). This
is the suggested rule.

## What it can't answer

With the splits bypassed, the table doesn't give the MRCA or the TMRCA of allelic pairs. So it
can't serve {meth}`~giglib.Tables.find_homology`, which reports them. Queries that need times use
the iedges (after simplifying, which is the right speed-up for them: see the findings in
`evaluation/README.md`). Mutations are unaffected: they stay above their nodes in the real graph.

## When it helps

A query traces each lineage through every kept node, in each region (local tree) of the genome.
After simplifying, the cost for a pair of genomes is roughly (nodes on a lineage in a local tree,
about 2 ln *n* for *n* samples in a coalescent) × (number of local trees). With the table, a genome
costs its number of pieces. So the gain is large for **many samples, with recombination, and few
structural variants**: the human-like case. It is small when nearly every node changes coordinates.
For example, in the single-origin simulator most kept nodes are copies anyway.

## Keeping it up to date

In a forward simulation, a child's map is its parents' maps composed through the child's iedges.
That is proportional to the number of pieces, with no tracing. Adjacent pieces with the same top
and a continuous map merge (as {meth}`~giglib.tables.IEdgeTable.squash` does), so without
structural variants a genome stays at about one piece per region. A newly flagged duplication node
becomes the top of the duplicated region for its descendants.

**Defragmentation.** Over a long simulation, a genome's map collects the breakpoints of every
structural variant since its tops, even those now fixed in the population (e.g. once a deletion is
fixed, the pieces either side could be one). Re-rooting at a more recent node, once all the
lineages pass through it, merges them. This is equivalent to rebuilding the table with the
simplify core. It is only needed for speed, not for correctness, so it can be done whenever the
graph is simplified.

## Evaluation and checks

Measure first. The `microsatellites_population_size` case in `evaluation/performance.py` (the time
per generation of a simulation that finds homology at every meiosis, as the population grows)
shows whether tracing is the bottleneck. Then measure:

- the time to find homologous locations, and homology blocks, with and without the table, against
  the population size and the depth of the graph;
- the cost of keeping the table up to date, per generation, and its size compared with the iedges;
- the cost and benefit of defragmenting, and how often to do it.

Checks of correctness: the homologous pairs of positions, and their orientations, found with the
table equal those from {meth}`~giglib.Tables.find_homology` with `mode="all"` (ignoring the
MRCAs), on the graphs used to test {meth}`~giglib.Tables.simplify` (`tests/test_simplify.py`).

## Suggested order

1. Measure, as above.
2. Build the table from a set of tables, with the simplify core and the rule above, and check it.
3. Keep it up to date in a simulator, and use it to place crossovers.

## Open questions

- **Several equivalent positions.** A position can have more than one equivalent in the target
  genome (e.g. after a tandem duplication), or none (after a deletion). Should
  `homologous_location()` return all of them (the most general; the others can be built on it),
  take a way of choosing one (e.g. at random, to place a crossover), or report the ambiguity?
- **Duplication nodes.** Flagged by simulators only, or also detected from the graph?
- **Where it lives in memory,** and how it is shared between a {class}`~giglib.Tables` and the
  {class}`~giglib.Graph` views of it (the same question as for the other indexes, see
  {ref}`sec_design_storage`).
