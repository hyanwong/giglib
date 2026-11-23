"""
Convert between GIGs with insertions and deletions, and tree sequences whose
coordinates are the columns of a multiple sequence alignment.

This module is experimental: see :ref:`sec_tools`. Discussion: :issue:`139` and
:issue:`194`.

**Representation.** A tree sequence cannot represent changes in the coordinate system,
but insertions and deletions (indels) can be "hacked" into one by using the columns of
an alignment as its coordinates. A gap in a genome is represented by the character
``-``. Each site can span several columns, given by the length of its alleles, which must
all be the same length. So a deletion of three bases can be a single mutation from
``ATCG`` to ``A---``, an insertion a mutation from ``A---`` to ``ATCG`` (or ``ANNN`` if
the inserted bases are unknown), and a SNP within an indel region is simply another allele
of the same site. No anchor base is needed, but anchored (VCF-like) alleles also work. The
genome of a node is then its non-gap columns, in order. As
:meth:`tskit:tskit.TreeSequence.alignments` cannot decode multi-character alleles, this
module provides :func:`alignments` (using :func:`split_sites`).

**Tree sequence to GIG** (:func:`from_tree_sequence`). Each node's coordinates are those
of its non-gap columns. Each edge becomes iedges over the runs of columns that both the
child and the parent have, breaking only where exactly one of them has a gap. Columns
that only the child has are inserted positions, with no parent, and their bases come
from mutations above the child: every inserted position has one (``N`` if the base is
unknown), as required by :ref:`sec_design_insertions`. Where a node is not in a tree,
its coordinates follow the gaps of the ancestral state, so that they stay the same as
those of the roots. Sites must not overlap, must not extend beyond the end of the
sequence, and the columns of a site must all be in the same tree (a site's mutations
belong to the tree at its position).

**GIG to tree sequence** (:func:`to_tree_sequence`). The roots of the GIG must share
coordinates, which give the reference columns. Each position inherited through an iedge
takes its parent's column. A run of positions with no parent could be the root of one
of a tree sequence's trees (colinear with the other genomes) or an insertion (with new
columns), and only the iedges either side of it can tell: it fills the reference columns
between them, and any excess positions become inserted columns. A run at the start or
end of a genome is bounded by its children, where they inherit it next to other
material. Columns are ordered by sortable keys: a reference column ``x`` has key
``(x,)``, and the ``i``-th column inserted by node ``u`` after key ``k`` has key
``k + (u, i)``, so nested insertions sort within their parent insertion, and independent
insertions at the same place are ordered by node ID. Each iedge becomes an edge over its
columns, and the columns that a node lacks (deletions, and other nodes' insertions) are
covered by extending an edge, so that the node inherits a gap there. Each maximal run of
columns in a tree where any node has a gap becomes one site, other columns with a known
base become single-column sites, and there is one mutation above each node whose string
differs from its parent's. See :func:`to_tree_sequence` for the details.

**Round trips, and what is lost.** Converting a tree sequence to a GIG and back gives
the same haplotypes (the alignments with their gaps removed), but not necessarily the
same columns or sites: columns in which every genome has a gap are dropped, independent
insertions at the same place may be ordered differently, multi-column sites without
gaps (blocks of SNPs) are split into single columns, and anchor bases are dropped. For
a tree sequence without indels, the alignments are the same. Converting a GIG to a tree
sequence and back gives the same haplotypes, but missing data (positions with no parent
and no mutation) can come back as unknown bases (the allele ``N``).

For a GIG with rearrangements that are not colinear, a tree sequence can keep the
sequences, but not the homology between bases:

- an in-place inversion could become a reverse-complemented allele over the same columns
  (``AACG`` to ``CGTT``), which converts back to substitutions;
- a duplicated copy could become inserted columns with known bases, which converts back
  to an insertion unrelated to the original copy;
- a translocation within a chromosome could become a deletion and an insertion;
- several chromosomes could become one tree sequence per chromosome.

Conversion to a tree sequence raises an error for rearrangements that are not colinear,
so that no information is silently lost. An option (e.g. ``rearrangements="as_indels"``)
could later encode them as above.
"""

import bisect

import numpy as np
import tskit

import giglib

GAP = "-"
UNKNOWN = "N"


def split_sites(ts):
    """
    Return a copy of a tree sequence in which each site whose alleles have more than one
    character is split into one site per column, with single-character alleles. Each
    mutation becomes one mutation for each column in which its derived state differs
    from the previous state at that column, at the same node and with the same time
    and metadata. The genotypes of every node in each column are unchanged.

    :param tskit.TreeSequence ts: A tree sequence whose sites satisfy the requirements of
        :func:`from_tree_sequence`.
    :return: The tree sequence, with single-character alleles
    :rtype: tskit.TreeSequence
    """
    _check_sites(ts)
    tables = ts.dump_tables()
    tables.sites.clear()
    tables.mutations.clear()
    derived_state = {}
    for tree in ts.trees():
        for site in tree.sites():
            position = int(site.position)
            site_ids = [
                tables.sites.add_row(position=position + j, ancestral_state=char, metadata=site.metadata)
                for j, char in enumerate(site.ancestral_state)
            ]
            for mutation in site.mutations:
                previous = site.ancestral_state if mutation.parent == tskit.NULL else derived_state[mutation.parent]
                derived_state[mutation.id] = mutation.derived_state
                # A mutation above an isolated node establishes its state in every
                # column (otherwise its unchanged columns would be missing data)
                isolated = _is_isolated(tree, mutation.node)
                for site_id, char, previous_char in zip(site_ids, mutation.derived_state, previous):
                    if isolated or char != previous_char:
                        tables.mutations.add_row(
                            site=site_id,
                            node=mutation.node,
                            derived_state=char,
                            time=mutation.time,
                            metadata=mutation.metadata,
                        )
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


def alignments(ts, *, missing_data_character=None):
    """
    Return the alignment of each sample in a tree sequence, as a list of strings, one
    per sample, each with one character per column (i.e. of length
    ``ts.sequence_length``). Unlike :meth:`tskit:tskit.TreeSequence.alignments`, sites
    can have alleles of more than one character (all of the same length), each covering
    the corresponding number of columns, starting at the site's position. This is done
    by splitting the sites into single columns (see :func:`split_sites`), and using
    :meth:`tskit:tskit.TreeSequence.alignments`.

    :param tskit.TreeSequence ts: A tree sequence whose sites satisfy the requirements of
        :func:`from_tree_sequence`.
    :param str missing_data_character: The character for missing data, including columns
        not covered by any site (default: ``N``).
    :return: A list of strings, one for each sample, in the order of ``ts.samples()``.
    :rtype: list(str)
    """
    if missing_data_character is None:
        missing_data_character = UNKNOWN
    split = split_sites(ts)
    # tskit does not allow alleles to equal the missing data character, but an allele
    # can be N (e.g. an unknown inserted base). So use a placeholder character that is
    # not an allele, and replace it afterwards.
    used = set(split.tables.sites.ancestral_state.tobytes()) | set(split.tables.mutations.derived_state.tobytes())
    placeholder = next(chr(c) for c in range(1, 128) if c not in used)
    # Pass a reference sequence of placeholders, so that columns without sites are always
    # missing data (otherwise tskit would use any reference sequence in the tree sequence)
    reference = placeholder * int(ts.sequence_length)
    return [
        row.replace(placeholder, missing_data_character)
        for row in split.alignments(missing_data_character=placeholder, reference_sequence=reference)
    ]


def _site_spans(ts):
    """
    Return a list of (position, length) for each site, checking that all the alleles
    at a site have the same length
    """
    spans = []
    for site in ts.sites():
        position = site.position
        if position != int(position):
            raise ValueError("Site positions must be integers")
        lengths = {len(site.ancestral_state)} | {len(m.derived_state) for m in site.mutations}
        if len(lengths) != 1:
            raise ValueError(f"Site {site.id}: all alleles must have the same length")
        length = lengths.pop()
        if length == 0:
            raise ValueError(f"Site {site.id}: alleles must not be empty")
        spans.append((int(position), length))
    return spans


def _check_sites(ts):
    """
    Check that sites have alleles of a consistent length, do not overlap each other,
    do not extend beyond the end of the sequence, and do not cross a tree boundary.
    Return the (position, length) of each site.
    """
    spans = _site_spans(ts)
    breakpoints = np.array(list(ts.breakpoints()))
    end = 0
    for site_id, (position, length) in enumerate(spans):
        if position < end:
            raise ValueError(f"Site {site_id} overlaps the columns of the previous site")
        end = position + length
        if end > ts.sequence_length:
            raise ValueError(f"Site {site_id} extends beyond the end of the sequence")
        # No tree boundary strictly inside the columns of the site
        i = np.searchsorted(breakpoints, position, side="right")
        if i < len(breakpoints) and breakpoints[i] < end:
            raise ValueError(
                f"Site {site_id} crosses a tree boundary at position {breakpoints[i]}: the columns "
                "of a site must all be in the same tree"
            )
    return spans


def _is_isolated(tree, u):
    return tree.parent(u) == tskit.NULL and tree.num_children(u) == 0


def _node_states(tree, site, num_nodes):
    """
    Return a list of the allele of every node at a site. As in tskit, the state of an
    isolated node (with no parent or children in the tree) with no mutation is unknown,
    so is all ``N`` (which does not change its coordinates), rather than the ancestral
    state. A node that is not in the tree has unknown bases, but the gaps of the ancestral
    state, so that its coordinates stay the same as those of the roots (its material is
    unrecorded there, so only the numbering of its positions is affected).
    """
    mutation_state = {}
    for mutation in site.mutations:  # mutations are ordered so that later ones are more recent
        mutation_state[mutation.node] = mutation.derived_state
    absent = "".join(GAP if char == GAP else UNKNOWN for char in site.ancestral_state)
    states = [absent] * num_nodes
    for u in tree.nodes(order="preorder"):
        parent = tree.parent(u)
        if parent != tskit.NULL:
            states[u] = mutation_state.get(u, states[parent])
        elif tree.num_children(u) > 0:  # a root that is not isolated
            states[u] = mutation_state.get(u, site.ancestral_state)
        else:  # an isolated node
            states[u] = mutation_state.get(u, UNKNOWN * len(site.ancestral_state))
    return states


def from_tree_sequence(ts):
    """
    Create a GIG from a tree sequence whose coordinates are alignment columns, with
    gaps represented by ``-`` (see the module description). Each node's genome consists
    of the columns at which the node does not have a gap, so the coordinates of nodes
    change wherever their alleles contain gaps. Each edge becomes one or more iedges,
    covering the columns that both the child and the parent have.

    Mutations become mutations above the same node, one for each column in which the
    derived state differs from the previous state (a column changing to a gap is
    instead represented by the change in coordinates). As in
    :meth:`giglib.Graph.from_tree_sequence`, the ancestral state of each site becomes
    mutations above the roots of the local tree. An inserted base whose state is
    unknown (``N``) becomes a mutation to the allele ``N``: as in the tree sequence,
    this is an allele (the IUPAC code for any base), not missing data, so it is shown
    as ``N`` whatever ``missing_data_character`` is used. As in tskit, an
    isolated node (with no parent or children in a tree) that has no mutation at a site
    has an unknown state there (``N``, not a gap), while a mutation above an isolated
    node gives its state in every column of the site.

    The haplotype of each sample in the resulting GIG is the alignment of that sample
    (see :func:`alignments`) with the gaps removed.

    :param tskit.TreeSequence ts: The tree sequence. Site positions must be integers.
        All the alleles at a site must have the same length, giving the number of
        columns covered by the site. Sites must not overlap, and the columns of a site
        must all be in the same tree.
    :return: A GIG
    :rtype: giglib.Graph
    """
    _check_sites(ts)
    if int(ts.sequence_length) != ts.sequence_length:
        raise ValueError("The sequence length must be an integer")

    # Nodes and individuals (and checks on e.g. migrations) come from the standard
    # conversion of a copy of the tree sequence without its edges, sites and mutations
    bare = ts.dump_tables()
    bare.edges.clear()
    bare.mutations.clear()
    bare.sites.clear()
    tables = giglib.Tables.from_tree_sequence(bare.tree_sequence())

    # The gap columns of each node, in increasing order
    gaps = [[] for _ in range(ts.num_nodes)]
    for tree in ts.trees():
        for site in tree.sites():
            alleles = [site.ancestral_state, *(m.derived_state for m in site.mutations)]
            if not any(GAP in allele for allele in alleles):
                continue
            position = int(site.position)
            for u, state in enumerate(_node_states(tree, site, ts.num_nodes)):
                gaps[u].extend(position + j for j, char in enumerate(state) if char == GAP)

    def node_position(u, column):
        # The position in the genome of node u (which must not have a gap at column)
        return column - bisect.bisect_left(gaps[u], column)

    # Edges become iedges. An iedge must break at a column where exactly one of the child
    # and parent has a gap (an insertion or deletion between them). Columns where both
    # have a gap are not in either genome, so don't need a break.
    for edge in ts.edges():
        left, right = int(edge.left), int(edge.right)
        if left != edge.left or right != edge.right:
            raise ValueError("Edge coordinates must be integers")
        child, parent = edge.child, edge.parent
        child_gaps, parent_gaps = (
            set(g[bisect.bisect_left(g, left) : bisect.bisect_left(g, right)]) for g in (gaps[child], gaps[parent])
        )
        start = left
        for column in [*sorted(child_gaps ^ parent_gaps), right]:
            child_left, child_right = node_position(child, start), node_position(child, column)
            parent_left, parent_right = node_position(parent, start), node_position(parent, column)
            assert child_right - child_left == parent_right - parent_left
            if child_right > child_left:
                tables.iedges.add_row(
                    child,
                    parent,
                    child_left=child_left,
                    child_right=child_right,
                    parent_left=parent_left,
                    parent_right=parent_right,
                )
            start = column + 1

    # Mutations, one per changed column that the node has (a change to a gap is instead
    # represented by the change in coordinates)
    mutation_derived_state = {}
    for tree in ts.trees():
        if tree.num_sites == 0:
            continue
        parent_array = tree.parent_array[:-1]
        num_children = tree.num_children_array[:-1]
        roots = np.flatnonzero((parent_array == tskit.NULL) & (num_children > 0))
        for site in tree.sites():
            position = int(site.position)
            ancestral = site.ancestral_state
            # Ancestral states first, so that they precede other mutations after sorting
            for root in roots:
                for j, char in enumerate(ancestral):
                    column = position + j
                    if char in (GAP, UNKNOWN) or _has_gap(gaps[root], column):
                        continue
                    tables.mutations.add_row(int(root), node_position(root, column), char)
            for mutation in site.mutations:
                previous = ancestral if mutation.parent == tskit.NULL else mutation_derived_state[mutation.parent]
                mutation_derived_state[mutation.id] = mutation.derived_state
                time = None if tskit.is_unknown_time(mutation.time) else mutation.time
                # A mutation above an isolated node establishes its state in every column
                isolated = _is_isolated(tree, mutation.node)
                for j, (char, previous_char) in enumerate(zip(mutation.derived_state, previous)):
                    column = position + j
                    if char == GAP or (char == previous_char and not isolated):
                        continue
                    if _has_gap(gaps[mutation.node], column):
                        continue  # deleted by a later mutation above the same node
                    tables.mutations.add_row(
                        mutation.node,
                        node_position(mutation.node, column),
                        char,
                        time=time,
                        metadata=mutation.metadata,
                    )

    tables.sort()
    # These properties follow from those of the tree sequence's edges
    tables.iedges.set_bitflag(
        giglib.ValidFlags.IEDGES_INTEGERS
        | giglib.ValidFlags.IEDGES_INTERVALS
        | giglib.ValidFlags.IEDGES_FOR_CHILD_NONOVERLAPPING
        | giglib.ValidFlags.IEDGES_PARENT_OLDER_THAN_CHILD
        | giglib.ValidFlags.IEDGES_SAME_PARENT_CHILD_FOR_EDGE
        | giglib.ValidFlags.IEDGES_CHILD_INTERVAL_POSITIVE
    )
    return tables.graph()


def _has_gap(node_gaps, column):
    i = bisect.bisect_left(node_gaps, column)
    return i < len(node_gaps) and node_gaps[i] == column


def to_tree_sequence(gig):
    """
    Convert a GIG with insertions and deletions to a tree sequence whose coordinates are
    the columns of an alignment, with gaps represented by ``-`` (see the module
    description). This is the reverse of :func:`from_tree_sequence`: the haplotype of each
    sample in the GIG is its alignment in the tree sequence (see :func:`alignments`) with
    the gaps removed, with missing data as ``N``.

    The columns are placed as follows. The roots of the GIG must share coordinates
    (e.g. as for a GIG converted from a tree sequence), which give the *reference
    columns*. Each position inherited through an iedge takes the column of its parent.
    A run of positions with no parent fills the reference columns between the positions
    either side of it, so that it is colinear with the other genomes (as at the root of
    one of a tree sequence's trees). Only the positions in excess of the available
    reference columns are *inserted columns*, placed at the end of the run. A run at the
    start or end of a genome is only bounded on one side, so is taken to be colinear,
    unless a child inherits it next to a column that it would overlap, in which case it
    is bounded by that column.

    Each iedge becomes an edge over the columns of its positions. A node's positions with
    no parent have no edge, so the node is a root there. The columns that a node lacks
    (e.g. deleted positions, or other nodes' inserted positions) are covered by extending
    the edge to their left (or, before the first edge, to their right), so that the node
    inherits a gap there. Columns after the last position of a node that is not a sample
    are not covered, as the end of its genome may simply be unrecorded. Each maximal run
    of columns, within a tree, in which any node
    has a gap becomes one site. Other columns become single-column sites if any node has
    a known base. There is one mutation above each node whose string at a site differs
    from its parent's (or, for a root, from the ancestral state, which is the string of
    the first root that has children), with unknown times. An isolated sample (e.g. at
    its positions with no parent, if it has no children) is missing data in a tree
    sequence unless it has a mutation, so it has one wherever any of its states are known.

    Positions with no parent and no mutation are missing data, and become ``N``. This can
    convert back as an ``N`` allele: e.g. at a node that is the root of a tree, an
    ``N`` that differs from the ancestral state needs a mutation, which
    :func:`from_tree_sequence` reads as an unknown base, not missing data.

    This is a first version, which requires the iedges and mutations to be on a single
    chromosome, with no inversions, and each node's positions to be in the same order as
    the columns of its ancestors (so no duplications or translocations). The time taken
    is proportional to the number of columns times the number of nodes.

    :param giglib.Graph gig: The GIG
    :return: A tree sequence
    :rtype: tskit.TreeSequence
    :raises ValueError: If the GIG cannot be converted
    """
    chromosome = _single_chromosome(gig)
    for ie in gig.iedges:
        if ie.is_inversion():
            raise ValueError(f"Cannot convert iedge {ie.id} to a tree sequence: inversions are not supported")
    keys = _column_keys(gig, chromosome)
    all_keys = sorted({key for node_keys in keys for key in node_keys})
    if len(all_keys) == 0:
        raise ValueError("Cannot convert a GIG with no positions to a tree sequence")
    column_of_key = {key: column for column, key in enumerate(all_keys)}
    columns = [[column_of_key[key] for key in node_keys] for node_keys in keys]
    num_columns = len(all_keys)

    tables = gig._tskit_tables(num_columns)
    _add_alignment_edges(gig, tables, columns, num_columns)
    tables.sort()
    tables.build_index()
    _add_alignment_sites(gig, tables, columns, num_columns, chromosome)
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


def _single_chromosome(gig):
    iedges, mutations = gig.tables.iedges, gig.tables.mutations
    chromosomes = set(iedges.child_chromosome.tolist()) | set(iedges.parent_chromosome.tolist())
    chromosomes |= set(mutations.chromosome.tolist())
    if len(chromosomes) > 1:
        raise ValueError("Cannot convert to a tree sequence: iedges and mutations must all be on one chromosome")
    return chromosomes.pop() if chromosomes else 0


def _column_keys(gig, chromosome):
    """
    Return, for each node, a list giving a sortable key for the column of each of its
    positions. A reference column ``x`` has the key ``(x,)``, and the ``i``-th inserted
    column of node ``u`` after the column with key ``k`` has the key ``k + (u, i)``.

    A run of positions with no parent at the start or end of a genome is only bounded on
    one side, so is taken to be colinear with the reference columns, unless the children
    show otherwise: if a child inherits the end of such a run, followed by a column
    that must come after it, but that the run would overlap, the run is bounded by that
    column (and similarly for the start of a run), and the keys are found again.
    """
    bounds = {}  # (node, run start): (lowest first reference column, highest end)
    while True:
        try:
            return _assign_column_keys(gig, chromosome, bounds)
        except _ColumnConflictError as conflict:
            if not conflict.tighten(bounds):
                u, x = conflict.node, conflict.position
                raise ValueError(
                    f"Cannot convert to a tree sequence: the positions of node {u} are not in the same "
                    f"order as the columns of its ancestors at position {x}: e.g. a duplication or "
                    "translocation, or ancestors that are roots that do not share coordinates"
                ) from None


class _ColumnConflictError(Exception):
    """
    The keys of positions ``position - 1`` and ``position`` of ``node`` are not in
    increasing order. ``origins`` gives, for each, the run of positions with no parent
    (in an ancestor) that the key came from, if that run is open at one end.
    """

    def __init__(self, node, position, keys, origins):
        self.node, self.position, self.keys, self.origins = node, position, keys, origins

    def tighten(self, bounds):
        """
        Tighten the bounds of an open run so that the two keys can be in order, returning
        False if neither key came from such a run (or its bounds can't be tightened)
        """
        (before, after), (before_origin, after_origin) = self.keys, self.origins
        if before_origin is not None and before_origin["open_end"]:
            # The run must end before the key after it
            end = after[0] if len(after) == 1 else after[0] + 1
            if end < before_origin["end"]:
                first, _ = bounds.get(before_origin["run"], (0, end))
                bounds[before_origin["run"]] = (first, end)
                return True
        if after_origin is not None and after_origin["open_start"]:
            # The run must start after the key before it
            first = before[0] + 1
            if first > after_origin["first"]:
                _, end = bounds.get(after_origin["run"], (first, float("inf")))
                bounds[after_origin["run"]] = (first, end)
                return True
        return False


def _assign_column_keys(gig, chromosome, bounds):
    num_nodes = len(gig.nodes)
    lengths = [gig.sequence_length(u, chromosome=chromosome) for u in range(num_nodes)]
    above = [list(gig.iedges_for_child(u, chromosome=chromosome)) for u in range(num_nodes)]
    # The limit for runs at the end of a genome (unless the children show otherwise)
    reference_length = max(lengths, default=0)
    keys = [None] * num_nodes
    origins = [None] * num_nodes  # the open run (if any) that each position's key came from
    # Parents are older than their children, so their keys are found first
    for u in sorted(range(num_nodes), key=lambda u: -gig.nodes[u].time):
        node_keys = [None] * lengths[u]
        node_origins = [None] * lengths[u]
        for ie in above[u]:
            for x in range(ie.child_left, ie.child_right):
                node_keys[x] = keys[ie.parent][x - ie.child_left + ie.parent_left]
                node_origins[x] = origins[ie.parent][x - ie.child_left + ie.parent_left]
        # Runs of positions with no parent
        x = 0
        while x < lengths[u]:
            if node_keys[x] is not None:
                x += 1
                continue
            start = x
            while x < lengths[u] and node_keys[x] is None:
                x += 1
            left = node_keys[start - 1] if start > 0 else None
            right = node_keys[x] if x < lengths[u] else None
            # The reference columns between the keys either side: (y,) with left < (y,)
            # < right. An inserted key (z, ...) sorts after (z,) but before (z + 1,).
            first = 0 if left is None else left[0] + 1
            if right is None:
                end = reference_length
            else:
                end = right[0] if len(right) == 1 else right[0] + 1
            lowest_first, highest_end = bounds.get((u, start), (first, end))
            first, end = max(first, lowest_first), min(end, highest_end)
            num_reference = max(0, min(x - start, end - first))
            for i in range(num_reference):
                node_keys[start + i] = (first + i,)
            base = node_keys[start + num_reference - 1] if num_reference > 0 else left
            if base is None:
                base = (first - 1,)  # (inserted before the first reference column)
            for i in range(x - start - num_reference):
                node_keys[start + num_reference + i] = (*base, u, i)
            if left is None or right is None:
                origin = {"run": (u, start), "open_start": left is None, "open_end": right is None}
                origin.update(first=first, end=end)
                for y in range(start, x):
                    node_origins[y] = origin
        for x in range(1, lengths[u]):
            if not node_keys[x - 1] < node_keys[x]:
                raise _ColumnConflictError(
                    u, x, (node_keys[x - 1], node_keys[x]), (node_origins[x - 1], node_origins[x])
                )
        keys[u], origins[u] = node_keys, node_origins
    return keys


def _add_alignment_edges(gig, tables, columns, num_columns):
    """
    Add an edge for each iedge, over the columns of its positions, and extend edges over
    the columns that the child lacks
    """
    for u in range(len(gig.nodes)):
        parent_of = [tskit.NULL] * num_columns
        own = [False] * num_columns
        for column in columns[u]:
            own[column] = True
        has_parents = False
        for ie in gig.iedges_for_child(u):
            for x in range(ie.child_left, ie.child_right):
                parent_of[columns[u][x]] = ie.parent
                has_parents = True
        if not has_parents:
            continue
        # Columns that the node lacks take the parent of the nearest column to their left
        # with a parent (or, before the first, to their right). Columns after the node's
        # last position are only lacked by a sample, whose genome is complete: for other
        # nodes, the end of the genome may simply be unrecorded.
        end = num_columns if gig.nodes[u].is_sample() else columns[u][-1] + 1
        last = tskit.NULL
        for column in range(end):
            if parent_of[column] != tskit.NULL:
                last = parent_of[column]
            elif not own[column]:
                parent_of[column] = last
        first = next(p for p in parent_of if p != tskit.NULL)
        for column in range(num_columns):
            if parent_of[column] != tskit.NULL:
                break
            if not own[column]:
                parent_of[column] = first
        # Merge runs of columns with the same parent into edges
        start = 0
        for column in range(1, num_columns + 1):
            if column == num_columns or parent_of[column] != parent_of[start]:
                if parent_of[start] != tskit.NULL:
                    tables.edges.add_row(left=start, right=column, parent=parent_of[start], child=u)
                start = column


def _add_alignment_sites(gig, tables, columns, num_columns, chromosome):
    """
    Add sites and mutations, given the edges, from each node's state in each column
    """
    states = []
    for u in range(len(gig.nodes)):
        row = [GAP] * num_columns
        length = len(columns[u])
        if length > 0:
            haplotype = gig.haplotype(u, chromosome=chromosome, right=length, missing_data_character=UNKNOWN)
            for column, char in zip(columns[u], haplotype):
                row[column] = char
        states.append(row)
    ts = tables.tree_sequence()
    for tree in ts.trees():
        nodes = list(tree.nodes(order="preorder"))
        roots = [u for u in tree.roots if tree.num_children(u) > 0] or list(tree.roots)
        left, right = int(tree.interval.left), int(tree.interval.right)
        column = left
        while column < right:
            if any(states[u][column] == GAP for u in nodes):
                end = column
                while end < right and any(states[u][end] == GAP for u in nodes):
                    end += 1
            elif any(states[u][column] != UNKNOWN for u in nodes):
                end = column + 1
            else:
                column += 1  # every node's state is unknown: no site needed
                continue
            strings = {u: "".join(states[u][column:end]) for u in nodes}
            ancestral_state = strings[roots[0]]
            site = tables.sites.add_row(position=column, ancestral_state=ancestral_state)
            for u in nodes:
                parent = tree.parent(u)
                if parent == tskit.NULL and tree.num_children(u) == 0:
                    # An isolated sample is missing data unless it has a mutation, whatever
                    # the ancestral state, so needs one unless its states are all unknown
                    if set(strings[u]) != {UNKNOWN}:
                        tables.mutations.add_row(site=site, node=u, derived_state=strings[u])
                    continue
                previous = ancestral_state if parent == tskit.NULL else strings[parent]
                if strings[u] != previous:
                    tables.mutations.add_row(site=site, node=u, derived_state=strings[u])
            column = end
