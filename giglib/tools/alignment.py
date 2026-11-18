"""
Convert between GIGs with insertions and deletions, and tree sequences whose
coordinates are the columns of a multiple sequence alignment.

This module is experimental: see :ref:`sec_tools`. Discussion: :issue:`139`.

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
from mutations above the child. Sites must not overlap, must not extend beyond the end
of the sequence, and the columns of a site must all be in the same tree (a site's
mutations belong to the tree at its position).

**GIG to tree sequence** (planned). The reference coordinates are those shared by the
roots, and the ancestral state of each column is given by the first mutation above the
roots there (as in :meth:`giglib.Graph.to_tree_sequence`). All roots must agree.

- *Columns.* Working from the roots down, each position inherited through an iedge takes
  its parent's column. Each inserted position (with no parent) gets a new column after
  the column of the node's previous position. Columns are ordered by sortable keys: a
  reference position ``x`` has key ``(x,)``, and the ``i``-th base inserted by node
  ``u`` after key ``k`` has key ``k + (u, i)``. So nested insertions sort within their
  parent insertion, and independent insertions at the same place are ordered by node
  ID. Each node's positions must map to strictly increasing columns.
- *Edges.* Each iedge becomes an edge over its columns. Gaps between a child's iedges
  (deletions, and other nodes' insertions) are filled by extending the edge to their
  left, so a child inherits ``-`` where its parent lacks a column. For an insertion at a
  recombination breakpoint, this choice of parent is arbitrary but harmless, as neither
  parent has those columns.
- *Sites and mutations.* Each maximal run of columns where any node has a gap becomes
  one site, split at tree boundaries. Other columns with mutations become single-column
  sites. There is one mutation above each node whose string at a site differs from its
  parent's in the local tree (or, for a root, from the ancestral string).

**Round trips, and what is lost.** Converting a tree sequence to a GIG and back should
give the same alignments, but not necessarily the same sites: multi-column sites without
gaps (blocks of SNPs) would be split into single columns, and anchor bases would be
dropped. Converting a GIG to a tree sequence and back should give the same haplotypes.
For a GIG with rearrangements that are not colinear, a tree sequence can keep the
sequences, but not the homology between bases:

- an in-place inversion could become a reverse-complemented allele over the same columns
  (``AACG`` to ``CGTT``), which converts back to substitutions;
- a duplicated copy could become inserted columns with known bases, which converts back
  to an insertion unrelated to the original copy;
- a translocation within a chromosome could become a deletion and an insertion;
- several chromosomes could become one tree sequence per chromosome.

The plan is for conversion to a tree sequence to raise an error for rearrangements that
are not colinear by default, so that no information is silently lost, with an option
(e.g. ``rearrangements="as_indels"``) to encode them as above.
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
    state.
    """
    mutation_state = {}
    for mutation in site.mutations:  # mutations are ordered so that later ones are more recent
        mutation_state[mutation.node] = mutation.derived_state
    unknown = UNKNOWN * len(site.ancestral_state)
    states = [mutation_state.get(u, unknown) for u in range(num_nodes)]
    for u in tree.nodes(order="preorder"):
        parent = tree.parent(u)
        if parent != tskit.NULL:
            states[u] = mutation_state.get(u, states[parent])
        elif tree.num_children(u) > 0:  # a root that is not isolated
            states[u] = mutation_state.get(u, site.ancestral_state)
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
