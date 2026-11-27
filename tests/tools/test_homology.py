"""
Tests for giglib.tools.homology: homology blocks between two genomes, and chains of them
"""

import msprime
import numpy as np
import pytest

import giglib as gigl
from giglib.tools import homology
from tests.gigutil import same_homology


def two_samples(iedges_u, iedges_v, *, root_length=None):
    """
    Tables with a root (node 0) and two samples (1 and 2), whose iedges from the root are
    given as lists of (child_left, child_right, parent_left, parent_right)
    """
    tables = gigl.Tables()
    tables.nodes.add_row(1)
    for _ in range(2):
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
    for child, iedges in ((1, iedges_u), (2, iedges_v)):
        for cl, cr, pl, pr in iedges:
            tables.iedges.add_row(child, 0, child_left=cl, child_right=cr, parent_left=pl, parent_right=pr)
    tables.sort()
    tables.validate()
    return tables


def spans(chain):
    return [(b.u_left, b.u_right, b.v_left, b.v_right) for b in chain]


class TestHomologyBlocks:
    def test_tree_sequence_is_one_block(self):
        # Without structural variation, all the homology is one block, however many trees
        ts = msprime.sim_ancestry(4, sequence_length=200, recombination_rate=0.02, random_seed=1)
        assert ts.num_trees > 1
        tables = gigl.Tables.from_tree_sequence(ts)
        assert homology.homology_blocks(tables, 0, 5) == [homology.Block(0, 200, 0, 0, 200, 0)]

    def test_shift(self):
        # v has an extra 5 positions (an insertion) at 10, so the map shifts there
        tables = two_samples([(0, 30, 0, 30)], [(0, 10, 0, 10), (15, 35, 10, 30)])
        blocks = homology.homology_blocks(tables, 1, 2)
        assert spans(blocks) == [(0, 10, 0, 10), (10, 30, 15, 35)]
        assert blocks[1].map == gigl.PositionMap(1, 5)

    def test_inversion(self):
        tables = two_samples([(0, 30, 0, 30)], [(0, 10, 0, 10), (10, 20, 20, 10), (20, 30, 20, 30)])
        blocks = homology.homology_blocks(tables, 1, 2)
        assert spans(blocks) == [(0, 10, 0, 10), (10, 20, 20, 10), (20, 30, 20, 30)]
        assert [b.is_reversed for b in blocks] == [False, True, False]

    def test_copy_number_variant(self):
        # u = A B B C, v = A B C (with B = positions 10-20 of the root): both copies of B in
        # u are homologous to v's B
        tables = two_samples([(0, 20, 0, 20), (20, 30, 10, 20), (30, 40, 20, 30)], [(0, 30, 0, 30)])
        blocks = homology.homology_blocks(tables, 1, 2)
        # The second copy of B continues into C with the same map, so they are one block
        assert spans(blocks) == [(0, 20, 0, 20), (20, 40, 10, 30)]


class TestChains:
    def test_small_shift_joins_a_chain(self):
        tables = two_samples([(0, 30, 0, 30)], [(0, 10, 0, 10), (15, 35, 10, 30)])
        blocks = homology.homology_blocks(tables, 1, 2)
        chains = homology.chains(blocks, max_gap=5)
        assert len(chains) == 1
        assert homology.chain_length(chains[0]) == 30

    def test_large_shift_breaks_a_chain(self):
        tables = two_samples([(0, 30, 0, 30)], [(0, 10, 0, 10), (15, 35, 10, 30)])
        blocks = homology.homology_blocks(tables, 1, 2)
        chains = homology.chains(blocks, max_gap=4)
        assert sorted(homology.chain_length(c) for c in chains) == [10, 20]

    def test_inversion_breaks_a_chain(self):
        tables = two_samples([(0, 30, 0, 30)], [(0, 10, 0, 10), (10, 20, 20, 10), (20, 30, 20, 30)])
        chains = homology.chains(homology.homology_blocks(tables, 1, 2), max_gap=100)
        # The two forward blocks can be chained across the inversion (a gap of 10), but
        # the reversed block can't join them
        assert [spans(c) for c in chains] == [[(0, 10, 0, 10), (20, 30, 20, 30)], [(10, 20, 20, 10)]]

    def test_copy_number_variant_is_trimmed(self):
        # The allelic chain is A, the first copy of B, (a gap for the second copy), C: the
        # block for the second copy of B and C is trimmed to C. The second copy of B,
        # paired with v's B, is left as a chain of its own (the unequal pairing).
        tables = two_samples([(0, 20, 0, 20), (20, 30, 10, 20), (30, 40, 20, 30)], [(0, 30, 0, 30)])
        chains = homology.chains(homology.homology_blocks(tables, 1, 2), max_gap=10)
        assert [spans(c) for c in chains] == [[(0, 20, 0, 20), (30, 40, 20, 30)], [(20, 30, 10, 20)]]

    def test_every_position_in_one_chain(self):
        tables = two_samples([(0, 20, 0, 20), (20, 30, 10, 20), (30, 40, 20, 30)], [(0, 30, 0, 30)])
        blocks = homology.homology_blocks(tables, 1, 2)
        chains = homology.chains(blocks, max_gap=10)
        assert sum(homology.chain_length(c) for c in chains) == sum(b.length for b in blocks)

    def test_no_blocks(self):
        assert homology.chains([], max_gap=10) == []

    @pytest.mark.parametrize("max_gap", [0, 5, 100])
    def test_single_block(self, max_gap):
        block = homology.Block(0, 10, 0, 5, 15, 0)
        assert homology.chains([block], max_gap=max_gap) == [[block]]


def cnv_tables():
    """
    An old ancestor (0) has A (0-30), B (30-40), C (40-80). B was duplicated in the common
    ancestor (1) of genomes 2 and 3 (A, B1, B2, C), and 3 inherits it unchanged, while in 2
    the second copy of B was duplicated again (A, B1, B2, B3, C)
    """
    tables = gigl.Tables()
    tables.nodes.add_row(10)
    tables.nodes.add_row(1)
    tables.nodes.add_row(0)
    tables.nodes.add_row(0)
    for child, pieces in [
        (1, [(0, 40, 0, 0), (40, 50, 0, 30), (50, 90, 0, 40)]),
        (3, [(0, 90, 1, 0)]),
        (2, [(0, 50, 1, 0), (50, 60, 1, 40), (60, 100, 1, 50)]),
    ]:
        for cl, cr, parent, pl in pieces:
            tables.iedges.add_row(
                child, parent, child_left=cl, child_right=cr, parent_left=pl, parent_right=pl + cr - cl
            )
    tables.sort()
    tables.validate()
    return tables


class TestFindAllHomology:
    """
    The homology found by find_homology with mode="all", as used for the plots
    """

    def test_copies_in_a_common_ancestor(self):
        # As well as the recent homology, the copies made in the common ancestor are
        # homologous to each other through the old ancestor (which mode "first_mrca" misses)
        found = cnv_tables().find_homology(2, 3, mode="all")
        assert same_homology(
            found,
            [
                # Allelic homology, through the common ancestor (node 1), at the same
                # positions as in it (as genome 3 inherits it unchanged)
                ((0, 50, 0), (0, 50, 0), 1, (0, 50, 0)),
                ((50, 100, 0), (40, 90, 0), 1, (40, 90, 0)),
                # The copies of B (30-40 of node 0) in each genome, through node 0
                ((30, 40, 0), (40, 50, 0), 0, (30, 40, 0)),
                ((40, 50, 0), (30, 40, 0), 0, (30, 40, 0)),
                ((50, 60, 0), (30, 40, 0), 0, (30, 40, 0)),
            ],
        )

    def test_time_cutoff(self):
        tables = cnv_tables()
        found = tables.find_homology(2, 3, mode="all", time_cutoff=5)
        assert {h.mrca for h in found} == {1}

    def test_plot_shows_old_homology(self):
        ax = homology.plot(cnv_tables(), 2, 3)
        (collection,) = ax.collections
        assert set(collection.get_array()) == {1, 10}

    def test_plot_closest_homologues(self):
        # Each copy's closest homologue is through the recent common ancestor (time 1), so
        # the older homology between the copies (time 10) isn't shown
        ax = homology.plot(cnv_tables(), 2, 3, mode="closest_homologues")
        (collection,) = ax.collections
        assert set(collection.get_array()) == {1}


class TestRandomHomologousPositions:
    def test_both_positions_chosen(self):
        # Two positions, each homologous to the same position in v
        ts = msprime.sim_ancestry(2, sequence_length=2, random_seed=1)
        tables = gigl.Tables.from_tree_sequence(ts)
        shared = tables.find_homology(0, 1)
        rng = np.random.default_rng(1)
        chosen = set()
        for _ in range(20):
            pos = homology.random_homologous_positions(shared, rng)
            assert isinstance(pos.u, gigl.Position)
            assert isinstance(pos.v, gigl.Position)
            assert not pos.opposite_orientations
            assert pos.u == pos.v
            chosen.add(pos.u)
        assert chosen == {(0, 0), (1, 0)}

    def test_chromosomes_and_orientation(self):
        # u (node 0) and v (node 1) inherit from node 2: u on chromosome 1, and v on
        # chromosome 0, inverted
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1)
        tables.iedges.add_row(0, 2, child_right=10, child_chromosome=1, parent_chromosome=1)
        tables.iedges.add_row(
            1, 2, child_left=0, child_right=10, parent_left=10, parent_right=0, child_chromosome=0, parent_chromosome=1
        )
        tables.sort()
        tables.validate()
        shared = tables.find_homology(0, 1)
        rng = np.random.default_rng(1)
        for _ in range(10):
            pos = homology.random_homologous_positions(shared, rng)
            assert pos.u.chromosome == 1
            assert pos.v.chromosome == 0
            assert pos.opposite_orientations
            assert pos.u.position + pos.v.position == 9

    def test_needs_first_mrca(self):
        # In mode "all", u's copy at 30-40 pairs with v's copy at 40-50 through node 0,
        # but not with v's copy at 30-40 (whose MRCA with it is node 1)
        shared = cnv_tables().find_homology(2, 3, mode="all")
        with pytest.raises(ValueError, match="first_mrca"):
            homology.random_homologous_positions(shared, np.random.default_rng(1))

    def test_copies(self):
        # Positions 40-50 of the common ancestor (node 1) have two copies in genome 2
        # (40-50 and 50-60), and one in genome 3 (40-50): either copy can be chosen
        shared = cnv_tables().find_homology(2, 3)
        rng = np.random.default_rng(1)
        offsets = set()
        for _ in range(200):
            pos = homology.random_homologous_positions(shared, rng)
            if 40 <= pos.v.position < 50:
                offsets.add(pos.u.position - pos.v.position)
        assert offsets == {0, 10}


class TestPlot:
    def test_segments_coloured_by_tmrca(self):
        # u has a copy of a segment from an older ancestor (node 0), and otherwise
        # inherits from a recent one (node 1), as does v
        tables = gigl.Tables()
        tables.nodes.add_row(10)
        tables.nodes.add_row(1)
        tables.nodes.add_row(0)
        tables.nodes.add_row(0)
        tables.iedges.add_row(1, 0, child_right=20)
        tables.iedges.add_row(2, 1, child_right=20)
        tables.iedges.add_row(2, 0, child_left=20, child_right=25, parent_left=30, parent_right=35)
        tables.iedges.add_row(3, 1, child_right=20)
        tables.iedges.add_row(3, 0, child_left=20, child_right=25, parent_left=30, parent_right=35)
        tables.sort()
        tables.validate()
        ax = homology.plot(tables, 2, 3)
        (collection,) = ax.collections
        lines = sorted((tuple(map(tuple, seg)), t) for seg, t in zip(collection.get_segments(), collection.get_array()))
        assert lines == [(((0, 0), (20, 20)), 1), (((20, 20), (25, 25)), 10)]


class TestReviewCases:
    def test_inverted_segment_with_two_copies(self):
        # u inherits the root's positions 0-10 inverted, and v has two (forward) copies
        # of them, so u's segment matches both copies, through the same inverted map
        tables = two_samples([(0, 10, 10, 0)], [(0, 10, 0, 10), (10, 20, 0, 10)])
        blocks = homology.homology_blocks(tables, 1, 2)
        assert spans(blocks) == [(0, 10, 10, 0), (0, 10, 20, 10)]
        assert all(b.is_reversed for b in blocks)

    def test_predecessor_that_starts_later(self):
        # A starts after B, but ends before it, so can precede it in a chain: the best
        # chain is C, A, then B (trimmed where it overlaps A), with 40 positions
        c = homology.Block(0, 10, 0, 0, 10, 0)
        b = homology.Block(5, 40, 0, 15, 50, 0)
        a = homology.Block(10, 20, 0, 15, 25, 0)
        chains = homology.chains([c, b, a], max_gap=5)
        assert spans(chains[0]) == [(0, 10, 0, 10), (10, 20, 15, 25), (20, 40, 30, 50)]
        assert homology.chain_length(chains[0]) == 40

    def test_parent_not_older(self):
        # A node that is its own parent would make the tracing loop forever: the
        # requirements for tracing are checked instead
        tables = gigl.Tables()
        tables.nodes.add_row(0)
        tables.nodes.add_row(0)
        tables.iedges.add_row(0, 0, child_right=10)
        tables.iedges.add_row(1, 0, child_right=10)
        tables.iedges.set_bitflag(
            gigl.ValidFlags.IEDGES_FOR_CHILD_ADJACENT | gigl.ValidFlags.IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC
        )
        with pytest.raises(ValueError, match="validity flags"):
            tables.find_homology(0, 1)

    def test_axes_cover_a_root(self):
        # A root (0) has no iedges above it, but the axes cover its homology with a child
        tables = gigl.Tables()
        tables.nodes.add_row(1)
        tables.nodes.add_row(0)
        tables.iedges.add_row(1, 0, child_right=100, parent_left=100, parent_right=0)
        tables.sort()
        tables.validate(gigl.ValidFlags.IEDGES_FOR_TRACING)  # (only these flags)
        ax = homology.plot(tables, 0, 1)
        assert ax.get_xlim() == (0, 100)
        assert ax.get_ylim() == (0, 100)
