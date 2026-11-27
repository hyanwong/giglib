"""
Tests for Tables.simplify (and Graph.simplify): simplification keeps the samples and the
branch points of the position tree, so the ancestry of the samples is unchanged (see the
simplification design note)
"""

import msprime
import numpy as np
import pytest
import tskit

import giglib as gigl
from giglib.tools.simulation.one_break_inversions import DTWF_one_break_no_rec_inversions_slow_sim
from giglib.tools.simulation.single_origin import sim_single_origin
from tests.gigutil import homologous_positions


def clonal_tables_with_rearrangements(seed, generations=12, size=5):
    """
    A small clonal population, from one genome of 20 positions, in which each child has an
    inversion, a tandem duplication, or a deletion, of up to 5 positions (or none). The
    single genome at the start has a random base at every position, and each child gains a
    mutation, so the haplotypes depend on mutations passed through (often chained)
    inversions and copies.
    """
    rng = np.random.default_rng(seed)
    tables = gigl.Tables()
    root = tables.nodes.add_row(generations)
    lengths = {root: 20}
    for x in range(20):
        tables.mutations.add_row(root, x, str(rng.choice(list("ACGT"))))
    population = [root]
    for time in range(generations - 1, -1, -1):
        children = []
        for _ in range(size):
            parent = population[rng.integers(len(population))]
            n = lengths[parent]
            child = tables.nodes.add_row(time, flags=gigl.NODE_IS_SAMPLE if time == 0 else 0)
            length = int(rng.integers(1, min(5, n) + 1))
            start = int(rng.integers(0, n - length + 1))
            end = start + length
            event = rng.choice(["inversion", "duplication", "deletion", "none"])
            # Each piece is (child_left, child_right, parent_left, parent_right)
            if event == "inversion":
                pieces = [(0, start, 0, start), (start, end, end, start), (end, n, end, n)]
                lengths[child] = n
            elif event == "duplication":
                pieces = [(0, end, 0, end), (end, end + length, start, end), (end + length, n + length, end, n)]
                lengths[child] = n + length
            elif event == "deletion" and length < n:
                pieces = [(0, start, 0, start), (start, n - length, end, n)]
                lengths[child] = n - length
            else:
                pieces = [(0, n, 0, n)]
                lengths[child] = n
            for child_left, child_right, parent_left, parent_right in pieces:
                if child_right > child_left:
                    tables.iedges.add_row(
                        child,
                        parent,
                        child_left=child_left,
                        child_right=child_right,
                        parent_left=parent_left,
                        parent_right=parent_right,
                    )
            tables.mutations.add_row(child, int(rng.integers(lengths[child])), str(rng.choice(list("ACGT"))))
            children.append(child)
        population = children
    tables.sort()
    tables.validate()
    return tables


def check_homology_unchanged(tables, simplified, node_map, samples, mode):
    """
    find_homology gives the same results before and after simplifying, for pairs of the
    samples, once the nodes are relabelled
    """
    for u, v in [(samples[0], samples[1]), (samples[1], samples[2]), (samples[0], samples[3])]:
        before = {
            (x, y, int(node_map[mrca]), z)
            for x, y, mrca, z in homologous_positions(tables.find_homology(u, v, mode=mode))
        }
        after = homologous_positions(simplified.find_homology(int(node_map[u]), int(node_map[v]), mode=mode))
        assert after == before


class TestWithoutStructuralVariants:
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_same_as_tskit(self, seed):
        # 50 samples, simplified to 20 of them, keeping unary nodes: simplifying the GIG
        # gives the same as tskit's simplify (without keeping unary nodes)
        ts = msprime.sim_ancestry(25, sequence_length=1000, recombination_rate=1e-3, random_seed=seed)
        ts = msprime.sim_mutations(ts, rate=1e-3, random_seed=seed)
        subset = np.arange(20)
        ts = ts.simplify(subset, keep_unary=True)
        gig = gigl.Graph.from_tree_sequence(ts)
        simplified = gig.simplify()
        assert len(simplified.nodes) < len(gig.nodes)
        ours = simplified.to_tree_sequence().dump_tables()
        expected = ts.simplify(keep_input_roots=True).dump_tables()
        assert ours.nodes.time.tolist() == expected.nodes.time.tolist()
        assert ours.nodes.flags.tolist() == expected.nodes.flags.tolist()
        ours.edges.squash()
        expected.edges.squash()
        assert ours.edges == expected.edges
        for u in subset:
            assert simplified.haplotype(u) == gig.haplotype(u)

    def test_forward_simulation(self):
        # A forward simulation, without structural variants, also has many unary nodes. It
        # has not fully coalesced, so the roots of lineages are kept (as with tskit's
        # keep_input_roots), but numbered in a different order: compare the edges using
        # the original node IDs
        gig = DTWF_one_break_no_rec_inversions_slow_sim().run(num_diploids=10, seq_lens=[200], gens=10, random_seed=1)
        samples = gig.sample_nodes(time=0).ids
        tables = gig.tables.copy()
        node_map = tables.simplify(samples)
        ts, ts_node_map = gig.to_tree_sequence().simplify(
            samples, filter_individuals=False, keep_input_roots=True, map_nodes=True
        )
        ours = tables.graph().to_tree_sequence().dump_tables()
        expected = ts.dump_tables()
        ours.edges.squash()
        expected.edges.squash()
        ours_original = {new: old for old, new in enumerate(node_map) if new != gigl.NULL}
        expected_original = {new: old for old, new in enumerate(ts_node_map) if new != tskit.NULL}
        assert {(e.left, e.right, ours_original[e.parent], ours_original[e.child]) for e in ours.edges} == {
            (e.left, e.right, expected_original[e.parent], expected_original[e.child]) for e in expected.edges
        }

    def test_already_simplified(self):
        # msprime's tree sequences are already simplified, so nothing is removed
        ts = msprime.sim_ancestry(5, sequence_length=100, recombination_rate=0.01, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.02, random_seed=1)
        gig = gigl.Graph.from_tree_sequence(ts)
        simplified = gig.simplify()
        assert len(simplified.nodes) == len(gig.nodes)
        assert len(simplified.iedges) == len(gig.iedges)
        for u in gig.sample_nodes().ids:
            assert simplified.haplotype(u) == gig.haplotype(u)


class TestWithRearrangements:
    @pytest.mark.parametrize("seed", range(10))
    def test_haplotypes_unchanged(self, seed):
        tables = clonal_tables_with_rearrangements(seed)
        samples = [int(u) for u in tables.sample_nodes()]
        gig = tables.copy().graph()
        simplified = tables.copy()
        node_map = simplified.simplify(samples)
        simplified = simplified.graph()
        assert len(simplified.nodes) < len(gig.nodes)
        for u in samples:
            assert simplified.haplotype(int(node_map[u])) == gig.haplotype(u)

    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("mode", ["first_mrca", "closest_homologues", "all"])
    def test_homology_unchanged(self, seed, mode):
        tables = clonal_tables_with_rearrangements(seed)
        samples = [int(u) for u in tables.sample_nodes()]
        simplified = tables.copy()
        node_map = simplified.simplify(samples)
        check_homology_unchanged(tables, simplified, node_map, samples, mode)

    @pytest.mark.skip(reason="needs find_homology of a genome with itself (#236)")
    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("mode", ["first_mrca", "closest_homologues", "all"])
    def test_homology_with_itself_unchanged(self, seed, mode):
        # The paralogous homology of a genome with itself is unchanged too
        tables = clonal_tables_with_rearrangements(seed)
        samples = [int(u) for u in tables.sample_nodes()]
        simplified = tables.copy()
        node_map = simplified.simplify(samples)
        for u in samples[:3]:
            before = {
                (x, y, int(node_map[mrca]), z)
                for x, y, mrca, z in homologous_positions(tables.find_homology(u, u, mode=mode))
            }
            after = homologous_positions(simplified.find_homology(int(node_map[u]), int(node_map[u]), mode=mode))
            assert after == before

    @pytest.mark.parametrize("mode", ["first_mrca", "closest_homologues", "all"])
    def test_single_origin(self, mode):
        # Genomes grown from a single base pair: everything is homologous, through copies
        gig = sim_single_origin(10, 150, duplication_rate=0.1, deletion_rate=0.03, mean_length=5, random_seed=1)
        samples = [int(u) for u in gig.sample_nodes().ids]
        tables = gig.tables.copy()
        simplified = gig.tables.copy()
        node_map = simplified.simplify(samples)
        assert len(simplified.nodes) < len(tables.nodes) / 10
        check_homology_unchanged(tables, simplified, node_map, samples, mode)


class TestSimplify:
    def test_twice_is_the_same_as_once(self):
        tables = clonal_tables_with_rearrangements(1)
        tables.simplify()
        once = tables.copy()
        node_map = tables.simplify()
        assert list(node_map) == list(range(len(once.nodes)))
        assert tables == once

    def test_samples_first_in_order(self):
        tables = clonal_tables_with_rearrangements(1)
        samples = [int(u) for u in tables.sample_nodes()][::-1]
        node_map = tables.simplify(samples)
        assert [int(node_map[u]) for u in samples] == list(range(len(samples)))
        assert list(tables.sample_nodes()) == list(range(len(samples)))

    def test_tandem_duplication_keeps_the_parent(self):
        # The copies of a tandem duplication merge in the parent, which is kept, but the
        # duplicated genome (if not a sample) is bypassed
        tables = gigl.Tables()
        tables.nodes.add_row(2)  # 0: the parent
        tables.nodes.add_row(1)  # 1: with a tandem duplication of the parent's 0-10
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)  # 2: a sample, inheriting 1
        tables.iedges.add_row(1, 0, child_left=0, child_right=10, parent_left=0, parent_right=10)
        tables.iedges.add_row(1, 0, child_left=10, child_right=20, parent_left=0, parent_right=10)
        tables.iedges.add_row(2, 1, child_right=20)
        tables.sort()
        tables.validate()
        node_map = tables.simplify()
        assert list(node_map) == [1, gigl.NULL, 0]
        assert [
            (ie.child, ie.parent, ie.child_left, ie.child_right, ie.parent_left, ie.parent_right)
            for ie in tables.iedges
        ] == [
            (0, 1, 0, 10, 0, 10),
            (0, 1, 10, 20, 0, 10),
        ]

    def test_graph(self):
        tables = clonal_tables_with_rearrangements(2)
        gig = tables.graph()
        samples = gig.sample_nodes()
        simplified = gig.simplify(samples)  # Node objects are accepted
        for i, u in enumerate(samples):
            assert simplified.haplotype(i) == gig.haplotype(u)

    def test_distinct_samples(self):
        tables = clonal_tables_with_rearrangements(1)
        with pytest.raises(ValueError, match="distinct"):
            tables.simplify([20, 20])

    @pytest.mark.parametrize("sample", [-1, 1000])
    def test_samples_are_nodes(self, sample):
        tables = clonal_tables_with_rearrangements(1)
        with pytest.raises(ValueError, match="not a node ID"):
            tables.simplify([sample])

    @pytest.mark.parametrize("positions", [[], [0, 1, 2, 3, 4]])
    def test_missing_data_at_the_end_is_kept(self, positions):
        # A sample inherits 0-10 from a root with mutations at these positions only, so
        # the rest is missing data. The root is kept, as otherwise the sample's genome
        # would only be known up to its last mutation
        tables = gigl.Tables()
        tables.nodes.add_row(1)  # 0: the root
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)  # 1: the sample
        tables.iedges.add_row(1, 0, child_left=0, child_right=10, parent_left=0, parent_right=10)
        for x in positions:
            tables.mutations.add_row(0, x, "A")
        tables.sort()
        tables.validate()
        haplotype = tables.copy().graph().haplotype(1)
        tables.simplify()
        gig = tables.graph()
        assert gig.sequence_length(0) == 10
        assert gig.haplotype(0) == haplotype
