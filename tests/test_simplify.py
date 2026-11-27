"""
Tests for Tables.simplify (and Graph.simplify): simplification keeps the samples and the
branch points of the position tree, so the ancestry of the samples is unchanged (see the
simplification design note)
"""

import itertools

import msprime
import numpy as np
import pytest
import tskit

import giglib as gigl
from giglib.tools.simulation.microsatellites_forward import DTWF_microsatellite_sim
from giglib.tools.simulation.one_break_inversions import DTWF_one_break_no_rec_inversions_slow_sim
from giglib.tools.simulation.single_origin import sim_single_origin
from tests.gigutil import homologous_positions, random_tables


def check_homology_unchanged(tables, simplified, node_map, samples, mode):
    """
    find_homology gives the same results before and after simplifying, for each pair of
    the first four samples, once the nodes are relabelled
    """
    for u, v in itertools.combinations(samples[:4], 2):
        before = {
            (x, y, int(node_map[mrca]), z)
            for x, y, mrca, z in homologous_positions(tables.find_homology(u, v, mode=mode))
        }
        after = homologous_positions(simplified.find_homology(int(node_map[u]), int(node_map[v]), mode=mode))
        assert after == before


def length_from_ancestry(tables, u, chromosome):
    """
    The length of a chromosome of u as known from u's own ancestry: from the iedges above
    u and the mutations above it, but not from the iedges to its children
    """
    ends = [tables.iedges[i].child_right for i in tables.iedges.ids_for_child(u, chromosome=chromosome)]
    for i in tables.mutations.ids_for_node(u):
        if tables.mutations[i].chromosome == chromosome:
            ends.append(tables.mutations[i].position + 1)
    return max(ends, default=0)


def check_invariants(gig, samples):
    """
    The invariants of simplify() (see its docstring), for these samples of a GIG
    """
    tables = gig.tables.copy()
    node_map = tables.simplify(samples)
    simplified = tables.graph()
    for u in samples:
        for chromosome in gig.chromosomes(u):
            before = gig.haplotype(u, chromosome=chromosome)
            after = simplified.haplotype(int(node_map[u]), chromosome=chromosome)
            # The genome is known at least as far as its own ancestry shows. Beyond that, it
            # can be shorter, if it was only known from children that are removed (a GIG
            # has no record of genome lengths, #158), but that is all missing data
            assert len(after) >= length_from_ancestry(gig.tables, u, chromosome)
            assert before == after + "N" * (len(before) - len(after))
    for mode in ["first_mrca", "closest_homologues", "all"]:
        check_homology_unchanged(gig.tables, tables, node_map, samples, mode)
    again = tables.copy()
    again.simplify()
    assert again == tables


def some_nodes(gig, seed):
    """
    Up to six nodes of a GIG, chosen at random: samples of mixed times, some of them
    ancestors of others or roots
    """
    rng = np.random.default_rng(seed)
    return [int(u) for u in rng.choice(len(gig.nodes), size=min(6, len(gig.nodes)), replace=False)]


class TestInvariants:
    """
    The invariants of simplify(), on the hand-built examples and the simulators, using the
    samples and also nodes of mixed times as samples
    """

    @pytest.mark.parametrize(
        "name",
        [
            "trivial_gig",
            "extended_inversion_gig",
            "double_inversion_gig",
            "multi_chromosome_gig",
            "all_sv_types_no_re_gig",
            "all_sv_types_1re_gig",
            "all_sv_types_2re_gig",
            "inverted_duplicate_gig",
            "inverted_duplicate_with_missing_gig",
            "unequal_crossover_gig",
        ],
    )
    def test_examples(self, name, request):
        gig = request.getfixturevalue(name)
        check_invariants(gig, [int(u) for u in gig.sample_nodes().ids])
        for seed in range(3):
            check_invariants(gig, some_nodes(gig, seed))

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("recombination", [0, 0.3])
    def test_random_tables(self, seed, recombination):
        gig = random_tables(seed, recombination=recombination).graph()
        check_invariants(gig, [int(u) for u in gig.sample_nodes().ids])
        check_invariants(gig, some_nodes(gig, seed))

    @pytest.mark.parametrize("seed", range(1, 6))
    def test_microsatellites(self, seed):
        loci = ((20, 2, 5), (70, 3, 4), (120, 2, 6))
        sim = DTWF_microsatellite_sim(loci, mutation_rate=0.2, min_chain_length=100, max_gap=20)
        gig = sim.run(num_diploids=5, seq_lens=[200], gens=8, random_seed=seed)
        check_invariants(gig, [int(u) for u in gig.sample_nodes(time=0).ids])
        check_invariants(gig, some_nodes(gig, seed))

    @pytest.mark.parametrize("seed", range(1, 4))
    def test_inversions_two_chromosomes(self, seed):
        gig = DTWF_one_break_no_rec_inversions_slow_sim().run(
            num_diploids=6, seq_lens=[100, 50], gens=10, random_seed=seed
        )
        check_invariants(gig, [int(u) for u in gig.sample_nodes(time=0).ids])
        check_invariants(gig, some_nodes(gig, seed))

    @pytest.mark.parametrize("seed", range(1, 4))
    def test_single_origin(self, seed):
        gig = sim_single_origin(6, 40, duplication_rate=0.1, deletion_rate=0.03, mean_length=5, random_seed=seed)
        check_invariants(gig, [int(u) for u in gig.sample_nodes().ids])
        check_invariants(gig, some_nodes(gig, seed))

    def test_length_known_only_from_removed_children(self):
        # A sample with no mutations, known to have 10 positions only from its child,
        # which is removed: its genome is known only as far as the other child shows
        tables = gigl.Tables()
        tables.nodes.add_row(1, flags=gigl.NODE_IS_SAMPLE)  # 0: the sample
        tables.nodes.add_row(0)  # 1: removed
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)  # 2: another sample
        tables.iedges.add_row(1, 0, child_left=0, child_right=10, parent_left=0, parent_right=10)
        tables.iedges.add_row(2, 0, child_left=0, child_right=6, parent_left=0, parent_right=6)
        tables.sort()
        tables.validate()
        assert tables.copy().graph().sequence_length(0) == 10
        tables.simplify()
        assert tables.graph().sequence_length(0) == 6


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
    @pytest.mark.parametrize("recombination", [0, 0.3])
    def test_haplotypes_unchanged(self, seed, recombination):
        tables = random_tables(seed, recombination=recombination)
        samples = [int(u) for u in tables.sample_nodes()]
        gig = tables.copy().graph()
        simplified = tables.copy()
        node_map = simplified.simplify(samples)
        simplified = simplified.graph()
        assert len(simplified.nodes) < len(gig.nodes)
        for u in samples:
            assert simplified.haplotype(int(node_map[u])) == gig.haplotype(u)

    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("recombination", [0, 0.3])
    @pytest.mark.parametrize("mode", ["first_mrca", "closest_homologues", "all"])
    def test_homology_unchanged(self, seed, recombination, mode):
        tables = random_tables(seed, recombination=recombination)
        samples = [int(u) for u in tables.sample_nodes()]
        simplified = tables.copy()
        node_map = simplified.simplify(samples)
        check_homology_unchanged(tables, simplified, node_map, samples, mode)

    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("recombination", [0, 0.3])
    @pytest.mark.parametrize("mode", ["first_mrca", "closest_homologues", "all"])
    def test_homology_with_itself_unchanged(self, seed, recombination, mode):
        # The paralogous homology of a genome with itself is unchanged too
        tables = random_tables(seed, recombination=recombination)
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
        tables = random_tables(1)
        tables.simplify()
        once = tables.copy()
        node_map = tables.simplify()
        assert list(node_map) == list(range(len(once.nodes)))
        assert tables == once

    def test_samples_first_in_order(self):
        tables = random_tables(1)
        samples = [int(u) for u in tables.sample_nodes()][::-1]
        node_map = tables.simplify(samples)
        assert [int(node_map[u]) for u in samples] == list(range(len(samples)))
        assert list(tables.sample_nodes()) == list(range(len(samples)))

    @pytest.mark.parametrize(
        ("parent_left", "parent_right", "expected"),
        [
            (3, 10, (2, "T")),  # X has A's 3-10: A's position 5 is X's 2
            (10, 0, (4, "A")),  # X has A's 0-10 inverted: A's 5 is X's 4, and T becomes A
        ],
    )
    def test_mutation_moved_to_its_position_below(self, parent_left, parent_right, expected):
        # A (node 1) has a mutation T at its position 5, and is bypassed, so the mutation
        # moves down to the sample X (node 2), at the position that A's 5 maps to in X
        tables = gigl.Tables()
        tables.nodes.add_row(2)  # 0: R
        tables.nodes.add_row(1)  # 1: A
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)  # 2: X
        tables.iedges.add_row(1, 0, child_right=10)
        length = abs(parent_right - parent_left)
        tables.iedges.add_row(2, 1, child_right=length, parent_left=parent_left, parent_right=parent_right)
        tables.mutations.add_row(1, 5, "T")
        tables.sort()
        tables.validate()
        node_map = tables.simplify()
        assert node_map[1] == gigl.NULL
        assert [(m.node, m.position, m.derived_state) for m in tables.mutations] == [(0, *expected)]

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
        tables = random_tables(2)
        gig = tables.graph()
        samples = gig.sample_nodes()
        simplified = gig.simplify(samples)  # Node objects are accepted
        for i, u in enumerate(samples):
            assert simplified.haplotype(i) == gig.haplotype(u)

    def test_distinct_samples(self):
        tables = random_tables(1)
        with pytest.raises(ValueError, match="distinct"):
            tables.simplify([20, 20])

    @pytest.mark.parametrize("sample", [-1, 1000])
    def test_samples_are_nodes(self, sample):
        tables = random_tables(1)
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
