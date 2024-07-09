import numpy as np
import pytest
import tskit

import GeneticInheritanceGraphLibrary as gigl

from . import sim

INVERTED_CHILD_FLAG = 1 << 17


# Tests for functions in tests/gigutil.py
class tskit_DTWF_simulator:
    """
    A simple tree sequence simulator class, for testing against the equivalent
    GIG simulator. This is a simplified version of the tskit tutorial:
    https://tskit.dev/tutorials/forward_sims.html
    """

    # For visualising unsimplified tree sequences, it helps to flag all nodes as samples
    default_node_flags = tskit.NODE_IS_SAMPLE

    def make_diploid(self, time, parent_individuals=None) -> tuple[int, tuple[int, int]]:
        """
        Make an individual and its diploid genomes by adding to tables, returning IDs.
        Specifying parent_individuals is optional but stores the pedigree stored.
        """
        individual_id = self.tables.individuals.add_row(parents=parent_individuals)
        return individual_id, (
            self.tables.nodes.add_row(time=time, flags=self.default_node_flags, individual=individual_id),
            self.tables.nodes.add_row(time=time, flags=self.default_node_flags, individual=individual_id),
        )

    def new_population(self, time, prev_pop, recombination_rate=0, seq_len=None) -> dict[int, tuple[int, int]]:
        """
        if seq_len is specified, use this as the expected sequence length of the
        parents of the new population, otherwise take from the parent.sequence_length
        """
        pop = {}  # fill with individual_ID: (maternal_genome_ID, paternal_genome_ID)

        # Cache the list of individual IDs in the previous population, for efficiency
        prev_individuals = np.array([i for i in prev_pop.keys()], dtype=np.int32)

        # 1. Pick individual parent ID pairs at random, `replace=True` allows selfing
        mum_dad_arr = self.random.choice(prev_individuals, (len(prev_pop), 2), replace=True)

        for mum_and_dad in mum_dad_arr:
            # 2. Get 1 new individual ID + 2 new node IDs
            child_id, child_genomes = self.make_diploid(time, mum_and_dad)
            pop[child_id] = child_genomes  # store the genome IDs
            # 3. Add inheritance paths to both child genomes
            for child_genome, parent_individual in zip(child_genomes, mum_and_dad):
                parent_genomes = prev_pop[parent_individual]
                self.add_inheritance_paths(parent_genomes, child_genome)
        return pop

    def add_inheritance_paths(self, parent_nodes, child):
        edges = self.tables.edges
        L = self.tables.sequence_length
        random_pos = self.random.integers(L)
        rnd = self.random.integers(4)
        break_to_right_of_position = rnd & 1
        breakpt = random_pos + break_to_right_of_position

        if rnd & 2:  # Use 2nd bit to randomly choose 1st or 2nd parent node
            # We need to randomise the order of parent nodes to avoid
            # all children of this parent having the same genome to left / right
            lft_parent, rgt_parent = parent_nodes
        else:
            rgt_parent, lft_parent = parent_nodes

        if breakpt > 0:
            edges.add_row(left=0, right=breakpt, child=child, parent=lft_parent)
        if breakpt < L:
            edges.add_row(left=breakpt, right=L, child=child, parent=rgt_parent)

    def initialise_population(self, time, size) -> dict[int, tuple[int, int]]:
        return dict(self.make_diploid(time) for _ in range(size))

    def __init__(self, sequence_length):
        self.tables = tskit.TableCollection(sequence_length=sequence_length)
        self.tables.time_units = "generations"  # optional, but helpful when plotting

    def run(self, num_diploids, gens, random_seed=None):
        self.random = np.random.default_rng(random_seed)
        pop = self.initialise_population(gens, num_diploids)
        while gens > 0:
            gens = gens - 1
            pop = self.new_population(gens, pop)
        # make sure only nodes at time 0 are samples
        self.tables.nodes.flags = np.where(self.tables.nodes.time == 0, tskit.NODE_IS_SAMPLE, 0).astype(
            self.tables.nodes.flags.dtype
        )
        self.tables.sort()
        return self.tables.tree_sequence()


class DTWF_one_break_no_rec_inversions_test(sim.DTWF_one_break_no_rec_inversions_slow):
    """
    A GIG simulator used for testing: this version should result in the same breakpoints
    as in the tskit_DTWF_simulator.
    """

    def find_comparable_points(self, gig, parent_nodes, chroms):
        """ """
        mrcas = gig.find_mrca_regions(*parent_nodes, u_chromosomes=[chroms[0]], v_chromosomes=[chroms[1]])
        # Create a new mrca dict with arbitrary keys but where each value is a single
        # interval with the appropriate matching coords in u and v. Items in the dict
        # are sorted by the left coordinate of the mrca. Keys can be arbitrary because
        # we don't use the identity of the MRCA node to determine breakpoint dynamics.
        tmp = []
        for mrca_regions in mrcas.values():
            for region, equivalents in mrca_regions.items():
                tmp.append((region, equivalents))
        mrcas = gigl.tables.MRCAdict({k: {v[0]: v[1]} for k, v in enumerate(sorted(tmp, key=lambda x: x[0][0]))})
        comparable_pts = mrcas.random_match_pos(self.rng)
        # Don't bother with inversions: testing doesn't use them
        return (
            np.array([comparable_pts.u, comparable_pts.v], dtype=np.int64),
            (comparable_pts.chr_u, comparable_pts.chr_v),
        )


# MAIN TESTS BELOW


class TestSimpleSims:
    def test_no_recomb_sim(self):
        gens = 10
        simulator = sim.DTWF_no_recombination()
        gig = simulator.run(num_diploids=10, seq_lens=[100], gens=gens, random_seed=1)
        assert len(np.unique(gig.tables.nodes.time)) == gens + 1
        assert gig.num_iedges > 0

    def test_variable_population_size(self):
        gens = 2
        simulator = sim.DTWF_no_recombination()
        gig = simulator.run(num_diploids=(2, 10, 20), seq_lens=[100], gens=gens, random_seed=1)
        times, counts = np.unique(gig.tables.nodes.time, return_counts=True)
        assert np.array_equal(times, [0, 1, 2])
        assert np.array_equal(counts, [40, 20, 4])
        assert len(gig.individuals) == 2 + 10 + 20

    def test_multi_chromosomes(self):
        gens = 2
        simulator = sim.DTWF_no_recombination()
        gig = simulator.run(num_diploids=(2, 10, 20), seq_lens=[100, 50], gens=gens, random_seed=1)
        gig = gig.sample_resolve()
        for u in gig.sample_ids:
            assert set(gig.chromosomes(u)) == {0, 1}
            assert gig.sequence_length(u, 0) == gig.max_pos(u, 0) == 100
            assert gig.sequence_length(u, 1) == gig.max_pos(u, 1) == 50
            assert gig.sequence_length(u, 2) == 0
            assert gig.max_pos(u, 2) is None

    def test_run_more(self):
        simulator = sim.DTWF_no_recombination()
        gens = 2
        gig = simulator.run(num_diploids=2, seq_lens=[100], gens=gens, random_seed=1)
        assert len(np.unique(gig.tables.nodes.time)) == gens + 1
        new_gens = 3
        gig = simulator.run_more(num_diploids=4, gens=new_gens, random_seed=1)
        assert len(np.unique(gig.tables.nodes.time)) == gens + new_gens + 1


class TestDTWF_one_break_no_rec_inversions_slow:
    default_gens = 5
    seq_lens = (531,)
    inversion = tskit.Interval(100, 200)

    simulator = None
    ts = None  # only used to extract a tree sequence from subfunctions

    def test_plain_sim(self):
        gens = self.default_gens
        self.simulator = sim.DTWF_one_break_no_rec_inversions_slow()
        gig = self.simulator.run(num_diploids=10, seq_lens=self.seq_lens, gens=gens, random_seed=1)
        assert len(np.unique(gig.tables.nodes.time)) == gens + 1
        assert gig.num_iedges > 0
        ts = gig.to_tree_sequence()
        assert ts.num_samples == len(gig.sample_ids)
        assert ts.num_trees > 1
        assert ts.at_index(0).num_edges > 0

    def test_run_more(self):
        gens = self.default_gens
        self.simulator = sim.DTWF_one_break_no_rec_inversions_slow()
        gig = self.simulator.run(num_diploids=10, seq_lens=self.seq_lens, gens=gens, random_seed=1)
        new_gens = 2
        gig = self.simulator.run_more(num_diploids=(4, 2), gens=new_gens, random_seed=1)
        times, counts = np.unique(gig.tables.nodes.time, return_counts=True)
        assert len(times) == gens + new_gens + 1
        assert times[0] == 0
        assert counts[0] == 4  # 2 * 2
        assert times[1] == 1
        assert counts[1] == 8  # 4 * 2
        assert times[2] == 2
        assert counts[2] == 20

    @pytest.mark.parametrize("seed", [123, 321])
    def test_vs_tskit_implementation(self, seed):
        # The tskit_DTWF_simulator should produce identical results to the GIG simulator
        gens = self.default_gens
        self.simulator = DTWF_one_break_no_rec_inversions_test()
        ts_simulator = tskit_DTWF_simulator(sequence_length=self.seq_lens[0])
        gig = self.simulator.run(7, self.seq_lens, gens=gens, random_seed=seed)
        ts = ts_simulator.run(7, gens=gens, random_seed=seed)
        ts.tables.assert_equals(gig.to_tree_sequence().tables, ignore_provenance=True)
        assert ts.num_trees > 0

        # also test that sample_resolving is equivalent to simplifying with keep_nodes
        ts = ts.simplify(keep_unary=True, filter_individuals=False, filter_nodes=False)
        gig = gig.sample_resolve()
        ts.tables.assert_equals(gig.to_tree_sequence().tables, ignore_provenance=True)

    def test_inversion(self):
        """
        Run a simulation in which a single inversion is introduced then the
        population explodes so that the inversion is not lost.
        Note that this routine can be called directly e.g.
            from tests.test_gigutil import TestDTWF_one_break_no_rec_inversions_slow

            cls = TestDTWF_one_break_no_rec_inversions_slow()
            cls.test_inversion()
            print(cls.ts)
        """
        final_pop_size = 100
        self.simulator = sim.DTWF_one_break_no_rec_inversions_slow(
            initial_sizes={
                "nodes": 2 * final_pop_size * self.default_gens,
                "edges": 2 * final_pop_size * self.default_gens * 2,
                "individuals": final_pop_size * self.default_gens,
            },
        )

        self.simulator.run(
            num_diploids=2,
            seq_lens=self.seq_lens,
            gens=1,
            random_seed=123,
            further_node_flags=np.array([[INVERTED_CHILD_FLAG, 0], [0, 0]], dtype=np.int32),
        )
        # Insert an inversion by editing the tables
        tables = self.simulator.tables
        times, inverses = np.unique(tables.nodes.time, return_inverse=True)
        assert len(times) == 3  # also includes the grand MRCA
        first_gen = np.where(inverses == np.where([times == 1])[0])[0]
        second_gen = np.where(inverses == np.where([times == 0])[0])[0]
        assert len(first_gen) == 4  # haploid genomes
        assert len(second_gen) == 4  # haploid genomes
        # Edit the existing iedges to create an inversion
        # on a single iedge
        new_tables = tables.copy(omit_iedges=True)
        inverted_child_id = np.where(tables.nodes.flags == INVERTED_CHILD_FLAG)[0]
        assert len(inverted_child_id) == 1
        inverted_child_id = inverted_child_id[0]
        assert inverted_child_id in second_gen
        for ie in tables.iedges:
            if ie.child == inverted_child_id and ie.child_left == 0:
                assert ie.parent_left == 0
                assert ie.child_right == ie.parent_right
                assert ie.child_right > 200  # check seed gives breakpoint a bit along

                new_tables.add_iedge_row(
                    0,
                    self.inversion.left,
                    0,
                    self.inversion.left,
                    child=inverted_child_id,
                    parent=ie.parent,
                    **self.simulator.add_iedge_params(),
                )
                new_tables.add_iedge_row(
                    self.inversion.left,
                    self.inversion.right,
                    self.inversion.right,
                    self.inversion.left,
                    child=inverted_child_id,
                    parent=ie.parent,
                    **self.simulator.add_iedge_params(),
                )
                new_tables.add_iedge_row(
                    self.inversion.right,
                    ie.child_right,
                    self.inversion.right,
                    ie.parent_right,
                    child=inverted_child_id,
                    parent=ie.parent,
                    **self.simulator.add_iedge_params(),
                )
            else:
                new_tables.add_iedge_row(**ie._asdict(), **self.simulator.add_iedge_params())
        new_tables.sort()
        self.simulator.tables = new_tables
        # Check it gives a valid gig
        gig = self.simulator.tables.copy().graph()
        num_inversions = 0
        for ie in gig.iedges:
            if ie.is_inversion():
                num_inversions += 1
                assert ie.child_left == self.inversion.left
                assert ie.child_right == self.inversion.right
                assert ie.child == inverted_child_id
            else:
                assert ie.child_left == ie.parent_left
                assert ie.child_right == ie.parent_right
        assert num_inversions == 1

        # Can progress the simulation
        gig = self.simulator.run_more(
            num_diploids=final_pop_size,
            gens=self.default_gens - 1,
            random_seed=1,
        )
        # should have deleted the grand MRCA (used for matching)
        assert len(gig.nodes) == len(self.simulator.tables.nodes) - 1
        assert gig.num_samples == 100 * 2
        # check we still have a single inversion in the ancestry (not lost by drift)
        # (the child ID will have changed by removing the grand MRCA)
        num_inversions = 0
        for ie in self.simulator.tables.graph().iedges:
            if ie.is_inversion():
                num_inversions += 1
                assert ie.child == inverted_child_id
                inverted_child_id = ie.child
            else:
                assert ie.child_left == ie.parent_left
                assert ie.child_right == ie.parent_right

        assert num_inversions == 1

        # Check we can turn the decapitated gig tables into a tree sequence
        # (as long as this isn't simplified, decapitation should remove the only SV)
        tables = gig.tables.copy()
        node_map = tables.decapitate(time=gig.max_time)
        decapitated_gig = tables.graph()
        # check that decapitation has removed the inversion
        for ie in decapitated_gig.iedges:
            assert ie.child != node_map[inverted_child_id]
            assert not ie.is_inversion()
            assert decapitated_gig.nodes[ie.child].time + 1 == decapitated_gig.nodes[ie.parent].time
            assert ie.child_left == ie.parent_left
            assert ie.child_right == ie.parent_right
        new_gig = decapitated_gig.sample_resolve()
        ts = new_gig.to_tree_sequence()
        # check sample resolve has had an effect
        assert len(new_gig.iedges) != len(decapitated_gig.iedges)
        assert new_gig.nodes[node_map[inverted_child_id]]
        self.ts = ts  # also store in the class
        assert ts.max_time == new_gig.max_time
        assert ts.max_time < gig.max_time
        inversion_above_node = np.where(ts.nodes_flags == INVERTED_CHILD_FLAG)[0]
        assert len(inversion_above_node) == 1
        inversion_samples = list(ts.at(150).samples(inversion_above_node[0]))
        non_inversion_samples = np.zeros(ts.num_nodes, dtype=bool)
        non_inversion_samples[ts.samples()] = True
        non_inversion_samples[inversion_samples] = False
        non_inversion_samples = np.where(non_inversion_samples)[0]

        # the average divergence between samples with the inversion and without
        # should be maxed out within the inversion site but not elsewhere
        max_divergence = ts.max_time * 2
        breaks = list(ts.breakpoints())
        assert len(breaks) > 40
        sample_sets = [inversion_samples, non_inversion_samples]
        divergence = ts.divergence(sample_sets, mode="branch", windows="trees")
        # check av divergence at LHS and RHS of inversion definitely lower than max
        # it won't be much lower though, because we have rather few generations
        # relative to the population size
        assert np.all(divergence[:10] < max_divergence * 0.999)
        assert np.all(divergence[-10:] < max_divergence * 0.999)
        for left, right, val in zip(breaks[:-1], breaks[1:], divergence):
            if right > self.inversion.left and left < self.inversion.right:
                assert np.isclose(val, max_divergence)
        windows = [0, self.inversion.left, self.inversion.right, ts.sequence_length]
        divergence = ts.divergence(sample_sets, mode="branch", windows=windows)
        assert divergence[0] < max_divergence * 0.999
        assert np.isclose(divergence[1], max_divergence)
        assert divergence[2] < max_divergence * 0.999

    def test_multi_chromosomes(self):
        gens = 10
        simulator = sim.DTWF_one_break_no_rec_inversions_slow()
        # a few rounds of tiny population sizes, so that we get coalescence
        simulator.run(num_diploids=(2, 2, 2), seq_lens=[100, 50, 200], random_seed=1)
        gig = simulator.run_more(num_diploids=10, gens=gens, random_seed=1)
        gig = gig.sample_resolve()
        # No recombinations between chromosomes in this simulation: check there are no
        # MRCA regions shared between chromosomes
        s1 = gig.sample_ids[0]
        s2 = gig.sample_ids[1]
        for chromA in gig.chromosomes(s1):
            for chromB in gig.chromosomes(s2):
                if chromA != chromB:
                    assert len(gig.find_mrca_regions(s1, s2, u_chromosomes=[chromA], v_chromosomes=[chromB])) == 0
                else:
                    assert len(gig.find_mrca_regions(s1, s2, u_chromosomes=[chromA], v_chromosomes=[chromB])) > 0
