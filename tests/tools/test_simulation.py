import numpy as np
import pytest
import tskit

from giglib.tools import homology
from giglib.tools.simulation.microsatellites_backward import sim_microsatellites_backward
from giglib.tools.simulation.microsatellites_forward import DTWF_microsatellite_sim
from giglib.tools.simulation.no_recombination import DTWF_no_recombination_sim
from giglib.tools.simulation.one_break_inversions import DTWF_one_break_no_rec_inversions_slow_sim
from giglib.tools.simulation.single_origin import sim_single_origin

INVERTED_CHILD_FLAG = 1 << 17


# Tests for the simulators in giglib.tools.simulation
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


# MAIN TESTS BELOW


class TestSimpleSims:
    def test_no_recomb_sim(self):
        gens = 10
        simulator = DTWF_no_recombination_sim()
        gig = simulator.run(num_diploids=10, seq_lens=[100], gens=gens, random_seed=1)
        assert len(np.unique(gig.tables.nodes.time)) == gens + 1
        assert len(gig.iedges) > 0

    def test_variable_population_size(self):
        gens = 2
        simulator = DTWF_no_recombination_sim()
        gig = simulator.run(num_diploids=(2, 10, 20), seq_lens=[100], gens=gens, random_seed=1)
        times, counts = np.unique(gig.tables.nodes.time, return_counts=True)
        assert np.array_equal(times, [0, 1, 2])
        assert np.array_equal(counts, [40, 20, 4])
        assert len(gig.individuals) == 2 + 10 + 20

    def test_multi_chromosomes(self):
        gens = 2
        simulator = DTWF_no_recombination_sim()
        gig = simulator.run(num_diploids=(2, 10, 20), seq_lens=[100, 50], gens=gens, random_seed=1)
        gig = gig.sample_resolve()
        for u in gig.tables.sample_nodes():
            assert set(gig.chromosomes(u)) == {0, 1}
            assert gig.sequence_length(u, chromosome=0) == gig.max_pos(u, chromosome=0) == 100
            assert gig.sequence_length(u, chromosome=1) == gig.max_pos(u, chromosome=1) == 50
            assert gig.sequence_length(u, chromosome=2) == 0
            assert gig.max_pos(u, chromosome=2) is None

    def test_run_more(self):
        simulator = DTWF_no_recombination_sim()
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
        self.simulator = DTWF_one_break_no_rec_inversions_slow_sim()
        gig = self.simulator.run(num_diploids=10, seq_lens=self.seq_lens, gens=gens, random_seed=1)
        assert len(np.unique(gig.tables.nodes.time)) == gens + 1
        assert len(gig.iedges) > 0
        ts = gig.to_tree_sequence()
        assert ts.num_samples == len(gig.tables.sample_nodes())
        assert ts.num_trees > 1
        assert ts.at_index(0).num_edges > 0

    def test_run_more(self):
        gens = self.default_gens
        self.simulator = DTWF_one_break_no_rec_inversions_slow_sim()
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
        # The tskit_DTWF_simulator should produce identical results to the GIG simulator:
        # without duplications, the homology between two parents is sorted by position, so
        # the breakpoints are chosen in the same way. This relies on both drawing random
        # numbers in the same order, so if the GIG simulator changes how it chooses
        # breakpoints (e.g. by rejection sampling, issue 102), change the tskit one to match.
        gens = self.default_gens
        self.simulator = DTWF_one_break_no_rec_inversions_slow_sim()
        ts_simulator = tskit_DTWF_simulator(sequence_length=self.seq_lens[0])
        gig = self.simulator.run(7, self.seq_lens, gens=gens, random_seed=seed)
        ts = ts_simulator.run(7, gens=gens, random_seed=seed)
        ts.tables.assert_equals(gig.to_tree_sequence().tables, ignore_provenance=True)
        assert ts.num_trees > 0

        # also test that sample_resolving is equivalent to simplifying with keep_nodes
        ts = ts.simplify(keep_unary=True, filter_individuals=False, filter_nodes=False)
        gig = gig.sample_resolve()
        ts.tables.assert_equals(gig.to_tree_sequence().tables, ignore_provenance=True)

    def test_tandem_duplication(self):
        """
        Insert a large tandem duplication into a few genomes, then carry on simulating.
        Nothing stops crossovers between different copies of the duplicated region, so
        unequal crossing-over changes the number of copies (from issue #100). However
        many copies a genome has, its length must be the original length plus a whole
        number of duplicated regions.
        """
        length, dup_left, dup_right = 531, 50, 500  # positions 50-500 are duplicated
        simulator = DTWF_one_break_no_rec_inversions_slow_sim()
        simulator.run(num_diploids=10, seq_lens=[length], gens=1, random_seed=2)
        # Replace the iedges of 5 genomes in the last generation with those of a tandem
        # duplication of their (first) parent's genome: the parent's positions up to
        # dup_right, then a copy of dup_left onwards
        tables = simulator.tables
        duplicated = [int(u) for u in np.where(tables.nodes.time == 0)[0][:5]]
        new_tables = tables.copy()
        new_tables.iedges.clear()
        for ie in tables.iedges:
            if ie.child not in duplicated:
                new_tables.iedges.add_row(**ie._asdict())
        for u in duplicated:
            parent = next(ie.parent for ie in tables.iedges if ie.child == u)
            new_tables.iedges.add_row(u, parent, child_right=dup_right)
            new_tables.iedges.add_row(
                u,
                parent,
                child_left=dup_right,
                child_right=length + dup_right - dup_left,
                parent_left=dup_left,
                parent_right=length,
            )
        new_tables.sort()
        new_tables.validate()
        simulator.tables = new_tables

        gig = simulator.run_more(num_diploids=20, gens=20, random_seed=2)
        dup_length = dup_right - dup_left
        num_copies = []
        for u in gig.sample_nodes(time=0).ids:
            extra = gig.sequence_length(u) - length
            assert extra % dup_length == 0
            num_copies.append(1 + extra // dup_length)
        # With this seed, genomes with one and with two copies remain (other seeds may lose
        # the duplication by drift)
        assert set(num_copies) == {1, 2}, "the random seed doesn't suit this test"

    def test_inversion(self):
        """
        Run a simulation in which a single inversion is introduced then the
        population explodes so that the inversion is not lost.
        Note that this routine can be called directly e.g.
            from tests.tools.test_simulation import TestDTWF_one_break_no_rec_inversions_slow

            cls = TestDTWF_one_break_no_rec_inversions_slow()
            cls.test_inversion()
            print(cls.ts)
        """
        final_pop_size = 100
        self.simulator = DTWF_one_break_no_rec_inversions_slow_sim(
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
        new_tables = tables.copy()
        new_tables.iedges.clear()
        inverted_child_id = np.where(tables.nodes.flags == INVERTED_CHILD_FLAG)[0]
        assert len(inverted_child_id) == 1
        inverted_child_id = inverted_child_id[0]
        assert inverted_child_id in second_gen
        for ie in tables.iedges:
            if ie.child == inverted_child_id and ie.child_left == 0:
                assert ie.parent_left == 0
                assert ie.child_right == ie.parent_right
                # The inversion is inserted into this iedge, so it must cover it. This holds
                # for this seed (if the simulator draws its breakpoints differently, pick a
                # seed for which it does)
                assert ie.child_right > self.inversion.right, "the random seed doesn't suit this test"

                new_tables.iedges.add_row(
                    inverted_child_id,
                    ie.parent,
                    child_right=self.inversion.left,
                )
                new_tables.iedges.add_row(
                    inverted_child_id,
                    ie.parent,
                    child_left=self.inversion.left,
                    child_right=self.inversion.right,
                    parent_left=self.inversion.right,
                    parent_right=self.inversion.left,
                )
                new_tables.iedges.add_row(
                    inverted_child_id,
                    ie.parent,
                    child_left=self.inversion.right,
                    child_right=ie.child_right,
                    parent_left=self.inversion.right,
                    parent_right=ie.parent_right,
                )
            else:
                new_tables.iedges.add_row(**ie._asdict())
        new_tables.sort()
        new_tables.validate()
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
        assert len(gig.sample_nodes()) == 100 * 2
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
        simulator = DTWF_one_break_no_rec_inversions_slow_sim()
        # a few rounds of tiny population sizes, so that we get coalescence
        simulator.run(num_diploids=(2, 2, 2), seq_lens=[100, 50, 200], random_seed=1)
        gig = simulator.run_more(num_diploids=10, gens=gens, random_seed=1)
        gig = gig.sample_resolve()
        # No recombinations between chromosomes in this simulation: check there is no
        # homology between chromosomes
        s1 = gig.tables.sample_nodes()[0]
        s2 = gig.tables.sample_nodes()[1]
        for chromA in gig.chromosomes(s1):
            for chromB in gig.chromosomes(s2):
                if chromA != chromB:
                    assert len(gig.find_homology(s1, s2, u_chromosome=chromA, v_chromosome=chromB)) == 0
                else:
                    assert len(gig.find_homology(s1, s2, u_chromosome=chromA, v_chromosome=chromB)) > 0


class TestMicrosatelliteSim:
    """
    Microsatellites change the lengths of genomes, so homology between parents is broken
    into blocks, which crossovers must chain across (see issue 202)
    """

    loci = ((20, 2, 5), (70, 3, 4), (120, 2, 6))

    def simulate(self, *, loci=None, mutation_rate=0.2, max_gap=20, min_chain_length=100, gens=8, seed=1):
        sim = DTWF_microsatellite_sim(
            self.loci if loci is None else loci,
            mutation_rate=mutation_rate,
            min_chain_length=min_chain_length,
            max_gap=max_gap,
        )
        gig = sim.run(num_diploids=5, seq_lens=[200], gens=gens, random_seed=seed)
        return sim, gig

    def test_lengths_change_only_at_microsatellites(self):
        sim, gig = self.simulate()
        lengths = set()
        for u in gig.sample_nodes().ids:
            # (node IDs in the GIG are one less than in the simulation, which has a grand MRCA)
            changes = [unit * (num - sim.founder_loci[i][2]) for i, _, unit, num in sim.loci[u + 1]]
            assert gig.sequence_length(u) == 200 + sum(changes)
            lengths.add(gig.sequence_length(u))
        assert len(lengths) > 1  # the microsatellites have mutated

    def test_crossovers_continue(self):
        # Every child has a crossover (inheriting from both parents), however much the
        # parents' microsatellites differ. (An unequal crossover would raise an error, as
        # a microsatellite would be inherited twice, or not at all.)
        sim, _ = self.simulate()
        parents = {}
        for ie in sim.tables.iedges:
            parents.setdefault(ie.child, set()).add(ie.parent)
        children = [u for u, ps in parents.items() if sim.grand_mrca not in ps]
        assert len(children) > 0
        assert all(len(parents[u]) == 2 for u in children)

    def test_dense_microsatellites_need_chains(self):
        # With a microsatellite every 20 positions, no single block is long enough for a
        # crossover once the parents' microsatellites differ, but chains are
        loci = [(x, 2, 3) for x in range(5, 200, 20)]
        self.simulate(loci=loci, max_gap=10)
        with pytest.raises(ValueError, match="No place for a crossover"):
            self.simulate(loci=loci, max_gap=0)

    def test_no_mutations(self):
        _, gig = self.simulate(mutation_rate=0)
        assert {gig.sequence_length(u) for u in gig.sample_nodes().ids} == {200}

    def test_mostly_microsatellite(self):
        # A microsatellite covers 180 of the 200 positions, leaving room for crossovers
        # only either side of it: every meiosis still has one (with no limit on tries)
        _, gig = self.simulate(loci=[(10, 1, 180)], mutation_rate=0, gens=20)
        for ie in gig.iedges:
            # Each break between two parents is outside the microsatellite
            assert ie.child_left < 10 or ie.child_left > 190

    def test_breaks_are_inside_the_genome(self):
        # With genomes of 2 positions, the only break strictly inside them is between
        # the two, so every child gets one position from each genome of its parent
        sim = DTWF_microsatellite_sim([], mutation_rate=0, min_chain_length=1, max_gap=0)
        gig = sim.run(num_diploids=3, seq_lens=[2], gens=5, random_seed=1)
        for u in gig.nodes:
            if u.time < 5:
                iedges = gig.iedges_for_child(u)
                assert [(ie.child_left, ie.child_right) for ie in iedges] == [(0, 1), (1, 2)]
                assert iedges[0].parent != iedges[1].parent

    def test_overlapping_loci(self):
        with pytest.raises(ValueError, match="must not overlap"):
            DTWF_microsatellite_sim([(10, 2, 5), (15, 2, 5)], mutation_rate=0, min_chain_length=1, max_gap=0)

    def test_loci_before_the_sequence(self):
        with pytest.raises(ValueError, match="within the sequence"):
            DTWF_microsatellite_sim([(-5, 2, 5)], mutation_rate=0, min_chain_length=1, max_gap=0)

    def test_loci_beyond_the_sequence(self):
        sim = DTWF_microsatellite_sim([(195, 2, 5)], mutation_rate=0, min_chain_length=1, max_gap=0)
        with pytest.raises(ValueError, match="within the sequence"):
            sim.run(num_diploids=2, seq_lens=[200], gens=1, random_seed=1)


class TestMicrosatellitesBackward:
    """
    Microsatellites laid on a genealogy simulated with msprime (see issue 202)
    """

    loci = ((200, "AT"), (500, "CAG"), (800, "GATA"))

    def simulate(self, seed=1, **kwargs):
        params = {"max_units": 10, "population_size": 1e4, "recombination_rate": 1e-6, "mutation_rate": 1e-4}
        params.update(kwargs)
        return sim_microsatellites_backward(10, 1000, self.loci, random_seed=seed, **params)

    def repeats(self, gig, u):
        # The known bases of a genome are its microsatellites, in order
        bases = "".join(c for c in gig.haplotype(u) if c != "N")
        counts = []
        for _, motif in self.loci:
            n = 0
            while bases.startswith(motif):
                bases, n = bases[len(motif) :], n + 1
            counts.append(n)
        assert bases == ""
        return counts

    def test_lengths_change_only_at_microsatellites(self):
        gig = self.simulate()
        lengths = set()
        for u in gig.sample_nodes().ids:
            counts = self.repeats(gig, u)
            assert all(1 <= n <= 10 for n in counts)
            # (each locus is a single position in the sequence simulated by msprime)
            expected = 1000 - len(self.loci) + sum(len(motif) * n for (_, motif), n in zip(self.loci, counts))
            assert gig.sequence_length(u) == expected
            lengths.add(expected)
        assert len(lengths) > 1

    def test_chains_span_the_genome(self):
        # The homology between two samples is one chain, across the microsatellites
        gig = self.simulate()
        u, v = gig.sample_nodes().ids[:2]
        chains = homology.chains(homology.homology_blocks(gig.tables, u, v), max_gap=4 * 10)
        shortest = min(gig.sequence_length(u), gig.sequence_length(v))
        assert homology.chain_length(chains[0]) > shortest - 4 * 10 * len(self.loci)

    def test_reproducible(self):
        assert self.simulate(seed=3).tables == self.simulate(seed=3).tables

    def test_no_mutations(self):
        # Each locus still has its ancestral number of repeats, the same in every sample
        gig = self.simulate(mutation_rate=0)
        counts = {tuple(self.repeats(gig, u)) for u in gig.sample_nodes().ids}
        assert len(counts) == 1
        (counts,) = counts
        assert all(1 <= n <= 10 for n in counts)
        expected = 1000 - len(self.loci) + sum(len(motif) * n for (_, motif), n in zip(self.loci, counts))
        assert {gig.sequence_length(u) for u in gig.sample_nodes().ids} == {expected}

    def test_max_units_one(self):
        # With at most one repeat, a microsatellite can't vary
        with pytest.raises(ValueError, match="max_units must be at least 2"):
            sim_microsatellites_backward(
                2, 1000, self.loci, max_units=1, population_size=100, recombination_rate=0, mutation_rate=0
            )

    @pytest.mark.parametrize("loci", [[(200, "AT"), (200, "CA")], [(1000, "AT")], [(-1, "AT")]])
    def test_bad_loci(self, loci):
        with pytest.raises(ValueError, match="distinct, and within the sequence"):
            sim_microsatellites_backward(
                2, 1000, loci, max_units=5, population_size=100, recombination_rate=0, mutation_rate=0
            )


class TestSingleOrigin:
    def test_population_grows_then_stays_the_same_size(self):
        gig = sim_single_origin(10, 8, duplication_rate=0.1, deletion_rate=0.1, random_seed=1)
        times, counts = np.unique(gig.tables.nodes.time, return_counts=True)
        # From the most recent generation (time 0) back to the single genome (time 8)
        assert list(times) == list(range(9))
        assert list(counts) == [10, 10, 10, 10, 10, 8, 4, 2, 1]
        assert len(gig.sample_nodes()) == 10

    def test_no_duplications_or_deletions(self):
        gig = sim_single_origin(5, 10, duplication_rate=0, deletion_rate=0, random_seed=1)
        assert {gig.sequence_length(u) for u in gig.sample_nodes().ids} == {1}

    def test_duplications_only(self):
        # Each duplication adds at least one position, so after 10 generations a genome
        # has at least 11
        gig = sim_single_origin(5, 10, duplication_rate=1, deletion_rate=0, random_seed=1)
        assert min(gig.sequence_length(u) for u in gig.sample_nodes().ids) >= 11

    def test_long_stretches_are_redrawn(self):
        # With very long stretches, a duplication would double every genome if its length
        # were capped at the genome's length (giving 8 positions after 3 generations), but
        # it is redrawn, so the lengths vary
        gig = sim_single_origin(50, 3, duplication_rate=1, deletion_rate=0, mean_length=1000, random_seed=1)
        assert len({gig.sequence_length(u) for u in gig.sample_nodes().ids}) > 1

    def test_deletions_keep_a_position(self):
        gig = sim_single_origin(5, 50, duplication_rate=0.5, deletion_rate=0.5, random_seed=1)
        assert min(gig.sequence_length(u) for u in gig.sample_nodes().ids) >= 1

    def test_every_position_descends_from_the_single_base(self):
        gig = sim_single_origin(5, 30, duplication_rate=0.5, deletion_rate=0.2, mean_length=3, random_seed=1)
        root = 0
        assert gig.nodes[root].time == 30
        for u in gig.sample_nodes().ids:
            # Every position of u is homologous to the root's single position, through the root
            homology = gig.find_homology(root, u)
            assert all(h.u == (0, 1, 0) and h.mrca == root for h in homology)
            assert sum(h.length for h in homology) == gig.sequence_length(u)

    def test_reproducible(self):
        params = {"duplication_rate": 0.2, "deletion_rate": 0.1, "random_seed": 3}
        assert sim_single_origin(5, 20, **params).tables == sim_single_origin(5, 20, **params).tables

    def test_rates_too_high(self):
        with pytest.raises(ValueError, match="add up to more than 1"):
            sim_single_origin(5, 20, duplication_rate=0.6, deletion_rate=0.6)

    @pytest.mark.parametrize(("duplication_rate", "deletion_rate"), [(1.2, -0.3), (0.5, float("nan"))])
    def test_rates_not_probabilities(self, duplication_rate, deletion_rate):
        with pytest.raises(ValueError, match="between 0 and 1"):
            sim_single_origin(5, 20, duplication_rate=duplication_rate, deletion_rate=deletion_rate)

    @pytest.mark.parametrize(("population_size", "generations"), [(0, 10), (5, 0)])
    def test_too_small(self, population_size, generations):
        with pytest.raises(ValueError, match="at least 1"):
            sim_single_origin(population_size, generations, duplication_rate=0.1, deletion_rate=0.1)
