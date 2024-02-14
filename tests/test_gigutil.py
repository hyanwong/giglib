import numpy as np
import pytest
import tskit

from .gigutil import DTWF_no_recombination_sim
from .gigutil import DTWF_one_break_no_rec_inversions_slow_sim


# Tests for functions in tests/gigutil.py
class tskit_DTWF_simulator:
    """
    A simple tree sequence simulator class, for testing against the equivalent
    GIG simulator. This is a simplified version of the tskit tutorial:
    https://tskit.dev/tutorials/forward_sims.html
    """

    # For visualising unsimplified tree sequences, it helps to flag all nodes as samples
    default_node_flags = tskit.NODE_IS_SAMPLE

    def make_diploid(
        self, time, parent_individuals=None
    ) -> tuple[int, tuple[int, int]]:
        """
        Make an individual and its diploid genomes by adding to tables, returning IDs.
        Specifying parent_individuals is optional but stores the pedigree stored.
        """
        individual_id = self.tables.individuals.add_row(parents=parent_individuals)
        return individual_id, (
            self.tables.nodes.add_row(
                time=time, flags=self.default_node_flags, individual=individual_id
            ),
            self.tables.nodes.add_row(
                time=time, flags=self.default_node_flags, individual=individual_id
            ),
        )

    def new_population(
        self, time, prev_pop, recombination_rate=0, seq_len=None
    ) -> dict[int, tuple[int, int]]:
        """
        if seq_len is specified, use this as the expected sequence length of the
        parents of the new population, otherwise take from the parent.sequence_length
        """
        pop = {}  # fill with individual_ID: (maternal_genome_ID, paternal_genome_ID)

        # Cache the list of individual IDs in the previous population, for efficiency
        prev_individuals = np.array([i for i in prev_pop.keys()], dtype=np.int32)

        for _ in range(len(prev_pop)):
            # 1. Pick two individual parent IDs at random, `replace=True` allows selfing
            mum_and_dad = self.random.choice(prev_individuals, 2, replace=True)
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
        self.tables.sort()
        return self.tables.tree_sequence()


class DTWF_one_break_no_rec_inversions_test(DTWF_one_break_no_rec_inversions_slow_sim):
    """
    A GIG simulator used for testing: this version should result in the same breakpoints
    as in the tskit_DTWF_simulator.
    """

    def find_comparable_points(self, gig, parent_nodes):
        """ """
        mrcas = gig.find_mrca_regions(*parent_nodes)
        # Create a new mrca dict with arbitrary keys but where each value is a single
        # interval with the appropriate matching coords in u and v. Items in the dict
        # are sorted by the left coordinate of the mrca. Keys can be arbitrary because
        # we don't use the identity of the MRCA node to determine breakpoint dynamics.
        tmp = []
        for mrca_regions in mrcas.values():
            for region, equivalents in mrca_regions.items():
                tmp.append((region, equivalents))
        comparable_pts = gig.random_matching_positions(
            {k: {v[0]: v[1]} for k, v in enumerate(sorted(tmp, key=lambda x: x[0][0]))},
            self.random,
        )
        return comparable_pts  # Don't bother with inversions: testing doesn't use them


# MAIN TESTS BELOW


class TestSimpleSims:
    def test_no_recomb_sim(self):
        gens = 10
        simulator = DTWF_no_recombination_sim()
        gig = simulator.run(
            num_diploids=10, seq_len=100, generations=gens, random_seed=1
        )
        assert len(np.unique(gig.tables.nodes.time)) == gens + 1
        assert gig.num_iedges > 0


class TestDTWF_recombination_no_SV_sims:
    def test_one_break_slow_sim(self):
        gens = 10
        simulator = DTWF_one_break_no_rec_inversions_slow_sim()
        gig = simulator.run(num_diploids=10, seq_len=100, gens=gens, random_seed=1)
        assert len(np.unique(gig.tables.nodes.time)) == gens + 1
        assert gig.num_iedges > 0
        ts = gig.to_tree_sequence()
        assert ts.num_samples == len(gig.samples)
        assert ts.num_trees > 1
        assert ts.at_index(0).num_edges > 0

    @pytest.mark.parametrize("seed", [123, 321])
    def test_one_break_slow_sim_vs_tskit(self, seed):
        # The tskit_DTWF_simulator should produce identical results to the GIG simulator
        gens = 9
        L = 97
        gig_simulator = DTWF_one_break_no_rec_inversions_test()
        ts_simulator = tskit_DTWF_simulator(sequence_length=L)
        gig = gig_simulator.run(7, L, gens=gens, random_seed=seed)
        ts = ts_simulator.run(7, gens=gens, random_seed=seed)
        ts.tables.assert_equals(gig.to_tree_sequence().tables, ignore_provenance=True)
        assert ts.num_trees > 0

        # also test that sample_resolving is equivalent to simplifying with keep_nodes
        ts = ts.simplify(keep_unary=True, filter_individuals=False, filter_nodes=False)
        gig = gig.sample_resolve()
        ts.tables.assert_equals(gig.to_tree_sequence().tables, ignore_provenance=True)
