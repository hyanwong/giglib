"""
The base class for simple forward-time (discrete-time Wright-Fisher) simulators that record
their genealogies as GIGs. The simulators are based on the forward-time simulation tutorial
for tskit (https://tskit.dev/tutorials/forward_sims.html).
"""

import numpy as np

import giglib as gigl


class DTWF_simulator:
    """
    A simple base simulator class, mostly code from
    https://tskit.dev/tutorials/forward_sims.html
    """

    # For visualising unsimplified tree sequences, it helps to flag all nodes as samples
    default_node_flags = gigl.NODE_IS_SAMPLE

    def make_diploids(self, time, parent_individuals, node_flags=None):
        """
        Make a set of individuals with their diploid genomes by adding to tables,
        returning two arrays of IDs: the individual IDs and the node IDs.
        """
        individual_ids = [self.tables.individuals.add_row(parents=p) for p in parent_individuals]
        if node_flags is None:
            node_flags = self.default_node_flags
        individual = np.broadcast_to(individual_ids, (2, len(individual_ids))).T

        node_ids = []
        for data in zip(*np.broadcast_arrays(time, node_flags, individual)):
            node_ids.append(
                [self.tables.nodes.add_row(tm, flags=flags, individual=ind_id) for tm, flags, ind_id in zip(*data)]
            )
        return individual_ids, node_ids

    def new_population(self, time, size=None, recombination_rate=0, node_flags=None):
        """
        If num_diploids is specified, use this as the number of diploids in the new
        population, otherwise use the number of diploids in the previous population.
        """
        if self.pop is None:
            raise ValueError("Need to initialise a new population first")
        self.validate()  # the iedges of the parents, added in earlier generations
        prev_pop = self.pop
        self.pop = {}  # fill with individualID: (maternal_genomeID, paternal_genomeID)

        # Cache the list of individual IDs in the previous population, for efficiency
        prev_individuals = np.array([i for i in prev_pop.keys()], dtype=np.int32)

        if size is None:
            size = len(prev_pop)

        # 1. Pick individual parent ID pairs at random, `replace=True` allows selfing
        mum_dad_arr = self.rng.choice(prev_individuals, (size, 2), replace=True)
        # 2. Get new individual IDs + twice as many new node IDs
        child_id_arr, child_genomes_arr = self.make_diploids(time, mum_dad_arr, node_flags)
        assert len(mum_dad_arr) == len(child_id_arr) == len(child_genomes_arr)
        for mother_and_father, child_id, child_genomes in zip(mum_dad_arr, child_id_arr, child_genomes_arr):
            # assert (self.tables.nodes[child_genomes[0]].individual ==
            # self.tables.nodes[child_genomes[1]].individual)  # removed for speed
            self.pop[child_id] = child_genomes  # store the genome IDs

            # 3. Add inheritance paths to both child genomes
            for child_genome, parent_individual in zip(child_genomes, mother_and_father):
                parent_genomes = prev_pop[parent_individual]
                self.add_inheritance_paths(
                    parent_genomes,
                    child_genome,
                    recombination_rate,
                )

    def add_inheritance_paths(self, parent_nodes, child_node, recombination_rate):
        # For the first generation, this will need to use self.seq_lens
        # to establish the number of chromosomes and their correct bounds
        raise NotImplementedError("Implement an add_inheritance_paths method to make a DTWF simulator")

    def initialise_population(
        # seq_lens is a dict mapping chromosome number to sequence length
        self,
        time,
        size,
        seq_lens,
        node_flags=None,
    ) -> dict[int, tuple[int, int]]:
        self.tables.clear()
        self.pop = None
        self.seq_lens = seq_lens
        return dict(
            zip(
                *self.make_diploids(
                    time,
                    parent_individuals=np.array([], dtype=int).reshape(size, 0),
                    node_flags=node_flags,
                )
            )
        )

    def __init__(self, skip_validate=False, initial_sizes=None):
        """
        Create a simulation class, which can then simulate a forward Discrete Time
        Wright-Fisher model with varying population sizes over generations.

        If skip_validate is False (default), the iedges are validated at the start of
        each generation, so that find_mrca_regions() can be used on the parents (it only
        needs the parents' iedges, added in earlier generations, to be validated).
        Setting this to True instead marks them as valid without checking, which saves
        a little simulation time, at the expense of not doing any validation (dangerous!)

        If initial_sizes is specified, it should be an array of the expected sizes
        for each table, see the ``gigl.Tables.__init__`` method for more information.
        """
        self.tables = gigl.Tables(initial_sizes=initial_sizes)
        self.tables.time_units = "generations"  # optional, but helpful when plotting
        self.skip_validate = skip_validate

    def add_iedge_row(self, child, parent, **kwargs):
        self.tables.iedges.add_row(child, parent, **kwargs)

    def validate(self):
        """
        Validate the iedges added since the last call (i.e. in the previous generation),
        or if ``skip_validate`` is True, mark them as valid without checking
        """
        if self.skip_validate:
            self.tables.iedges.set_bitflag(gigl.ValidFlags.IEDGES_ALL)
        else:
            self.tables.validate(gigl.ValidFlags.IEDGES_ALL)

    def run(
        self,
        num_diploids,
        seq_lens,
        gens,
        *,
        num_chromosomes=1,
        random_seed=None,
        initial_node_flags=None,
        further_node_flags=None,
    ):
        """
        Initialise and run a new population for a given number of generations. The last
        generation will be at time 0 and the first at time `gens`.

        The num_diploids param can be an array of length `gens + 1` giving the diploid
        population size in each generation. This allows quick growth of a population.

        ``initial_node_flags`` will be passed as node_flags to the
        ``initialise_population()`` method. ``further_node_flags`` will be passed
        to subsequent generations.

        Returns a gig represeting the ancestry of the population at time 0
        """
        self.rng = np.random.default_rng(random_seed)
        try:
            seq_lens = {chrom: length for chrom, length in seq_lens.items()}
        except AttributeError:
            seq_lens = {chrom: length for chrom, length in enumerate(seq_lens)}
        if isinstance(num_diploids, int):
            num_diploids = [num_diploids] * (gens + 1)

        self.pop = self.initialise_population(
            gens,
            num_diploids[-gens - 1],
            node_flags=initial_node_flags,
            seq_lens=seq_lens,
        )
        while gens > 0:
            gens -= 1
            self.new_population(gens, size=num_diploids[-gens - 1], node_flags=further_node_flags)

        self.tables.sort()
        # We should probably simplify or at least sample_resolve here?
        # We should also mark gen 0 as samples and unmark the others.
        # Probably a parameter `simplify` would be useful?
        return self.tables.copy().graph()

    def run_more(self, num_diploids, gens, random_seed=None):
        """
        The num_diploids parameter can be an array of length `gens` giving the diploid
        population size in each generation.
        """
        if self.pop is None:
            raise ValueError("Need to call run() first")
        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)
        if isinstance(num_diploids, int):
            num_diploids = [num_diploids] * (gens)

        # augment the generations
        self.tables.change_times(gens)
        while gens > 0:
            gens -= 1
            self.new_population(gens, size=num_diploids[-gens - 1])

        self.tables.sort()
        return self.tables.copy().graph()
