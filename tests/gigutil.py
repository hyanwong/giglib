import GeneticInheritanceGraphLibrary as gigl
import numpy as np

# Utilities for creating and editing gigs


def make_iedges_table(arr, tables):
    """
    Make an intervals table from a list of tuples.
    """
    for row in arr:
        tables.iedges.add_row(
            child_left=row[0],
            child_right=row[1],
            parent_left=row[2],
            parent_right=row[3],
            child=row[4],
            parent=row[5],
        )
    return tables.iedges


def iedge(cl, cr, pl, pr, *, c, p):
    """
    A helper function to quickly make an IEdgeTableRow.
    """
    return gigl.tables.IEdgeTableRow(cl, cr, pl, pr, child=c, parent=p)


def make_nodes_table(arr, tables):
    """
    Make a nodes table from a list of tuples.
    """
    for row in arr:
        tables.nodes.add_row(time=row[0], flags=row[1], individual=gigl.NULL)
    return tables.nodes


class DTWF_simulator:
    """
    A simple base simulator class, mostly code from
    https://tskit.dev/tutorials/forward_sims.html
    """

    # For visualising unsimplified tree sequences, it helps to flag all nodes as samples
    default_node_flags = gigl.NODE_IS_SAMPLE

    def make_diploids(self, time, parent_individuals):
        """
        Make a set of individuals with their diploid genomes by adding to tables,
        returning two arrays of IDs: the individual IDs and the node IDs.
        """
        individual_ids = self.tables.individuals.add_rows(parents=parent_individuals)
        return (
            individual_ids,
            self.tables.nodes.add_rows(
                time=time,
                flags=self.default_node_flags,
                individual=np.broadcast_to(individual_ids, (2, len(individual_ids))).T,
            ),
        )

    def new_population(self, time, recombination_rate=0, size=None, seq_len=None):
        """
        If seq_len is specified, use this as the expected sequence length of the
        parents of the new population, otherwise take from the parent.sequence_length

        If num_diploids is specified, use this as the number of diploids in the new
        population, otherwise use the number of diploids in the previous population.
        """
        if self.pop is None:
            raise ValueError("Need to initialise a new population first")
        prev_pop = self.pop
        self.pop = {}  # fill with individualID: (maternal_genomeID, paternal_genomeID)

        # Cache the list of individual IDs in the previous population, for efficiency
        prev_individuals = np.array([i for i in prev_pop.keys()], dtype=np.int32)

        if size is None:
            size = len(prev_pop)

        # 1. Pick individual parent ID pairs at random, `replace=True` allows selfing
        mum_dad_arr = self.random.choice(prev_individuals, (size, 2), replace=True)
        # 2. Get new individual IDs + twice as many new node IDs
        child_id_arr, child_genomes_arr = self.make_diploids(time, mum_dad_arr)
        assert len(mum_dad_arr) == len(child_id_arr) == len(child_genomes_arr)
        for mother_and_father, child_id, child_genomes in zip(
            mum_dad_arr, child_id_arr, child_genomes_arr
        ):
            # assert (self.tables.nodes[child_genomes[0]].individual ==
            # self.tables.nodes[child_genomes[1]].individual)  # removed for speed
            self.pop[child_id] = child_genomes  # store the genome IDs

            # 3. Add inheritance paths to both child genomes
            for child_genome, parent_individual in zip(
                child_genomes, mother_and_father
            ):
                parent_genomes = prev_pop[parent_individual]
                self.add_inheritance_paths(
                    parent_genomes, child_genome, seq_len, recombination_rate
                )

    def add_inheritance_paths(
        self, parent_nodes, child_node, seq_len, recombination_rate
    ):
        raise NotImplementedError(
            "Implement an add_inheritance_paths method to make a DTWF simulator"
        )

    def initialise_population(self, time, size) -> dict[int, tuple[int, int]]:
        self.tables.clear()
        self.pop = None
        return dict(
            zip(
                *self.make_diploids(
                    time,
                    parent_individuals=np.array([], dtype=int).reshape(size, 0),
                )
            )
        )

    def __init__(self, use_validation=True):
        """
        Create a simulation class, which can then simulate a forward Discrete Time
        Wright-Fisher model with varying population sizes over generations.

        If use_validation is True (default), when adding edges we check that they
        create a valid set of tables suitable for using the find_mrca_regions() method.
        Setting this to False saves a small fraction (about 2%) of the simulation time,
        at the expense of not doing any validation (dangerous!)
        """
        self.tables = gigl.Tables()
        self.tables.time_units = "generations"  # optional, but helpful when plotting
        self.use_validation = use_validation

    def add_iedge_params(self):
        """
        Return the validation params to use when calling tables.add_iedge_row
        """
        return {
            "validate_child_adjacency": self.use_validation,
            "validate_intervals": self.use_validation,
            "validate_node_times": self.use_validation,
        }

    def run(self, num_diploids, seq_len, gens, *, random_seed=None):
        """
        Initialise and run a new population for a given number of generations. The last
        generation will be at time 0 and the first at time `gens`.

        The num_diploids param can be an array of length `gens + 1` giving the diploid
        population size in each generation. This allows quick growth of a population

        Returns a gig represeting the ancestry of the population at time 0
        """
        self.random = np.random.default_rng(random_seed)
        if isinstance(num_diploids, int):
            num_diploids = [num_diploids] * (gens + 1)

        self.pop = self.initialise_population(gens, num_diploids[-gens - 1])
        # First generation by hand, so that we can specify the sequence length
        gens -= 1
        self.new_population(gens, seq_len=100, size=num_diploids[-gens - 1])

        # Subsequent generations
        while gens > 0:
            gens -= 1
            self.new_population(gens, size=num_diploids[-gens - 1])

        self.tables.sort()
        # We should probably simplify or at least sample_resolve here?
        # Probably a parameter `simplify` would be useful?
        return self.tables.copy().graph()

    def run_more(self, num_diploids, seq_len, gens, random_seed=None):
        """
        The num_diploids parameter can be an array of length `gens` giving the diploid
        population size in each generation.
        """
        if self.pop is None:
            raise ValueError("Need to call run() first")
        if random_seed is not None:
            self.random = np.random.default_rng(random_seed)
        if isinstance(num_diploids, int):
            num_diploids = [num_diploids] * (gens)

        # augment the generations
        self.tables.change_times(gens)
        while gens > 0:
            gens -= 1
            self.new_population(gens, size=num_diploids[-gens - 1])

        self.tables.sort()
        return self.tables.copy().graph()


class DTWF_no_recombination_sim(DTWF_simulator):
    def add_inheritance_paths(self, parent_nodes, child, seq_len, recombination_rate):
        "Add inheritance paths from a randomly chosen parent genome to the child genome."
        if recombination_rate != 0:
            raise ValueError("Recombination rate must be zero for this simulation.")
        rand_parent = self.random.integers(2)  # randomly choose 1st or 2nd parent node
        parent = parent_nodes[rand_parent]
        lft = 0
        if seq_len is not None:
            rgt = seq_len
        else:
            rgt = np.max(
                self.tables.iedges.child_right[self.tables.iedges.child == parent]
            )

        self.tables.add_iedge_row(
            lft,
            rgt,
            lft,
            rgt,
            child=child,
            parent=parent,
            **self.add_iedge_params(),
        )


class DTWF_one_break_no_rec_inversions_slow_sim(DTWF_simulator):
    """
    A simple DTWF simulator with recombination, in which we do not allow recombination
    between regions which are in inverted orientation relative to each other (we simply
    pick a different breakpoint)

    We pick one breakpoint per meiosis, but it would be possible to modify the code
    easily enough to have any number of breakpoints. Interference between
    breakpoints is more tricky (2 randomly chosen breakpoints could be arbitrarily
    close together), but could be implemented by rejection sampling.:
    """

    def initialise_population(self, time, size, L) -> dict[int, tuple[int, int]]:
        # Make a "fake" MRCA node so we can use the find_mrca_regions method
        # to locate comparable regions for recombination in the parent genomes
        self.tables.clear()
        self.pop = None
        self.grand_mrca = self.tables.nodes.add_row(time=np.inf)
        temp_pop = dict(
            zip(
                *self.make_diploids(
                    time,
                    parent_individuals=np.array([], dtype=int).reshape(size, 0),
                )
            )
        )
        for nodes in temp_pop.values():
            for u in nodes:
                self.tables.add_iedge_row(
                    0,
                    L,
                    0,
                    L,
                    child=u,
                    parent=self.grand_mrca,
                    **self.add_iedge_params(),
                )
        return temp_pop

    def tables_to_gig_without_grand_mrca(self):
        # Remove the grand MRCA node at time np.inf, plus all edges to it.
        # This requires a bit rearrangement to the tables so we make a set of
        # new ones
        output_tables = gigl.Tables()
        output_tables.time_units = self.tables.time_units
        node_map = {}
        for individual in self.tables.individuals:
            # No need to change the individuals: the grand MRCA has no associated
            # individuals, and the existing individuals do not reference node IDs
            output_tables.individuals.append(individual)
        for i, node in enumerate(self.tables.nodes):
            if np.isfinite(node.time):
                node_map[i] = output_tables.nodes.append(node)
        for ie in self.tables.iedges:
            try:
                output_tables.add_iedge_row(
                    ie.child_left,
                    ie.child_right,
                    ie.parent_left,
                    ie.parent_right,
                    child=node_map[ie.child],
                    parent=node_map[ie.parent],
                )
            except KeyError:
                assert ie.parent == self.grand_mrca
        # Shouldn't need to sort really: we could do as a check?
        # No need to make a copy here, as we have started with a new tables object
        return output_tables.graph()

    def run(self, num_diploids, seq_len, gens, *, random_seed=None):
        """
        The num_diploids param can be an array of length `gens + 1` giving the diploid
        population size in each generation. This allows quick growth of a population
        """
        self.random = np.random.default_rng(random_seed)
        self.num_tries_for_breakpoint = 20  # number of tries to find a breakpoint
        if isinstance(num_diploids, int):
            num_diploids = [num_diploids] * (gens + 1)
        self.pop = self.initialise_population(
            gens, size=num_diploids[-gens - 1], L=seq_len
        )
        while gens > 0:
            gens -= 1
            self.new_population(gens, size=num_diploids[-gens - 1])
        return self.tables_to_gig_without_grand_mrca()

    def run_more(self, num_diploids, gens, random_seed=None):
        """
        The num_diploids parameter can be an array of length `gens` giving the diploid
        population size in each generation.
        """
        if self.pop is None:
            raise ValueError("Need to call run() first")
        if random_seed is not None:
            self.random = np.random.default_rng(random_seed)
        if isinstance(num_diploids, int):
            num_diploids = [num_diploids] * (gens)
        self.tables.change_times(gens)
        while gens > 0:
            gens -= 1
            self.new_population(gens, size=num_diploids[-gens - 1])
        return self.tables_to_gig_without_grand_mrca()

    def find_comparable_points(self, tables, parent_nodes):
        """
        Find comparable points in the parent nodes, and return the
        coordinates of the matching regions in the parent nodes.
        """
        mrcas = tables.find_mrca_regions(*parent_nodes)
        comparable_pts = tables.random_matching_positions(mrcas, self.random)
        # Pick a single breakpoint: if both breaks are inverted relative to the mrca
        # (i.e. negative) it's OK: both have the same orientation relative to each other
        tries = 0
        while comparable_pts[0] * comparable_pts[1] < 0:
            comparable_pts = tables.random_matching_positions(mrcas, self.random)
            tries += 1
            if tries > self.num_tries_for_breakpoint:
                raise ValueError(
                    "Could not find a pair of matching regions in the same orientation"
                )
        return comparable_pts

    def add_inheritance_paths(self, parent_nodes, child, seq_len, _):
        comparable_pts = self.find_comparable_points(self.tables, parent_nodes)

        rnd = self.random.integers(4)
        break_to_right_of_position = rnd & 1
        # Minor hack when both comparable_pts are negative, in which case
        # the positions mark the right of the break, rather than the left
        breaks = [abs(b + break_to_right_of_position) for b in comparable_pts]

        if rnd & 2:  # Use 2nd bit to randomly choose 1st or 2nd parent node
            # We need to randomise the order of parent nodes to avoid
            # all children of this parent having the same genome to left / right
            lft_parent, rgt_parent = parent_nodes
            lft_parent_break, rgt_parent_break = breaks
        else:
            rgt_parent, lft_parent = parent_nodes
            rgt_parent_break, lft_parent_break = breaks

        # Must add edges in the correct order to preserve edge sorting by left coord
        brk = lft_parent_break
        if brk > 0:  # If break not placed just before position 0
            pL, pR = 0, brk  # parent left and right start at the same pos as child
            self.tables.add_iedge_row(
                0,
                brk,
                pL,
                pR,
                child=child,
                parent=lft_parent,
                **self.add_iedge_params(),
            )
        if seq_len is None:
            # TODO - make this more efficient, as all the edges should be adjacent
            seq_len = np.max(
                self.tables.iedges.child_right[self.tables.iedges.child == rgt_parent]
            )
        if rgt_parent_break < seq_len:  # If break not just after the last pos
            pL, pR = rgt_parent_break, seq_len
            cR = brk + (pR - pL)  # child rgt must account for len of rgt parent region
            self.tables.add_iedge_row(
                brk,
                cR,
                pL,
                pR,
                child=child,
                parent=rgt_parent,
                **self.add_iedge_params(),
            )
