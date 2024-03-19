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


def add_iedge(tables, cl, cr, pl, pr, *, c, p, **kwargs):
    """
    A helper function to make an IEdgeTableRow using 'c' and 'p' abbreviations.
    """
    return tables.add_iedge_row(cl, cr, pl, pr, child=c, parent=p, **kwargs)


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

    def make_diploids(self, time, parent_individuals, node_flags=None):
        """
        Make a set of individuals with their diploid genomes by adding to tables,
        returning two arrays of IDs: the individual IDs and the node IDs.
        """
        individual_ids = self.tables.individuals.add_rows(parents=parent_individuals)
        if node_flags is None:
            node_flags = self.default_node_flags
        return (
            individual_ids,
            self.tables.nodes.add_rows(
                time=time,
                flags=node_flags,
                individual=np.broadcast_to(individual_ids, (2, len(individual_ids))).T,
            ),
        )

    def new_population(
        self, time, recombination_rate=0, size=None, seq_len=None, node_flags=None
    ):
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
        mum_dad_arr = self.rng.choice(prev_individuals, (size, 2), replace=True)
        # 2. Get new individual IDs + twice as many new node IDs
        child_id_arr, child_genomes_arr = self.make_diploids(
            time, mum_dad_arr, node_flags
        )
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

    def initialise_population(
        self, time, size, node_flags=None
    ) -> dict[int, tuple[int, int]]:
        self.tables.clear()
        self.pop = None
        return dict(
            zip(
                *self.make_diploids(
                    time,
                    parent_individuals=np.array([], dtype=int).reshape(size, 0),
                    node_flags=node_flags,
                )
            )
        )

    def __init__(self, skip_validate=False):
        """
        Create a simulation class, which can then simulate a forward Discrete Time
        Wright-Fisher model with varying population sizes over generations.

        If skip_validate is False (default), when adding edges we check that they
        create a valid set of tables suitable for using the find_mrca_regions() method.
        Setting this to False saves a small fraction (about 2%) of the simulation time,
        at the expense of not doing any validation (dangerous!)
        """
        self.tables = gigl.Tables()
        self.tables.time_units = "generations"  # optional, but helpful when plotting
        self.skip_validate = skip_validate

    def add_iedge_params(self):
        """
        Return the validation params to use when calling tables.add_iedge_row
        """
        return {
            "skip_validate": self.skip_validate,
            "validate": gigl.constants.ValidFlags.IEDGES_ALL,
        }

    def run(
        self,
        num_diploids,
        seq_len,
        gens,
        *,
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
        if isinstance(num_diploids, int):
            num_diploids = [num_diploids] * (gens + 1)

        self.pop = self.initialise_population(
            gens, num_diploids[-gens - 1], node_flags=initial_node_flags
        )
        # First generation by hand, so that we can specify the sequence length
        gens -= 1
        self.new_population(
            gens,
            seq_len=100,
            size=num_diploids[-gens - 1],
            node_flags=further_node_flags,
        )

        # Subsequent generations
        while gens > 0:
            gens -= 1
            self.new_population(
                gens, size=num_diploids[-gens - 1], node_flags=further_node_flags
            )

        self.tables.sort()
        # We should probably simplify or at least sample_resolve here?
        # We should also mark gen 0 as samples and unmark the others.
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


class DTWF_no_recombination_sim(DTWF_simulator):
    def add_inheritance_paths(self, parent_nodes, child, seq_len, recombination_rate):
        "Add inheritance paths from a randomly chosen parent genome to the child genome."
        if recombination_rate != 0:
            raise ValueError("Recombination rate must be zero for this simulation.")
        rand_parent = self.rng.integers(2)  # randomly choose 1st or 2nd parent node
        parent = parent_nodes[rand_parent]
        lft = 0
        if seq_len is not None:
            rgt = seq_len
        else:
            rgt = self.tables.iedges.max_child_pos(parent)

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
    pick a different breakpoint). The simulator is slow, partially because of the large
    number of recombination breakpoints generated (which does not depend on seq len)

    We pick one breakpoint per meiosis, which essentially treats the genome as a whole
    chromosome regardless of the sequence length. Note that this means we are likely to
    have very large numbers of recombinations in the ancestry. It would easily be
    possible to modify the code to have more or fewer breakpoints. Interference between
    breakpoints is more tricky (2 randomly chosen breakpoints could be arbitrarily
    close together), but could be implemented by rejection sampling.
    """

    def initialise_population(
        self, time, size, L, node_flags=None
    ) -> dict[int, tuple[int, int]]:
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
                    node_flags=node_flags,
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
        # This requires a bit rearrangement to the tables, so we make a set of
        # new ones. We also mark only the most recent generation as samples.
        output_tables = gigl.Tables()
        output_tables.time_units = self.tables.time_units
        node_map = {}
        for individual in self.tables.individuals:
            # No need to change the individuals: the grand MRCA has no associated
            # individuals, and the existing individuals do not reference node IDs
            output_tables.individuals.append(individual)
        for i, node in enumerate(self.tables.nodes):
            if np.isfinite(node.time):
                if node.time != 0:
                    flags = node.flags & ~gigl.NODE_IS_SAMPLE
                else:
                    flags = node.flags | gigl.NODE_IS_SAMPLE
                node_map[i] = output_tables.nodes.add_row(
                    time=node.time, flags=flags, individual=node.individual
                )
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

    def run(
        self,
        num_diploids,
        seq_len,
        gens,
        *,
        random_seed=None,
        initial_node_flags=None,
        further_node_flags=None,
    ):
        """
        The num_diploids param can be an array of length `gens + 1` giving the diploid
        population size in each generation. This allows quick growth of a population
        """
        self.rng = np.random.default_rng(random_seed)
        self.num_tries_for_breakpoint = 20  # number of tries to find a breakpoint
        if isinstance(num_diploids, int):
            num_diploids = [num_diploids] * (gens + 1)
        self.pop = self.initialise_population(
            gens, size=num_diploids[-gens - 1], L=seq_len, node_flags=initial_node_flags
        )
        while gens > 0:
            gens -= 1
            self.new_population(
                gens, size=num_diploids[-gens - 1], node_flags=further_node_flags
            )
        return self.tables_to_gig_without_grand_mrca()

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
        # Pick a single comparable location but ban recombination if one is
        # inverted and the other is not inverted
        tries = 0
        while (pts := mrcas.random_match_pos(self.rng)).opposite_orientations:
            tries += 1
            if tries > self.num_tries_for_breakpoint:
                raise ValueError(
                    "Could not find a pair of matching regions in the same orientation"
                    f"after {tries} tries"
                )
        return np.array([pts.u, pts.v], dtype=np.int64)

    def add_inheritance_paths(self, parent_nodes, child, seq_len, _):
        breaks = self.find_comparable_points(self.tables, parent_nodes)

        rnd = self.rng.integers(4)
        # To avoid bias we choose a breakpoint either to the left or the right
        # of the identified location. If one was inverted and the other not,
        # we would need to be careful here, but this is banned in this sim.
        # Here's an example:
        #     0123456789
        # u = abcgfedhij
        # v = ABCGFEDHIJ
        # We label the potential positions for breaks as 0..10 (inclusive)
        # SEQUENCE       a   b   c   g   f   e   d   h   i   j
        # BREAKPOINT   0   1   2   3   4   5   6   7   8   9   10
        # So if we get a breakpoint at 0 or 10 we can ignore it
        break_to_right_of_position = rnd & 1
        breaks += break_to_right_of_position

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
            seq_len = self.tables.iedges.max_child_pos(rgt_parent)
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
