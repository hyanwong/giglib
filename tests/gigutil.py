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
            mother_and_father = self.random.choice(prev_individuals, 2, replace=True)

            # 2. Get 1 new individual ID + 2 new node IDs
            child_id, child_genomes = self.make_diploid(time, mother_and_father)
            pop[child_id] = child_genomes  # store the genome IDs

            # 3. Add inheritance paths to both child genomes
            for child_genome, parent_individual in zip(
                child_genomes, mother_and_father
            ):
                parent_genomes = prev_pop[parent_individual]
                self.add_inheritance_paths(
                    parent_genomes, child_genome, seq_len, recombination_rate
                )
        return pop

    def add_inheritance_paths(
        self, parent_nodes, child_node, seq_len, recombination_rate
    ):
        raise NotImplementedError(
            "Implement an add_inheritance_paths method to make a DTWF simulator"
        )

    def initialise_population(self, time, size) -> dict[int, tuple[int, int]]:
        # Just return a dictionary by repeating step 2 above
        return dict(self.make_diploid(time) for _ in range(size))

    def __init__(self):
        self.tables = gigl.Tables()
        self.tables.time_units = "generations"  # optional, but helpful when plotting

    def run(self, num_diploids, seq_len, generations, random_seed=None):
        self.random = np.random.default_rng(random_seed)
        pop = self.initialise_population(generations, num_diploids)
        # First generation by hand, so that we can specify the sequence length
        generations = generations - 1
        pop = self.new_population(generations, pop, seq_len=100)

        # Subsequent generations
        while generations > 0:
            generations = generations - 1
            pop = self.new_population(generations, pop)

        self.tables.sort()
        return self.tables.graph()


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
        self.tables.iedges.add_row(lft, rgt, lft, rgt, child=child, parent=parent)


class DTWF_one_recombination_no_SV_slow_sim(DTWF_simulator):
    """
    A simple DTWF simulator with recombination but no structural variants.

    We pick one breakpoint per meiosis, but it would be possible to modify the code
    easily enough to have any number of breakpoints. Interference between
    breakpoints is more tricky (2 randomly chosen breakpoints could be arbitrarily
    close together), but could be implemented by rejection sampling.:
    """

    def initialise_population(self, time, size, L) -> dict[int, tuple[int, int]]:
        # Make a "fake" MRCA node so we can use the find_mrca_regions method
        # to locate comparable regions for recombination in the parent genomes
        self.grand_mrca = self.tables.nodes.add_row(time=np.inf)

        temp_pop = dict(self.make_diploid(time) for _ in range(size))
        for nodes in temp_pop.values():
            for u in nodes:
                self.tables.iedges.add_row(0, L, 0, L, child=u, parent=self.grand_mrca)
        return temp_pop

    def run(
        self, num_diploids, seq_len, gens, *, left_sort_mrcas=False, random_seed=None
    ):
        """
        if left_pos_sort_mrcas is True, then before calling gig.random_matching_positions
        the mrcas list will be sorted by the left position of the mrca region (rather
        than using the ID of the MRCA node). This is useful for testing against a simpler
        model that picks a breakpoint from left to right along the matching regions.
        """
        self.random = np.random.default_rng(random_seed)
        self.left_sort_mrcas = left_sort_mrcas
        pop = self.initialise_population(gens, num_diploids, seq_len)
        while gens > 0:
            gens = gens - 1
            # Since we do this in generations, we can freeze the tables into a GIG
            # each generation. This is an inefficient way to be able to run
            # find_mrca_regions, but it will have to do until we solve
            # https://github.com/hyanwong/GeneticInheritanceGraphLibrary/issues/69

            # Since we are not sorting, running .graph() here also helps us catch any
            # bugs dues to e.g. not adding edges in the expected order. Therefore
            # if this
            self.gig = self.tables.graph()
            pop = self.new_population(gens, pop)

        # Remove the grand MRCA node at time np.inf, plus all edges to it.
        output_tables = gigl.Tables()
        output_tables.time_units = self.tables.time_units
        node_map = {}
        for individual in self.gig.individuals:
            # No need to change the individuals: the grand MRCA has no associated
            # individuals, and the existing individuals do not reference node IDs
            output_tables.individuals.append(individual)
        for node in self.gig.nodes:
            if np.isfinite(node.time):
                node_map[node.id] = output_tables.nodes.append(node)
        for ie in self.gig.iedges:
            try:
                output_tables.iedges.add_row(
                    ie.child_left,
                    ie.child_right,
                    ie.parent_left,
                    ie.parent_right,
                    child=node_map[ie.child],
                    parent=node_map[ie.parent],
                )
            except KeyError:
                assert ie.parent == self.grand_mrca
        return output_tables.graph()

    def add_inheritance_paths(self, parent_nodes, child, seq_len, _):
        mrcas = self.gig.find_mrca_regions(*parent_nodes)

        if self.left_sort_mrcas:
            mrcas = self.sort_mrcas_by_left_coord(
                mrcas
            )  # Purely for testing: see docstring

        # pick a single breakpoint
        comparable_pts = self.gig.random_matching_positions(mrcas, self.random)

        if comparable_pts[0] * comparable_pts[1] < 0:
            raise ValueError("Recombination between inverted regions not yet supported")
            # NB: we could assume that a breakpoint between regions of interted
            # orientation leads to a nonviable embryo, and reject this meiosis
        # if both breaks are inverted relative to the mrca (i.e. negative) that's fine,
        # as they are both in the same orientation relative to each other. We hack it
        # here because if negative, the positions are out-by-one
        rnd = self.random.integers(4)
        break_to_right_of_position = rnd & 1
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
            self.tables.iedges.add_row(0, brk, pL, pR, child=child, parent=lft_parent)
        if seq_len is None:
            # TODO - make this more efficient, as all the edges should be adjacent
            seq_len = np.max(
                self.tables.iedges.child_right[self.tables.iedges.child == rgt_parent]
            )
        if rgt_parent_break < seq_len:  # If break not just after the last pos
            pL, pR = rgt_parent_break, seq_len
            cR = brk + (pR - pL)  # child rgt must account for len of rgt parent region
            self.tables.iedges.add_row(brk, cR, pL, pR, child=child, parent=rgt_parent)

    @staticmethod
    def sort_mrcas_by_left_coord(mrcas):
        """
        This is only used for testing: it creates a new mrca dict with arbitrary keys
        but where each value is a single interval with the appropriate matching coords in
        u and v. The items in the dict are sorted by the left coordinate of the mrca.
        The keys can be arbitray because we don;t use the identity of the MRCA node to
        determine breakpoint dynamics.
        """
        tmp = []
        for mrca_regions in mrcas.values():
            for region, equivalents in mrca_regions.items():
                tmp.append((region, equivalents))
        return {
            k: {v[0]: v[1]} for k, v in enumerate(sorted(tmp, key=lambda x: x[0][0]))
        }
