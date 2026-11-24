"""
A forward-time simulator with one crossover per meiosis, which avoids recombining across
inversions.
"""

import numpy as np

import giglib as gigl

from .dtwf import DTWF_simulator


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

    def initialise_population(self, time, size, seq_lens, node_flags=None) -> dict[int, tuple[int, int]]:
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
                for chrom, L in seq_lens.items():
                    self.add_iedge_row(
                        u,
                        self.grand_mrca,
                        child_right=L,
                        child_chromosome=chrom,
                        parent_chromosome=chrom,
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
                node_map[i] = output_tables.nodes.add_row(time=node.time, flags=flags, individual=node.individual)
        for ie in self.tables.iedges:
            try:
                output_tables.iedges.add_row(
                    node_map[ie.child],
                    node_map[ie.parent],
                    child_left=ie.child_left,
                    child_right=ie.child_right,
                    parent_left=ie.parent_left,
                    parent_right=ie.parent_right,
                    child_chromosome=ie.child_chromosome,
                    parent_chromosome=ie.parent_chromosome,
                )
            except KeyError:
                assert ie.parent == self.grand_mrca
        # Shouldn't need to sort really, as gMRCA is oldest we could do as a check?
        # No need to make a copy here, as we have started with a new tables object
        return output_tables.graph()

    def run(
        self,
        num_diploids,
        seq_lens,
        gens=None,
        *,
        num_chromosomes=1,
        random_seed=None,
        initial_node_flags=None,
        further_node_flags=None,
    ):
        """
        The num_diploids param can be an array of length `gens + 1` giving the diploid
        population size in each generation. This allows variable populations sizes over
        time, e.g. to simplement quick growth of a population.
        """
        self.rng = np.random.default_rng(random_seed)
        self.num_tries_for_breakpoint = 20  # number of tries to find a breakpoint
        try:
            seq_lens = {chrom: l for chrom, l in seq_lens.items()}
        except AttributeError:
            seq_lens = {chrom: l for chrom, l in enumerate(seq_lens)}
        if isinstance(num_diploids, int):
            num_diploids = [num_diploids] * (gens + 1)
        else:
            if gens is not None:
                raise ValueError("Cannot specify num gens with array of pop sizes")
            gens = len(num_diploids) - 1

        self.pop = self.initialise_population(
            gens,
            size=num_diploids[-gens - 1],
            seq_lens=seq_lens,
            node_flags=initial_node_flags,
        )
        while gens > 0:
            gens -= 1
            self.new_population(gens, size=num_diploids[-gens - 1], node_flags=further_node_flags)
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

    def find_comparable_points(self, tables, parent_nodes, parent_chroms):
        """
        Find comparable points in the parent nodes, and return the
        coordinates of the matching regions in the parent nodes.
        """
        u_chrom, v_chrom = parent_chroms
        mrcas = tables.find_mrca_regions(
            *parent_nodes,
            u_chromosomes=[u_chrom],
            v_chromosomes=[v_chrom],
        )
        # Pick a single comparable location but ban recombination if one is
        # inverted and the other is not inverted
        tries = 0
        while (pos := mrcas.random_match_pos(self.rng)).opposite_orientations:
            tries += 1
            if tries > self.num_tries_for_breakpoint:
                raise ValueError(
                    "Could not find a pair of matching regions in the same orientation" f"after {tries} tries"
                )
        return np.array([pos.u.position, pos.v.position], dtype=np.int64), (pos.u.chromosome, pos.v.chromosome)

    def add_inheritance_paths(self, parent_nodes, child, _):
        mum, dad = parent_nodes
        mum_chroms = self.tables.iedges.chromosomes_as_child(mum)
        if mum_chroms != self.tables.iedges.chromosomes_as_child(dad):
            raise ValueError("Parents must have the same chromosome IDs")
        for chrom in sorted(mum_chroms):
            chroms = (chrom, chrom)
            breaks, parent_chroms = self.find_comparable_points(self.tables, parent_nodes, chroms)
            assert parent_chroms[0] == chroms[0]
            assert parent_chroms[1] == chroms[1]
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
                lft_chrom, rgt_chrom = parent_chroms
                lft_parent_break, rgt_parent_break = breaks
            else:
                rgt_parent, lft_parent = parent_nodes
                rgt_chrom, lft_chrom = parent_chroms
                rgt_parent_break, lft_parent_break = breaks

                # Must add edges in correct order to preserve edge sorting by left coord
            brk = lft_parent_break
            if brk > 0:  # If break not placed just before position 0
                pL, pR = 0, brk  # parent left and right start at the same pos as child
                self.add_iedge_row(
                    child,
                    lft_parent,
                    child_left=0,
                    child_right=brk,
                    parent_left=pL,
                    parent_right=pR,
                    child_chromosome=chrom,
                    parent_chromosome=lft_chrom,
                )
            if not self.tables.iedges.edges_exist_for_child(rgt_parent, chromosome=rgt_chrom):
                raise ValueError("Must have edges for parents to get seq len")
            seq_len = self.tables.iedges.max_pos_as_child(rgt_parent, chromosome=rgt_chrom)
            if rgt_parent_break < seq_len:  # If break not just after the last pos
                pL, pR = rgt_parent_break, seq_len
                cR = brk + (pR - pL)  # child rgt must account for len of rgt parent region
                self.add_iedge_row(
                    child,
                    rgt_parent,
                    child_left=brk,
                    child_right=cR,
                    parent_left=pL,
                    parent_right=pR,
                    child_chromosome=chrom,
                    parent_chromosome=rgt_chrom,
                )
