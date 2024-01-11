import GeneticInheritanceGraphLibrary as gigl
import numpy as np

# Utilities for creating and editing gigs


def make_iedges_table(arr, tables):
    """
    Make an intervals table from a list of tuples.
    """
    for row in arr:
        tables.iedges.add_row(
            parent=row[0],
            child=row[1],
            child_left=row[2],
            child_right=row[3],
            parent_left=row[4],
            parent_right=row[5],
        )
    return tables.iedges


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
        generations = generations - 1
        pop = self.new_population(generations, pop, seq_len=100)
        while generations > 0:
            generations = generations - 1
            pop = self.new_population(generations, pop)

        self.tables.sort()
        return self.tables.graph()


class DTWF_no_recombination_sim(DTWF_simulator):
    def add_inheritance_paths(
        self, parent_nodes, child_node, seq_len, recombination_rate
    ):
        "Add inheritance paths from a randomly chosen parent genome to the child genome."
        if recombination_rate != 0:
            raise ValueError("Recombination rate must be zero for this simulation.")
        inherit_from = self.random.integers(2)  # randomly choose 1st or 2nd parent node
        parent = parent_nodes[inherit_from]
        left = 0
        if seq_len is not None:
            right = seq_len
        else:
            right = np.max(
                self.tables.iedges.child_right[self.tables.iedges.child == parent]
            )
        self.tables.iedges.add_row(
            parent=parent_nodes[inherit_from],
            child=child_node,
            parent_left=left,
            parent_right=right,
            child_left=left,
            child_right=right,
        )
