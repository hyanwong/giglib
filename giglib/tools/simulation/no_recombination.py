"""
A forward-time simulator with no recombination: each child inherits each chromosome whole.
"""

from .dtwf import DTWF_simulator


class DTWF_no_recombination_sim(DTWF_simulator):
    """
    A discrete-time Wright-Fisher model in which no recombination takes place,
    but a random parent is chosen for each chromosome.
    """

    def add_inheritance_paths(self, parent_nodes, child, recombination_rate):
        "Add inheritance paths from a randomly chosen parent genome to the child genome."
        if recombination_rate != 0:
            raise ValueError("Recombination rate must be zero for this simulation.")
        mum, dad = parent_nodes
        if self.tables.iedges.edges_exist_for_child(mum) != self.tables.iedges.edges_exist_for_child(dad):
            raise ValueError("Cannot combine parents with and without previous ancestry")
        if self.tables.iedges.edges_exist_for_child(mum):
            # Has previous ancestry, so we need to check the chromosomes
            mum_chroms = self.tables.iedges.chromosomes_as_child(mum)
            if mum_chroms != self.tables.iedges.chromosomes_as_child(dad):
                raise ValueError("Parents must have the same chromosome IDs")
            chromosomes = mum_chroms
            rand_parent = self.rng.integers(2, size=len(chromosomes))
            seq_lens = {
                ch: self.tables.iedges.max_pos_as_child(parent_nodes[idx], chromosome=ch)
                for ch, idx in zip(chromosomes, rand_parent)
            }
        else:
            rand_parent = self.rng.integers(2, size=len(self.seq_lens))
            seq_lens = self.seq_lens
        lft = 0
        for idx, (chrom, rgt) in zip(rand_parent, seq_lens.items()):
            parent = parent_nodes[idx]
            self.add_iedge_row(
                child,
                parent,
                child_left=lft,
                child_right=rgt,
                child_chromosome=chrom,
                parent_chromosome=chrom,
            )
