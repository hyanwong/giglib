"""
Tests for giglib.tools.alignment: tree sequences whose coordinates are alignment columns
"""

import msprime
import numpy as np
import pytest
import tskit

import giglib as gigl
from giglib.tools import alignment

BASES = "ACGT"


def simple_ts():
    """
    Three haploid samples (0, 1, 2) with no recombination, and sequence length 20:
    a SNP at 2, a deletion at 5 (above the parent of 0), and an insertion at 12, which
    node 1 has with known bases, and node 2 has with unknown bases.
    """
    ts = msprime.sim_ancestry(3, sequence_length=20, ploidy=1, random_seed=5)
    tables = ts.dump_tables()
    tree = ts.first()
    site = tables.sites.add_row(position=2, ancestral_state="A")
    tables.mutations.add_row(site=site, node=0, derived_state="C")
    site = tables.sites.add_row(position=5, ancestral_state="ATCG")
    tables.mutations.add_row(site=site, node=tree.parent(0), derived_state="A---")
    site = tables.sites.add_row(position=12, ancestral_state="G---")
    tables.mutations.add_row(site=site, node=1, derived_state="GTTA")
    tables.mutations.add_row(site=site, node=2, derived_state="GNN-")
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


def random_indel_ts(seed, *, num_samples=6, sequence_length=200, max_site_length=5):
    """
    A tree sequence with recombination, and sites of several columns, with random
    nested mutations: deletions, insertions (with known or unknown bases), and SNPs.
    Sites do not overlap, and do not cross tree boundaries.
    """
    rng = np.random.default_rng(seed)
    ts = msprime.sim_ancestry(
        num_samples,
        ploidy=1,
        sequence_length=sequence_length,
        recombination_rate=0.005,
        random_seed=seed,
    )
    tables = ts.dump_tables()
    end = 0
    for tree in ts.trees():
        position = max(int(tree.interval.left), end)
        while True:
            position += int(rng.integers(1, 15))
            length = int(rng.integers(1, max_site_length + 1))
            if position + length > tree.interval.right:
                break
            add_random_site(tables, tree, position, length, rng)
            end = position = position + length
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


def add_random_site(tables, tree, position, length, rng):
    ancestral = "".join("-" if rng.random() < 0.3 else rng.choice(list(BASES)) for _ in range(length))
    site = tables.sites.add_row(position=position, ancestral_state=ancestral)
    nodes = [u for u in tree.nodes() if tree.parent(u) != tskit.NULL]
    chosen = rng.choice(nodes, size=min(3, len(nodes)), replace=False)
    # Times between each node and its parent: add the oldest first
    mutations = []
    for u in chosen:
        low, high = tree.time(u), tree.time(tree.parent(u))
        mutations.append((low + (high - low) * rng.uniform(0.1, 0.9), int(u)))
    state_above = {}  # (node, time) of the mutations added so far, with their states
    for time, u in sorted(mutations, reverse=True):
        inherited = ancestral
        v = u
        while v != tskit.NULL:  # the most recent older mutation above u (or at u)
            candidates = [(t, s) for (w, t), s in state_above.items() if w == v and t > time]
            if candidates:
                inherited = min(candidates)[1]
                break
            v = tree.parent(v)
        derived = mutate(inherited, rng)
        if derived != inherited:
            tables.mutations.add_row(site=site, node=u, derived_state=derived, time=time)
            state_above[(u, time)] = derived


def mutate(state, rng):
    state = list(state)
    j = int(rng.integers(len(state)))
    kind = rng.choice(["deletion", "insertion", "snp"])
    if kind == "deletion":
        for k in range(j, min(len(state), j + int(rng.integers(1, 3)))):
            state[k] = "-"
    elif state[j] == "-":  # insertion of a known or unknown base
        state[j] = rng.choice(list(BASES + "N"))
    else:
        state[j] = rng.choice([b for b in BASES if b != state[j]])
    return "".join(state)


def assert_haplotypes_match(gig, ts):
    for u, row in zip(ts.samples(), alignment.alignments(ts)):
        expected = row.replace("-", "")
        # A sequence length from the iedges excludes inserted positions at the end of a
        # genome (which have no parent), so give the length explicitly
        assert gig.sequence_length(u) <= len(expected)
        assert gig.haplotype(u, right=len(expected)) == expected


class TestAlignments:
    def test_simple(self):
        assert alignment.alignments(simple_ts()) == [
            "NNCNNA---NNNG---NNNN",
            "NNANNATCGNNNGTTANNNN",
            "NNANNA---NNNGNN-NNNN",
        ]

    def test_missing_data_character(self):
        rows = alignment.alignments(simple_ts(), missing_data_character="?")
        assert rows[0] == "??C??A---???G---????"
        assert rows[2] == "??A??A---???GNN-????"  # N is an allele here, not missing data

    def test_single_character_sites_match_tskit(self):
        ts = msprime.sim_ancestry(4, sequence_length=50, recombination_rate=0.01, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.05, random_seed=1)
        assert ts.num_sites > 0
        assert alignment.alignments(ts) == list(ts.alignments(missing_data_character="N"))


class TestSplitSites:
    def test_single_character_sites_unchanged(self):
        ts = msprime.sim_ancestry(4, sequence_length=50, recombination_rate=0.01, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.05, random_seed=1)
        split = alignment.split_sites(ts)
        assert split.tables.sites == ts.tables.sites
        assert split.tables.mutations == ts.tables.mutations

    def test_split(self):
        split = alignment.split_sites(simple_ts())
        assert split.num_sites == 1 + 4 + 4
        assert all(len(site.ancestral_state) == 1 for site in split.sites())
        # The deletion becomes 3 mutations, the known insertion 3, the unknown one 2
        assert split.num_mutations == 1 + 3 + 3 + 2

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_genotypes_unchanged(self, seed):
        ts = random_indel_ts(seed)
        assert ts.num_sites > 5
        assert list(alignment.split_sites(ts).alignments(missing_data_character="?")) == alignment.alignments(
            ts, missing_data_character="?"
        )


class TestFromTreeSequence:
    def test_simple(self):
        ts = simple_ts()
        gig = alignment.from_tree_sequence(ts)
        assert_haplotypes_match(gig, ts)
        # The deletion is above node 3 (the parent of 0), so 3 has an iedge from 4 with
        # different coordinates in the child (from 6) and the parent (from 9)
        assert any(ie.child_left == 6 and ie.parent_left == 9 for ie in gig.iedges_for_child(3))
        # Unknown inserted bases have no mutations, so are missing data
        assert gig.haplotype(2) == "NNANNANNNGNNNNNN"

    @pytest.mark.parametrize("seed", range(1, 9))
    def test_random(self, seed):
        ts = random_indel_ts(seed)
        assert ts.num_trees > 1
        assert any(
            "-" in site.ancestral_state or any("-" in m.derived_state for m in site.mutations) for site in ts.sites()
        )
        gig = alignment.from_tree_sequence(ts)
        assert_haplotypes_match(gig, ts)

    def test_no_gaps_matches_graph_from_tree_sequence(self):
        ts = msprime.sim_ancestry(4, sequence_length=50, recombination_rate=0.01, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.05, random_seed=1)
        assert alignment.from_tree_sequence(ts).tables == gigl.Graph.from_tree_sequence(ts).tables

    def test_iedges_only_break_at_indels(self):
        # Columns where both child and parent have a gap don't break an iedge
        gig = alignment.from_tree_sequence(simple_ts())
        assert len(gig.iedges_for_child(0)) == 1  # 0 and its parent 3 both lack 6-8 and 13-15


class TestErrors:
    @staticmethod
    def ts_with_sites(sites, *, recombination_rate=0):
        ts = msprime.sim_ancestry(3, sequence_length=20, ploidy=1, recombination_rate=recombination_rate, random_seed=5)
        tables = ts.dump_tables()
        for position, ancestral, derived in sites:
            site = tables.sites.add_row(position=position, ancestral_state=ancestral)
            tables.mutations.add_row(site=site, node=0, derived_state=derived)
        tables.sort()
        return tables.tree_sequence()

    @pytest.mark.parametrize("function", [alignment.from_tree_sequence, alignment.alignments, alignment.split_sites])
    def test_unequal_allele_lengths(self, function):
        with pytest.raises(ValueError, match="same length"):
            function(self.ts_with_sites([(5, "ATCG", "A")]))

    def test_overlapping_sites(self):
        with pytest.raises(ValueError, match="overlaps"):
            alignment.from_tree_sequence(self.ts_with_sites([(5, "ATCG", "A---"), (7, "C", "G")]))

    def test_beyond_end(self):
        with pytest.raises(ValueError, match="beyond the end"):
            alignment.from_tree_sequence(self.ts_with_sites([(18, "ATCG", "A---")]))

    def test_crosses_tree_boundary(self):
        ts = msprime.sim_ancestry(3, sequence_length=20, ploidy=1, recombination_rate=0.1, random_seed=5)
        boundary = int(ts.first().interval.right)
        assert boundary < 20
        tables = ts.dump_tables()
        site = tables.sites.add_row(position=boundary - 1, ancestral_state="AT")
        tables.mutations.add_row(site=site, node=0, derived_state="A-")
        tables.sort()
        with pytest.raises(ValueError, match="crosses a tree boundary"):
            alignment.from_tree_sequence(tables.tree_sequence())
