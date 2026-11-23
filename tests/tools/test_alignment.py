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


def isolated_sample_ts(sites, *, sequence_length=6):
    """
    Samples 0 and 1 have parent 3 over the whole sequence, and sample 2 is isolated.
    Each site is (position, ancestral_state, [(node, derived_state), ...]), with the
    mutations listed oldest first.
    """
    tables = tskit.TableCollection(sequence_length=sequence_length)
    for _ in range(3):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.nodes.add_row(time=1)
    for child in (0, 1):
        tables.edges.add_row(left=0, right=sequence_length, parent=3, child=child)
    for position, ancestral, mutations in sites:
        site = tables.sites.add_row(position=position, ancestral_state=ancestral)
        for node, derived in mutations:
            tables.mutations.add_row(site=site, node=node, derived_state=derived)
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


def column_states(ts):
    """
    The state of each sample in each column, decoded directly from the variants (with
    missing data as None), for comparing tree sequences without using alignments()
    """
    states = [[None] * int(ts.sequence_length) for _ in range(ts.num_samples)]
    for variant in ts.variants():
        position = int(variant.site.position)
        for i, genotype in enumerate(variant.genotypes):
            allele = variant.alleles[genotype] if genotype != tskit.MISSING_DATA else None
            for j in range(len(variant.site.ancestral_state)):
                states[i][position + j] = None if allele is None else allele[j]
    return states


def assert_haplotypes_match(gig, ts):
    # Check with a missing data character other than N, so that missing data is
    # distinguished from N alleles: this checks that every inserted base (including an
    # unknown one, N) is given by a mutation, as required by the insertions design note
    for missing in ("N", "?"):
        for u, row in zip(ts.samples(), alignment.alignments(ts, missing_data_character=missing)):
            expected = row.replace("-", "")
            # A genome's length (from its iedges and mutations) excludes positions at the
            # end with neither, e.g. missing data in an isolated sample, so give the length
            # explicitly
            assert gig.sequence_length(u) <= len(expected)
            assert gig.haplotype(u, right=len(expected), missing_data_character=missing) == expected


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
        assert column_states(alignment.split_sites(ts)) == column_states(ts)

    def test_isolated_sample_unchanged_columns(self):
        # A mutation above an isolated sample establishes its state in every column
        ts = isolated_sample_ts([(0, "AC", [(2, "AT")])])
        assert column_states(alignment.split_sites(ts)) == column_states(ts)
        assert alignment.alignments(ts)[2] == "ATNNNN"


class TestFromTreeSequence:
    def test_simple(self):
        ts = simple_ts()
        gig = alignment.from_tree_sequence(ts)
        assert_haplotypes_match(gig, ts)
        # The deletion is above node 3 (the parent of 0), so 3 has an iedge from 4 with
        # different coordinates in the child (from 6) and the parent (from 9)
        assert any(ie.child_left == 6 and ie.parent_left == 9 for ie in gig.iedges_for_child(3))
        # Columns without sites have unknown states, and the unknown inserted bases of node
        # 2 are the allele N (so they are mutations, not missing data)
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


class TestEdgeCases:
    """
    Cases raised in review of the first version
    """

    def test_isolated_sample_unchanged_columns(self):
        ts = isolated_sample_ts([(0, "AC", [(2, "AT")])])
        gig = alignment.from_tree_sequence(ts)
        assert gig.haplotype(2, right=6) == "ATNNNN"

    def test_isolated_sample_unknown_is_not_a_gap(self):
        # Sample 2 is isolated, with no mutation at the first site, whose ancestral state is
        # a gap. Its state there is unknown (not a gap), so the column is kept.
        ts = isolated_sample_ts([(0, "-", []), (1, "A", [(2, "C")])])
        assert alignment.alignments(ts)[2] == "NCNNNN"
        gig = alignment.from_tree_sequence(ts)
        assert gig.haplotype(2, right=6) == "NCNNNN"

    def test_reference_sequence_ignored(self):
        tables = isolated_sample_ts([]).dump_tables()
        tables.reference_sequence.data = "ACGTAC"
        ts = tables.tree_sequence()
        assert ts.has_reference_sequence()
        assert alignment.alignments(ts) == ["NNNNNN"] * 3
        gig = alignment.from_tree_sequence(ts)
        assert [gig.haplotype(u, right=6) for u in ts.samples()] == ["NNNNNN"] * 3

    def test_unknown_reinsertion(self):
        # Ancestral A is deleted above node 3, and an unknown base reinserted above
        # sample 0, which must not inherit the A
        ts = isolated_sample_ts([(0, "A", [(3, "-"), (0, "N")])])
        assert alignment.alignments(ts)[0] == "NNNNNN"
        gig = alignment.from_tree_sequence(ts)
        assert_haplotypes_match(gig, ts)
        assert gig.allele_at(0, 0) == "N"

    def test_unknown_reinsertion_on_the_same_node(self):
        # The deletion and the unknown reinsertion are both above sample 0
        tables = isolated_sample_ts([]).dump_tables()
        site = tables.sites.add_row(position=0, ancestral_state="A")
        tables.mutations.add_row(site=site, node=0, derived_state="-", time=0.7)
        tables.mutations.add_row(site=site, node=0, derived_state="N", time=0.3)
        tables.sort()
        tables.build_index()
        tables.compute_mutation_parents()
        ts = tables.tree_sequence()
        assert alignment.alignments(ts, missing_data_character="?")[0] == "N?????"
        gig = alignment.from_tree_sequence(ts)
        assert_haplotypes_match(gig, ts)
        assert gig.allele_at(0, 0) == "N"

    def test_unknown_inserted_base_is_an_allele(self):
        # An unknown inserted base is the allele N, not missing data, in both the
        # alignment and the GIG, whatever the missing data character
        ts = simple_ts()
        row = alignment.alignments(ts, missing_data_character="?")[2]
        assert row == "??A??A---???GNN-????"
        gig = alignment.from_tree_sequence(ts)
        assert gig.haplotype(2, missing_data_character="?") == row.replace("-", "")

    def test_deleted_by_a_later_mutation_on_the_same_node(self):
        # AC becomes TC and then -C above the same node, so its first column is deleted
        tables = isolated_sample_ts([]).dump_tables()
        site = tables.sites.add_row(position=0, ancestral_state="AC")
        tables.mutations.add_row(site=site, node=0, derived_state="TC", time=0.8)
        tables.mutations.add_row(site=site, node=0, derived_state="-C", time=0.3)
        tables.sort()
        tables.build_index()
        tables.compute_mutation_parents()
        ts = tables.tree_sequence()
        assert alignment.alignments(ts)[0] == "-CNNNN"
        gig = alignment.from_tree_sequence(ts)
        assert gig.haplotype(0) == "CNNNN"


def round_trip(ts):
    """
    Convert ts (with alignment columns) to a GIG and back, check the result, and return
    the GIG and the new tree sequence
    """
    gig = alignment.from_tree_sequence(ts)
    new_ts = alignment.to_tree_sequence(gig)
    # Each sample's alignment, with the gaps removed, is its haplotype in the GIG
    for u, row in zip(ts.samples(), alignment.alignments(new_ts)):
        assert row.replace("-", "") == gig.haplotype(u)
    # ...and converting back gives the same haplotypes
    new_gig = alignment.from_tree_sequence(new_ts)
    for u in ts.samples():
        assert new_gig.haplotype(u, right=gig.sequence_length(u)) == gig.haplotype(u)
    return gig, new_ts


def tables_with_insertion():
    """
    A root (0) with samples 1 and 2 below it. Sample 1 inherits the root's genome, while
    sample 2 has 3 inserted bases (TTT) after its position 4
    """
    tables = gigl.Tables()
    root = tables.nodes.add_row(1)
    for _ in range(2):
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
    tables.iedges.add_row(1, root, child_right=10)
    tables.iedges.add_row(2, root, child_right=5)
    tables.iedges.add_row(2, root, child_left=8, child_right=13, parent_left=5, parent_right=10)
    for position, base in enumerate("ACGTACGTAC"):
        tables.mutations.add_row(root, position, base)
    for position in range(5, 8):
        tables.mutations.add_row(2, position, "T")
    tables.sort()
    return tables


class TestToTreeSequence:
    @pytest.mark.parametrize("seed", range(1, 11))
    def test_no_indels_round_trip_exactly(self, seed):
        # The columns are the tree sequence's positions, including where a node is the
        # root of one tree but has a parent elsewhere
        ts = msprime.sim_ancestry(6, sequence_length=300, recombination_rate=2e-3, population_size=1, random_seed=seed)
        ts = msprime.sim_mutations(ts, rate=5e-3, random_seed=seed)
        assert ts.num_trees > 1
        new_ts = alignment.to_tree_sequence(gigl.Graph.from_tree_sequence(ts))
        assert new_ts.sequence_length == ts.sequence_length
        assert alignment.alignments(new_ts) == alignment.alignments(ts)

    @pytest.mark.parametrize("seed", [*range(1, 13), 24, 161])
    def test_random_round_trip(self, seed):
        # (seed 24 has roots that are absent from some trees, and seed 161 an insertion at
        # the end of an internal node's recorded genome, which only its child reveals)
        round_trip(random_indel_ts(seed))

    def test_simple_round_trip(self):
        gig, new_ts = round_trip(simple_ts())
        rows = alignment.alignments(new_ts)
        # Sample 0 has the deletion (gaps), and the other samples gaps in the columns of
        # the insertion that only sample 1 (with known bases) and sample 2 have
        assert "---" in rows[0]
        assert rows[1].replace("-", "") == gig.haplotype(1)

    def test_insertion(self):
        new_ts = alignment.to_tree_sequence(tables_with_insertion().graph())
        assert new_ts.sequence_length == 13
        assert alignment.alignments(new_ts) == ["ACGTA---CGTAC", "ACGTATTTCGTAC"]

    def test_local_root_is_colinear(self):
        # A sample with no parent at positions 5 to 7, between iedges that leave room for
        # them, is the root of a tree there (as in a tree sequence), not an insertion
        tables = gigl.Tables()
        root = tables.nodes.add_row(1)
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(1, root, child_right=10)
        tables.iedges.add_row(2, root, child_right=5)
        tables.iedges.add_row(2, root, child_left=8, child_right=10)
        for position, base in enumerate("ACGTACGTAC"):
            tables.mutations.add_row(root, position, base)
        tables.mutations.add_row(2, 6, "G")
        tables.sort()
        new_ts = alignment.to_tree_sequence(tables.graph())
        assert new_ts.sequence_length == 10
        assert alignment.alignments(new_ts) == ["ACGTACGTAC", "ACGTANGNAC"]

    def test_missing_data_is_unknown(self):
        tables = tables_with_insertion()
        tables.mutations.clear()  # no known bases at all
        new_ts = alignment.to_tree_sequence(tables.graph())
        assert alignment.alignments(new_ts) == ["NNNNN---NNNNN", "NNNNNNNNNNNNN"]

    def test_inversion(self):
        tables = gigl.Tables()
        tables.nodes.add_row(1)
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(1, 0, child_right=10, parent_left=10, parent_right=0)
        with pytest.raises(ValueError, match="inversions are not supported"):
            alignment.to_tree_sequence(tables.graph())

    def test_multiple_chromosomes(self):
        tables = tables_with_insertion()
        tables.mutations.add_row(1, 0, "A", chromosome=1)
        tables.sort()
        with pytest.raises(ValueError, match="one chromosome"):
            alignment.to_tree_sequence(tables.graph())

    def test_duplication(self):
        tables = gigl.Tables()
        tables.nodes.add_row(1)
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(1, 0, child_right=10)
        tables.iedges.add_row(1, 0, child_left=10, child_right=15, parent_left=5, parent_right=10)
        with pytest.raises(ValueError, match="not in the same order"):
            alignment.to_tree_sequence(tables.graph())


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
