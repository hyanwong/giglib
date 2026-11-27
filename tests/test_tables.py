import collections

import msprime
import numpy as np
import pytest
import tskit

import giglib as gigl
from giglib.constants import Const, ValidFlags
from giglib.tools.homology import random_homologous_positions
from tests.gigutil import homologous_positions, random_tables, same_homology, valid_tables_without_flags


class TestCreation:
    # Test creation of a set of GiGtables
    def test_simple_from_tree_sequence(self, simple_ts):
        assert simple_ts.num_trees > 1
        tables = gigl.Tables.from_tree_sequence(simple_ts)
        assert len(tables.nodes) == simple_ts.num_nodes
        assert len(tables.iedges) == simple_ts.num_edges
        # test conversion from float to int for genomic coordinates
        assert simple_ts.edges_left.dtype == np.float64
        assert simple_ts.edges_right.dtype == np.float64
        for pos in ("parent_left", "parent_right", "child_left", "child_right"):
            assert getattr(tables.iedges, pos).dtype == np.int64
            assert isinstance(tables.iedges[0].child_left, (int, np.integer))

    @pytest.mark.parametrize("time", [0, 1])
    def test_simple_from_tree_sequence_with_timedelta(self, simple_ts, time):
        assert simple_ts.num_trees > 1
        tables = gigl.Tables.from_tree_sequence(simple_ts)
        tables.change_times(timedelta=time)
        assert tables.nodes[0].time == simple_ts.node(0).time + time

    def test_mutations(self, simple_ts_with_mutations):
        ts = simple_ts_with_mutations
        assert ts.num_mutations > 0
        tables = gigl.Tables.from_tree_sequence(ts)
        # one mutation per ancestral state, above the root(s) of each local tree
        num_ancestral = sum(tree.num_sites * len(tree.roots) for tree in ts.trees())
        assert len(tables.mutations) == ts.num_mutations + num_ancestral
        unknown = gigl.is_unknown_time(tables.mutations.time)
        assert np.sum(~unknown) == ts.num_mutations
        for mut in ts.mutations():
            matches = [
                m
                for m in tables.mutations
                if m.node == mut.node and m.position == ts.site(mut.site).position and m.time == mut.time
            ]
            assert len(matches) == 1
            assert matches[0].derived_state == mut.derived_state
        tables.graph()  # passes validation

    def test_ancestral_state_mutations(self, simple_ts_with_mutations):
        ts = simple_ts_with_mutations
        tables = gigl.Tables.from_tree_sequence(ts, chromosome=3)
        assert np.all(tables.mutations.chromosome == 3)
        for tree in ts.trees():
            for site in tree.sites():
                for root in tree.roots:
                    ids = tables.mutations.ids_for_node(root, chromosome=3)
                    at_site = [tables.mutations[i] for i in ids if tables.mutations[i].position == site.position]
                    # the ancestral state comes first, with unknown time
                    assert at_site[0].derived_state == site.ancestral_state
                    assert gigl.is_unknown_time(at_site[0].time)

    def test_isolated_nodes_have_no_ancestral_state(self):
        tables = tskit.TableCollection(sequence_length=10)
        for _ in range(3):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        root = tables.nodes.add_row(time=1)
        tables.edges.add_row(0, 10, root, 0)
        tables.edges.add_row(0, 10, root, 1)  # node 2 is isolated
        tables.sites.add_row(position=4, ancestral_state="A")
        ts = tables.tree_sequence()
        gig_tables = gigl.Tables.from_tree_sequence(ts)
        assert [m.node for m in gig_tables.mutations] == [root]

    def test_non_integer_site_positions(self):
        tables = tskit.TableCollection(sequence_length=10)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.sites.add_row(position=4.5, ancestral_state="A")
        with pytest.raises(ValueError, match="integers"):
            gigl.Tables.from_tree_sequence(tables.tree_sequence())

    def test_populations_not_implemented_error(self, ts_with_multiple_pops):
        # No need to test for migrations, since they necessarily
        # have multiple populations
        ts_tables = ts_with_multiple_pops.tables
        assert ts_tables.populations.num_rows > 1
        with pytest.raises(NotImplementedError):
            gigl.Tables.from_tree_sequence(ts_with_multiple_pops)

    def test_noninteger_positions(self):
        bad_ts = msprime.simulate(10, recombination_rate=10, random_seed=1)
        with pytest.raises(ValueError, match="integers"):
            gigl.Tables.from_tree_sequence(bad_ts)


class TestFreeze:
    @pytest.mark.parametrize(
        ("TableClass", "params"),
        [
            ("NodeTable", {"time": 0}),
            (
                "IEdgeTable",
                {
                    "child_left": 0,
                    "child_right": 1,
                    "parent_left": 0,
                    "parent_right": 1,
                    "child": 0,
                    "parent": 1,
                },
            ),
        ],
    )
    def test_freeze_table(self, TableClass, params):
        table = getattr(gigl.tables, TableClass)()
        table.add_row(**params)
        table.freeze()
        with pytest.raises((AttributeError, ValueError)):
            table.add_row(**params)
        with pytest.raises((AttributeError, ValueError)):
            table.clear()
        unfrozen = table.copy()
        unfrozen.add_row(**params)
        assert len(unfrozen) == 2
        unfrozen.clear()

    def test_freeze(self, trivial_gig):
        tables = trivial_gig.tables
        with pytest.raises(ValueError):
            tables.nodes.add_row(time=0)
        with pytest.raises(AttributeError):
            tables.tables.clear()
        with pytest.raises(AttributeError):
            tables.tables.sort()
        with pytest.raises(AttributeError):
            tables.iedges.flags = ValidFlags.NONE
        tables = tables.copy()
        # we can modify the copy
        tables.sort()
        tables.nodes.add_row(time=0)
        assert len(tables.nodes) == 1 + len(trivial_gig.nodes)
        tables.clear()
        assert len(tables.nodes) == 0


class TestCopy:
    def test_copy(self, trivial_gig):
        tables = trivial_gig.tables
        tables_copy = tables.copy()
        assert tables_copy == tables
        assert tables_copy is not tables

    def test_equals(self, trivial_gig):
        tables = trivial_gig.tables
        tables_copy = tables.copy()
        assert tables_copy == tables
        tables_copy.iedges.clear()
        assert tables_copy != tables
        tables_copy = tables.copy()
        tables_copy.time_units = "abcde"
        assert tables_copy != tables

    def test_iedges_flags_in_equality(self, trivial_gig):
        tables = trivial_gig.tables
        assert tables.iedges.flags == ValidFlags.GIG
        tables_copy = tables.copy()
        assert tables_copy.iedges == tables.iedges
        tables_copy.iedges.flags = ValidFlags.NONE
        assert tables_copy.iedges != tables.iedges


class TestExtractColumn:
    # Test extraction of columns from a gig
    def test_invalid_column_name(self, trivial_gig):
        tables = trivial_gig.tables
        with pytest.raises(AttributeError):
            _ = tables.iedges.foobar

    def test_column_from_empty_table(self, trivial_gig):
        tables = trivial_gig.tables
        assert len(tables.individuals.parents) == 0

    def test_extracted_columns(self, trivial_gig):
        tables = trivial_gig.tables
        assert np.array_equal(tables.nodes.time, [2, 1, 0, 0, 0])
        assert np.array_equal(tables.nodes.flags, [0, 0, 1, 1, 1])
        assert np.array_equal(tables.iedges.parent, [0, 0, 0, 1, 1])
        assert np.array_equal(tables.iedges.child, [1, 2, 3, 4, 4])
        assert np.array_equal(tables.iedges.child_left, [0, 0, 0, 0, 3])
        assert np.array_equal(tables.iedges.child_chromosome, [0, 0, 0, 0, 0])
        assert np.array_equal(tables.iedges.parent_chromosome, [0, 0, 0, 0, 0])
        assert np.array_equal(tables.iedges.edge, [Const.NULL] * 5)


class TestMethods:
    def test_samples(self, simple_ts):
        tables = gigl.Tables.from_tree_sequence(simple_ts)
        assert np.array_equal(tables.sample_nodes(), simple_ts.samples())

    def test_sort(self):
        tables = gigl.Tables()
        n_root = tables.nodes.add_row(2)
        n_internal = tables.nodes.add_row(1)
        n_samp0 = tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        n_samp1 = tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(n_samp0, n_root, child_left=0, child_right=5, parent_left=5, parent_right=0)
        tables.iedges.add_row(n_samp1, n_internal, child_right=5)
        tables.iedges.add_row(n_internal, n_root, child_right=5)  # Out of order
        tables.sort()
        assert tables.iedges[0].parent == 0
        assert tables.iedges[1].parent == 0
        assert tables.iedges[2].parent == 1
        assert tables.iedges[0].child == 1
        assert tables.iedges[1].child == 2
        assert tables.iedges[2].child == 3

    def test_change_times(self, trivial_gig):
        tables = trivial_gig.tables.copy()
        times = trivial_gig.tables.nodes.time
        tables.change_times(timedelta=1.5)
        assert np.isclose(tables.nodes.time, times + 1.5).all()

    def test_remove_non_ancestral_nodes(self):
        tables = gigl.Tables()
        tables.nodes.add_row(1)  # unused
        n_root = tables.nodes.add_row(2)
        tables.nodes.add_row(10)  # unused
        n_internal = tables.nodes.add_row(1)
        n_samp0 = tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        n_samp1 = tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(n_internal, n_root, child_right=5)
        tables.iedges.add_row(n_samp0, n_root, child_left=0, child_right=5, parent_left=5, parent_right=0)
        tables.iedges.add_row(n_samp1, n_internal, child_right=5)
        tables.validate()
        assert len(tables.nodes) == 6
        new_tables = tables.copy()
        mapping = new_tables.remove_non_ancestral_nodes()
        assert len(new_tables.iedges) == len(tables.iedges)
        assert len(new_tables.nodes) == 4
        gig = new_tables.graph()
        assert len(gig.nodes) == 4
        for e0, e1 in zip(tables.iedges, new_tables.iedges):
            assert e0.child == mapping[e1.child]
            assert e0.parent == mapping[e1.parent]
            assert e0.child_left == e1.child_left
            assert e0.child_right == e1.child_right
            assert e0.parent_left == e1.parent_left
            assert e0.parent_right == e1.parent_right
            assert e0.child_chromosome == e1.child_chromosome
            assert e0.parent_chromosome == e1.parent_chromosome
        for u in range(len(new_tables.nodes)):
            assert np.all(tables.iedges.ids_for_child(mapping[u]) == new_tables.iedges.ids_for_child(u))


class TestBaseTable:
    # Test various basic table methods
    def test_append_row_values(self, trivial_gig):
        tables = trivial_gig.tables.copy()
        assert len(tables.nodes) == 5
        tables.nodes.append({"time": 3, "flags": 0})
        assert len(tables.nodes) == 6
        assert tables.nodes.time[-1] == 3
        assert tables.nodes.flags[-1] == 0

    def test_iteration(self, trivial_gig):
        tables = trivial_gig.tables
        assert len(tables.nodes) > 0
        for row in tables.nodes:
            assert isinstance(row, gigl.tables.NodeTableRow)
        assert len(tables.iedges) > 0
        for row in tables.iedges:
            assert isinstance(row, gigl.tables.IEdgeTableRow)


class TestIEdgeTable:
    def test_append_integer_coords(self):
        tables = gigl.Tables()
        u = tables.nodes.add_row(flags=gigl.NODE_IS_SAMPLE, time=0)
        tables.iedges.add_row(0, u, child_left=0, child_right=1, parent_left=1, parent_right=0)
        assert tables.iedges.parent_left[0] == 1
        assert tables.iedges.child_left[0] == 0
        assert tables.iedges[0].child_span == 1
        assert tables.iedges[0].parent_span == -1

    def test_parent_coords_default_to_child_coords(self):
        tables = gigl.Tables()
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=0)
        tables.iedges.add_row(1, 0, child_left=10, child_right=20)
        assert (tables.iedges.parent_left[0], tables.iedges.parent_right[0]) == (10, 20)
        assert not tables.iedges[0].is_inversion()

    def test_parent_coords_default_with_validation(self):
        tables = gigl.Tables()
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=0)
        tables.iedges.add_row(1, 0, child_right=10)
        tables.iedges.add_row(1, 0, child_left=10, child_right=15, parent_left=5, parent_right=0)
        tables.validate()
        assert tables.iedges.has_all_bitflags(ValidFlags.GIG)
        assert list(tables.iedges.parent_left) == [0, 5]
        assert list(tables.iedges.parent_right) == [10, 0]

    def test_nodes_first_and_coords_by_keyword(self):
        iedges = gigl.Tables().iedges
        iedges.add_row(1, 0, child_right=100)
        row = iedges[0]
        assert (row.child, row.parent) == (1, 0)
        assert (row.child_left, row.child_right, row.parent_left, row.parent_right) == (0, 100, 0, 100)
        # The row fields are in the same order as the add_row arguments
        assert row._fields[:6] == ("child", "parent", "child_left", "child_right", "parent_left", "parent_right")

    def test_tskit_style_positional_coords_fail(self):
        iedges = gigl.Tables().iedges
        with pytest.raises(TypeError):
            iedges.add_row(0, 100, 1, 0)  # tskit order: left, right, parent, child
        assert len(iedges) == 0

    def test_default_coords(self):
        iedges = gigl.Tables().iedges
        iedges.add_row(1, 0)  # a single position, as in a default msprime simulation
        iedges.add_row(2, 0, child_right=10)
        assert list(iedges.child_left) == list(iedges.parent_left) == [0, 0]
        assert list(iedges.child_right) == list(iedges.parent_right) == [1, 10]

    def test_child_left_without_child_right(self):
        iedges = gigl.Tables().iedges
        with pytest.raises(ValueError, match="child_right must be given"):
            iedges.add_row(1, 0, child_left=5)
        assert len(iedges) == 0

    def test_row_fields_as_keywords(self):
        # A row can be passed back to add_row using **
        iedges = gigl.Tables().iedges
        iedges.add_row(1, 0, child_left=5, child_right=10, parent_left=20, parent_right=15, child_chromosome=1)
        copy = gigl.Tables().iedges
        copy.add_row(**iedges[0]._asdict())
        assert copy[0] == iedges[0]

    @pytest.mark.parametrize("kwargs", [{"parent_left": 10}, {"parent_right": 20}])
    def test_only_one_parent_coord(self, kwargs):
        iedges = gigl.Tables().iedges
        with pytest.raises(ValueError, match="both or neither"):
            iedges.add_row(1, 0, child_left=10, child_right=20, **kwargs)
        assert len(iedges) == 0

    def test_novalidate_add_row(self):
        tables = gigl.Tables()
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=0)
        assert tables.iedges.flags == ValidFlags.GIG
        tables.iedges.add_row(0, 1, child_right=1)
        assert tables.iedges.flags != ValidFlags.GIG

    def test_validate_after_add_row(self):
        tables = gigl.Tables()
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=0)
        assert tables.iedges.flags == ValidFlags.GIG
        tables.iedges.add_row(1, 0, child_right=1)
        assert tables.iedges.flags == 0
        tables.validate()
        assert tables.iedges.flags == ValidFlags.GIG
        # Make a valid addition but simply assume that it's valid
        tables.iedges.add_row(1, 0, child_left=1, child_right=2)
        tables.iedges.set_bitflag(ValidFlags.GIG)
        assert tables.iedges.flags == ValidFlags.GIG

    def test_child_iterator(self, all_sv_types_2re_gig):
        tables = all_sv_types_2re_gig.tables
        assert tables.iedges.flags == ValidFlags.GIG
        seen = set()
        for ie_row in tables.iedges:
            if ie_row.child not in seen:
                seen.add(ie_row.child)
                ie_rows = list(tables.iedges.ids_for_child(ie_row.child))
            assert tables.iedges[ie_rows.pop(0)] == ie_row

    def test_no_edges_child_iterator(self):
        tables = gigl.Tables()
        tables.nodes.add_row(time=0)
        ie_rows = list(tables.iedges.ids_for_child(0))
        assert len(ie_rows) == 0
        ie_rows = list(tables.iedges.ids_for_child(0, chromosome=10))
        assert len(ie_rows) == 0

    def test_bad_child_iterator(self, all_sv_types_2re_gig):
        tables = all_sv_types_2re_gig.tables.copy()
        tables.iedges.unset_bitflag(ValidFlags.IEDGES_FOR_CHILD_ADJACENT)
        with pytest.raises(ValueError, match="Cannot use this method"):
            tables.iedges.ids_for_child(10)

    def test_good_child_iterator(self, all_sv_types_2re_gig):
        flags = ValidFlags.IEDGES_FOR_CHILD_ADJACENT | ValidFlags.IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC
        tables = all_sv_types_2re_gig.tables.copy()
        # Add a valid edge and check it
        last_child = tables.iedges[-1].child
        tables.iedges.add_row(last_child, 0, child_left=1000, child_right=1001)
        tables.validate(flags)
        # Can iterate even if not guaranteed in order
        _ = tables.iedges.ids_for_child(10)

    def test_unverified_child_iterator(self, all_sv_types_2re_gig):
        tables = all_sv_types_2re_gig.tables.copy()
        expected = tables.iedges.ids_for_child(10)
        # Add a valid edge but don't check it
        last_child = tables.iedges[-1].child
        tables.iedges.add_row(last_child, 0, child_left=1000, child_right=1001)
        assert tables.iedges.num_validated == len(tables.iedges) - 1
        # The iedges of node 10 are all validated, but not those of the last child
        assert np.array_equal(tables.iedges.ids_for_child(10), expected)
        with pytest.raises(ValueError, match="have not been validated"):
            tables.iedges.ids_for_child(last_child)
        tables.validate()
        assert len(tables.iedges.ids_for_child(last_child)) > 0

    def test_clear(self, trivial_gig):
        tables = trivial_gig.tables.copy()
        tables.iedges.add_row(0, 4, child_left=10, child_right=20)
        assert tables.iedges.flags != ValidFlags.GIG
        assert len(tables.iedges) > 0
        tables.iedges.clear()
        assert tables.iedges.flags == ValidFlags.GIG
        assert len(tables.iedges) == 0
        assert len(tables.iedges.ids_for_child(0)) == 0

    def test_max_child_pos(self, trivial_gig):
        tables = trivial_gig.tables.copy()
        youngest_child = len(tables.nodes) - 1
        tables.iedges.add_row(youngest_child, 0, child_left=10, child_right=20)
        tables.validate()
        assert tables.iedges.max_pos_as_child(youngest_child, chromosome=0) == 20

    def test_bad_max_child_pos(self, trivial_gig):
        tables = trivial_gig.tables.copy()
        # Don't validate
        tables.iedges.add_row(0, 4, child_left=10, child_right=20)
        with pytest.raises(ValueError, match="have not been validated"):
            tables.iedges.max_pos_as_child(0, chromosome=0)
        tables.iedges.unset_bitflag(ValidFlags.IEDGES_FOR_CHILD_NONOVERLAPPING)
        with pytest.raises(ValueError, match="Cannot use this method"):
            tables.iedges.max_pos_as_child(1, chromosome=0)

    def test_max_child_simple(self):
        ts = tskit.Tree.generate_balanced(2, span=4).tree_sequence
        root = ts.first().root
        tables = gigl.Tables.from_tree_sequence(ts.keep_intervals([[1, 3]]))
        unconnected_node = tables.nodes.add_row(time=2)
        for leaf in ts.samples():
            assert tables.iedges.max_pos_as_child(leaf) == 3
            assert tables.iedges.max_pos_as_child(leaf, chromosome=0) == 3
            assert tables.iedges.max_pos_as_child(leaf, chromosome=1) is None
            assert tables.iedges.min_pos_as_child(leaf) == 1
            assert tables.iedges.min_pos_as_child(leaf, chromosome=0) == 1
            assert tables.iedges.min_pos_as_child(leaf, chromosome=1) is None
        assert tables.iedges.max_pos_as_child(root) is None
        assert tables.iedges.min_pos_as_child(root) is None
        assert tables.iedges.max_pos_as_child(unconnected_node) is None
        assert tables.iedges.min_pos_as_child(unconnected_node) is None


class TestBitflags:
    """
    Tests for setting and checking the validity flags on an iedge table
    """

    ADJ = ValidFlags.IEDGES_FOR_CHILD_ADJACENT
    CHR = ValidFlags.IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC
    INTS = ValidFlags.IEDGES_INTEGERS

    def test_has_all_bitflags(self):
        iedges = gigl.Tables().iedges
        iedges.flags = self.ADJ | self.INTS
        assert iedges.has_all_bitflags(self.ADJ) is True
        assert iedges.has_all_bitflags(self.ADJ | self.INTS) is True
        # Only one of the two flags is set: must be False
        assert iedges.has_all_bitflags(self.ADJ | self.CHR) is False
        assert iedges.has_all_bitflags(self.CHR) is False
        assert iedges.has_all_bitflags(ValidFlags.NONE) is True

    def test_set_unset_bitflag(self):
        iedges = gigl.Tables().iedges
        iedges.flags = ValidFlags.NONE
        iedges.set_bitflag(self.ADJ | self.CHR)
        assert iedges.has_all_bitflags(self.ADJ | self.CHR)
        iedges.unset_bitflag(self.CHR | self.INTS)
        assert iedges.flags == self.ADJ

    def test_ids_for_child_requires_all_flags(self):
        tables = gigl.Tables()
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=0)
        tables.iedges.add_row(1, 0, child_right=1)
        tables.iedges.flags = self.ADJ  # not guaranteed sorted by chromosome
        with pytest.raises(ValueError, match="ordered by chromosome"):
            tables.iedges.ids_for_child(1)
        tables.iedges.set_bitflag(self.CHR)
        assert list(tables.iedges.ids_for_child(1)) == [0]

    def test_max_min_pos_require_all_flags(self):
        tables = gigl.Tables()
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=0)
        tables.iedges.add_row(1, 0, child_right=1)
        tables.iedges.flags = ValidFlags.IEDGES_WITHIN_CHILD_SORTED  # overlap not checked
        for method in (tables.iedges.max_pos_as_child, tables.iedges.min_pos_as_child):
            with pytest.raises(ValueError, match="nonoverlapping"):
                method(1)

    def test_from_tree_sequence_flags(self, simple_ts):
        tables = gigl.Tables.from_tree_sequence(simple_ts)
        assert tables.iedges.has_all_bitflags(ValidFlags.IEDGES_ALL)


class TestReadOnlyColumns:
    """
    Columns are read-only, so that editing them cannot leave indexes or validity flags
    out of date
    """

    IEDGE_COLUMNS = (
        "child_left",
        "child_right",
        "parent_left",
        "parent_right",
        "child",
        "parent",
        "child_chromosome",
        "parent_chromosome",
        "edge",
    )

    @staticmethod
    def make_tables():
        tables = gigl.Tables()
        tables.nodes.add_row(1, individual=tables.individuals.add_row())
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(1, 0, child_right=5)
        tables.sort()
        return tables

    @pytest.mark.parametrize(
        ("table", "column"),
        [("iedges", c) for c in IEDGE_COLUMNS]
        + [("nodes", "time"), ("nodes", "flags"), ("nodes", "individual"), ("individuals", "flags")],
    )
    def test_cannot_edit(self, table, column):
        tables = self.make_tables()
        values = getattr(getattr(tables, table), column)
        before = values.copy()
        with pytest.raises(ValueError, match="read-only"):
            values[0] = 1
        assert np.array_equal(getattr(getattr(tables, table), column), before)

    def test_stale_index_prevented(self):
        tables = self.make_tables()
        with pytest.raises(ValueError, match="read-only"):
            tables.iedges.child[0] = 0
        assert list(tables.iedges.ids_for_child(1)) == [0]

    def test_columns_follow_storage_changes(self):
        # Read-only column views are created when the storage array is replaced, so
        # check they stay correct when that happens
        tables = gigl.Tables(initial_sizes={"nodes": 2, "iedges": 2})
        for t in range(10):  # grows past the initial size, so storage is replaced
            tables.nodes.add_row(t)
        assert list(tables.nodes.time) == list(range(10))
        for c in range(9):  # youngest child first, so not in sorted order
            tables.iedges.add_row(c, c + 1, child_right=5)
        assert list(tables.iedges.child) == list(range(9))
        tables.sort(shrink_memory=True)  # replaces the iedges storage array
        assert list(tables.iedges.child) == list(range(8, -1, -1))  # oldest child first
        copy = tables.copy()
        assert np.array_equal(copy.nodes.time, tables.nodes.time)
        copy.nodes.add_row(99)
        assert copy.nodes.time[-1] == 99
        assert len(tables.nodes.time) == 10
        copy.clear()
        assert len(copy.nodes.time) == 0
        copy.nodes.add_row(5)
        assert list(copy.nodes.time) == [5]
        gig = tables.graph()  # freezes the tables
        assert list(gig.tables.nodes.time) == list(range(10))
        with pytest.raises(ValueError, match="read-only"):
            gig.tables.nodes.time[0] = 1

    def test_internal_edits_still_possible(self):
        tables = self.make_tables()
        tables.nodes.add_row(0.5)  # not referenced by any iedge, so will be removed
        tables.change_times(1)
        assert list(tables.nodes.time) == [2, 1, 1.5]
        tables.iedges.set_bitflag(ValidFlags.IEDGES_PARENT_OLDER_THAN_CHILD)
        mapping = tables.remove_non_ancestral_nodes()
        assert list(mapping) == [0, 1]
        # columns remain read-only after internal edits
        with pytest.raises(ValueError, match="read-only"):
            tables.nodes.time[0] = 0


class TestIedgesAddRow:
    """
    Adding iedges makes no validity checks, apart from refusing non-integer values
    (validation is tested in TestValidate)
    """

    @pytest.mark.parametrize(
        "field", ["child", "parent", "child_left", "child_right", "parent_left", "parent_right", "edge"]
    )
    def test_refuses_non_integers(self, field):
        values = {"child": 1, "parent": 0, "child_left": 0, "child_right": 2, "parent_left": 0, "parent_right": 2}
        values[field] = 1.5
        tables = gigl.Tables()
        with pytest.raises(ValueError, match="must be integers"):
            tables.iedges.add_row(**values)
        assert len(tables.iedges) == 0

    def test_accepts_integer_valued_numbers(self):
        tables = gigl.Tables()
        tables.iedges.add_row(np.int64(1), 0, child_left=0.0, child_right=np.float64(2))
        assert tables.iedges[0] == (1, 0, 0, 2, 0, 2, 0, 0, gigl.NULL)

    def test_no_other_checks(self):
        tables = gigl.Tables()
        tables.iedges.add_row(5, 7, child_left=-3, child_right=-4, parent_left=0, parent_right=10)
        assert len(tables.iedges) == 1

    def test_adding_removes_flags_until_validated(self):
        tables = gigl.Tables()
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=0)
        assert tables.iedges.flags == ValidFlags.GIG  # an empty table is valid
        tables.iedges.add_row(1, 0, child_right=1)
        assert tables.iedges.flags == 0
        tables.validate(ValidFlags.IEDGES_INTERVALS)
        assert tables.iedges.flags == ValidFlags.IEDGES_INTERVALS


class TestFromTreeSequenceFlags:
    def test_flags(self):
        ts = msprime.sim_ancestry(4, sequence_length=50, recombination_rate=0.01, random_seed=1)
        tables = gigl.Tables.from_tree_sequence(ts)
        assert tables.iedges.has_all_bitflags(ValidFlags.IEDGES_FROM_TSKIT)
        # Integer coordinates are checked, and sorting gives the ordering flags
        assert tables.iedges.flags == ValidFlags.GIG


class TestValidate:
    """
    Tables.validate() checks requirements that are not already guaranteed, and sets
    their flags
    """

    valid_tables = staticmethod(valid_tables_without_flags)

    def test_sets_flags(self):
        tables = self.valid_tables()
        assert tables.iedges.flags == 0
        tables.validate()
        assert tables.iedges.flags == ValidFlags.GIG
        tables.graph()

    def test_subset(self):
        tables = self.valid_tables()
        tables.validate(ValidFlags.IEDGES_INTERVALS | ValidFlags.IEDGES_FOR_CHILD_ADJACENT)
        assert tables.iedges.flags == ValidFlags.IEDGES_INTERVALS | ValidFlags.IEDGES_FOR_CHILD_ADJACENT

    def test_empty(self):
        tables = gigl.Tables()
        tables.iedges.unset_bitflag(ValidFlags.IEDGES_ALL)
        tables.validate()
        assert tables.iedges.flags == ValidFlags.GIG

    def test_from_tree_sequence_rechecks(self):
        ts = msprime.sim_ancestry(5, sequence_length=100, recombination_rate=0.01, random_seed=1)
        tables = gigl.Tables.from_tree_sequence(ts)
        tables.validate(recheck=True)

    def test_trusts_set_flags_unless_rechecked(self):
        tables = self.valid_tables()
        tables.iedges.add_row(1, 0, child_left=10, child_right=11)  # child 1 is no longer adjacent
        tables.iedges.set_bitflag(ValidFlags.IEDGES_FOR_CHILD_ADJACENT)  # a false claim
        tables.validate(ValidFlags.IEDGES_FOR_CHILD_ADJACENT)  # trusted, so not checked
        with pytest.raises(ValueError, match="not adjacent"):
            tables.validate(ValidFlags.IEDGES_FOR_CHILD_ADJACENT, recheck=True)

    def test_node_dependent_flags_rechecked_if_nodes_changed(self):
        # Guarantees that depend on node times are rechecked if the node table has been
        # changed (other than by adding rows), e.g. by being cleared and refilled
        tables = gigl.Tables()
        tables.nodes.add_row(1)
        tables.nodes.add_row(0)
        tables.iedges.add_row(1, 0, child_right=1)
        tables.validate()
        assert tables.iedges.flags == ValidFlags.GIG
        tables.nodes.clear()
        tables.nodes.add_row(0)
        tables.nodes.add_row(1)  # now the parent is younger than the child
        with pytest.raises(ValueError, match="not older than child"):
            tables.validate()
        with pytest.raises(ValueError, match="not older than child"):
            tables.graph()

    def test_algorithms_recheck_node_dependent_flags(self):
        # Algorithms that rely on the flags also recheck those that depend on node times
        tables = gigl.Tables()
        for t in (2, 1, 0, 0):
            tables.nodes.add_row(t)
        tables.iedges.add_row(2, 0, child_right=1)
        tables.iedges.add_row(3, 0, child_right=1)
        tables.validate()
        tables.change_times(1)  # still valid
        tables.find_homology(2, 3)
        tables.nodes.clear()
        for t in (0, 1, 0, 0):
            tables.nodes.add_row(t)  # now the parent is no older than the children
        with pytest.raises(ValueError, match="not older than child"):
            tables.find_homology(2, 3)

    def test_self_edge_with_unknown_time(self):
        # NaN times are not valid node times, so can't satisfy the parent-age check
        tables = gigl.Tables()
        tables.nodes.add_row(gigl.UNKNOWN_TIME)
        tables.iedges.add_row(0, 0, child_right=1)
        with pytest.raises(ValueError, match="not older than child"):
            tables.validate(ValidFlags.IEDGES_PARENT_OLDER_THAN_CHILD)

    def test_negative_chromosomes_allowed(self):
        tables = self.valid_tables()
        tables.iedges.add_row(3, 1, child_left=0, child_right=5, child_chromosome=-1, parent_chromosome=-1)
        tables.validate(ValidFlags.IEDGES_INTEGERS)

    @pytest.mark.parametrize("method", ["find_homology", "sample_resolve"])
    def test_tracing_requires_all_its_flags(self, method):
        tables = self.valid_tables()
        tables.validate(ValidFlags.IEDGES_PARENT_OLDER_THAN_CHILD)  # but not e.g. adjacency
        args = (2, 3) if method == "find_homology" else ()
        with pytest.raises(ValueError, match="FOR_CHILD_ADJACENT"):
            getattr(tables, method)(*args)
        tables.validate(ValidFlags.IEDGES_FOR_TRACING)
        getattr(tables, method)(*args)

    @pytest.mark.parametrize(
        ("ends", "message"),
        [
            # A reversed child interval (a reversed parent interval is an inversion)
            (dict(child_left=10, child_right=0, parent_left=0, parent_right=10), "child_left >= child_right"),
            (dict(child_left=-10, child_right=-5, parent_left=0, parent_right=5), "negative child_left"),
        ],
    )
    def test_tracing_needs_valid_child_intervals(self, ends, message):
        # Tracing would silently skip these intervals, so they fail its checks instead
        tables = gigl.Tables()
        tables.nodes.add_row(1)
        tables.nodes.add_row(0)
        tables.nodes.add_row(0)
        for child in (1, 2):
            tables.iedges.add_row(child, 0, **ends)
        with pytest.raises(ValueError, match=message):
            tables.validate(ValidFlags.IEDGES_FOR_TRACING)

    def test_frozen(self):
        gig = self.valid_tables().graph()
        gig.tables.validate()  # all flags set, so nothing to do
        gig.tables.validate(recheck=True)  # everything rechecked, and no flags need setting

    @pytest.mark.parametrize(
        ("flag", "make_invalid", "match"),
        [
            (ValidFlags.IEDGES_INTEGERS, lambda t: t.iedges.add_row(2, 0, child_left=-1, child_right=0), "negative"),
            (
                ValidFlags.IEDGES_PARENT_OLDER_THAN_CHILD,
                lambda t: t.iedges.add_row(2, 9, child_left=10, child_right=11),
                "num nodes",
            ),
            (
                ValidFlags.IEDGES_INTERVALS,
                lambda t: t.iedges.add_row(2, 0, child_left=10, child_right=11, parent_left=0, parent_right=2),
                "spans",
            ),
            (
                ValidFlags.IEDGES_CHILD_INTERVAL_POSITIVE,
                lambda t: t.iedges.add_row(2, 0, child_left=12, child_right=11, parent_left=0, parent_right=1),
                "child_left >= child_right",
            ),
            (
                ValidFlags.IEDGES_PARENT_OLDER_THAN_CHILD,
                lambda t: t.iedges.add_row(0, 2, child_left=10, child_right=11),
                "not older than child",
            ),
            (
                ValidFlags.IEDGES_SAME_PARENT_CHILD_FOR_EDGE,
                lambda t: t.iedges.add_row(3, 0, child_left=10, child_right=11, edge=1),
                "same edge ID",
            ),
            (
                ValidFlags.IEDGES_FOR_CHILD_ADJACENT,
                lambda t: t.iedges.add_row(1, 0, child_left=10, child_right=11),
                "not adjacent",
            ),
            (
                ValidFlags.IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC,
                lambda t: t.iedges.add_row(3, 1, child_right=1),
                "sorted by chromosome",
            ),
            (
                ValidFlags.IEDGES_FOR_CHILD_SECONDARY_ORDER_LEFT_ASC,
                lambda t: t.iedges.add_row(3, 1, child_right=1, child_chromosome=1, parent_chromosome=0),
                "sorted by left pos",
            ),
            (
                ValidFlags.IEDGES_FOR_CHILD_NONOVERLAPPING,
                lambda t: t.iedges.add_row(
                    3, 0, child_left=5, child_right=11, child_chromosome=1, parent_left=0, parent_right=6
                ),
                "multiple or duplicate parents",
            ),
            (
                ValidFlags.IEDGES_PRIMARY_ORDER_CHILD_TIME_DESC,
                lambda t: t.iedges.add_row(1, 0, child_left=10, child_right=11),
                "not sorted by child time, descending",
            ),
            (
                ValidFlags.IEDGES_SECONDARY_ORDER_CHILD_ID_ASC,
                lambda t: t.iedges.add_row(2, 0, child_left=10, child_right=11),
                "then ID",
            ),
        ],
    )
    @pytest.mark.parametrize("incremental", [False, True])
    def test_each_requirement(self, flag, make_invalid, match, incremental):
        tables = self.valid_tables()
        if incremental:
            tables.validate()  # so only the invalid row will be checked
        make_invalid(tables)
        with pytest.raises(ValueError, match=match):
            tables.validate(flag)
        assert not tables.iedges.has_all_bitflags(flag)

    def test_graph_checks_edge_ids(self):
        # Previously, creating a Graph set this flag without checking it
        tables = self.valid_tables()
        tables.iedges.add_row(3, 0, child_left=10, child_right=11, edge=1)
        with pytest.raises(ValueError, match="same edge ID"):
            tables.graph()


class TestSquash:
    @staticmethod
    def split_iedges(tables, rng):
        # Split each iedge into pieces, giving iedges that can be squashed back together
        rows = list(tables.iedges)
        tables.iedges.clear()
        for row in rows:
            # Cut at up to two random points inside the interval (an iedge covering a
            # single position can't be cut)
            cuts = []
            if row.child_right - row.child_left > 1:
                cuts = sorted(set(rng.integers(row.child_left + 1, row.child_right, size=2).tolist()))
            for left, right in zip([row.child_left, *cuts], [*cuts, row.child_right]):
                offset = left - row.child_left
                if row.is_inversion():
                    parent_left, parent_right = row.parent_left - offset, row.parent_left - offset - (right - left)
                else:
                    parent_left, parent_right = row.parent_left + offset, row.parent_left + offset + (right - left)
                tables.iedges.add_row(
                    row.child,
                    row.parent,
                    child_left=left,
                    child_right=right,
                    parent_left=parent_left,
                    parent_right=parent_right,
                    child_chromosome=row.child_chromosome,
                    parent_chromosome=row.parent_chromosome,
                    edge=row.edge,
                )
        tables.sort()

    def test_squash_tree_sequence(self):
        ts = msprime.sim_ancestry(5, sequence_length=100, recombination_rate=0.01, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        original = gigl.Tables.from_tree_sequence(ts)
        tables = original.copy()
        self.split_iedges(tables, np.random.default_rng(1))
        assert len(tables.iedges) > len(original.iedges)
        flags = tables.iedges.flags
        tables.iedges.squash()
        assert list(tables.iedges) == list(original.iedges)
        assert tables.iedges.flags == flags

    def test_squash_inversions(self, all_sv_types_2re_gig):
        original = all_sv_types_2re_gig.tables.copy()
        assert any(ie.is_inversion() for ie in original.iedges)
        tables = original.copy()
        self.split_iedges(tables, np.random.default_rng(2))
        assert len(tables.iedges) > len(original.iedges)
        tables.iedges.squash()
        assert list(tables.iedges) == list(original.iedges)

    def test_single_position_iedges(self):
        # e.g. from a single-site simulation: these can't be split, and are unchanged
        tables = gigl.Tables()
        for t in (2, 1, 0):
            tables.nodes.add_row(t)
        tables.iedges.add_row(1, 0)  # covers position 0 only
        tables.iedges.add_row(2, 1)
        tables.sort()
        original = tables.copy()
        self.split_iedges(tables, np.random.default_rng(3))
        tables.iedges.squash()
        assert list(tables.iedges) == list(original.iedges)

    def test_not_merged(self):
        tables = gigl.Tables()
        for t in (2, 1, 0):
            tables.nodes.add_row(t)
        tables.iedges.add_row(2, 1, child_right=5)
        tables.iedges.add_row(2, 1, child_left=5, child_right=8, parent_left=8, parent_right=5)  # inversion
        tables.iedges.add_row(2, 0, child_left=8, child_right=9, parent_left=8, parent_right=9)  # other parent
        tables.iedges.add_row(2, 0, child_left=9, child_right=10, parent_left=20, parent_right=21)  # parent gap
        tables.iedges.add_row(2, 0, child_left=10, child_right=11, parent_left=21, parent_right=22, edge=7)
        tables.iedges.add_row(2, 0, child_right=5, child_chromosome=1, parent_chromosome=1)
        tables.sort()
        before = tables.iedges.copy()
        tables.iedges.squash()
        assert tables.iedges == before

    def test_requires_sorted(self):
        tables = gigl.Tables()
        tables.nodes.add_row(1)
        tables.nodes.add_row(0)
        tables.iedges.add_row(1, 0, child_left=5, child_right=8)
        tables.iedges.add_row(1, 0, child_right=5)
        with pytest.raises(ValueError, match="Squashing requires"):
            tables.iedges.squash()

    def test_frozen(self, all_sv_types_2re_gig):
        with pytest.raises((ValueError, AttributeError)):
            all_sv_types_2re_gig.tables.iedges.squash()


class TestZeroInitialSize:
    # Tables preallocated with no rows (e.g. converted from a tree sequence with no
    # edges) must still be able to grow
    @pytest.mark.parametrize("name", ["nodes", "iedges", "individuals", "mutations"])
    def test_add_rows(self, name):
        tables = gigl.Tables(initial_sizes={name: 0})
        add_row = {
            "nodes": lambda: tables.nodes.add_row(0),
            "iedges": lambda: tables.iedges.add_row(1, 0),
            "individuals": lambda: tables.individuals.add_row(),
            "mutations": lambda: tables.mutations.add_row(0, 0, "A"),
        }[name]
        for _ in range(3):
            add_row()
        assert len(getattr(tables, name)) == 3

    def test_from_tree_sequence_without_edges(self):
        ts = tskit.TableCollection(sequence_length=10)
        ts.nodes.add_row(time=1)
        ts.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
        tables = gigl.Tables.from_tree_sequence(ts.tree_sequence())
        assert len(tables.iedges) == 0
        tables.iedges.add_row(1, 0, child_right=10)
        assert len(tables.iedges) == 1


class TestIndividualTable:
    def test_sequences_stored_as_tuples(self):
        individuals = gigl.Tables().individuals
        individuals.add_row(location=np.array([1.5, 2]), parents=np.array([0, 1], dtype=np.int32))
        individuals.add_row()
        row = individuals[0]
        assert row.location == (1.5, 2.0)
        assert row.parents == (0, 1)
        assert all(isinstance(x, float) for x in row.location)
        assert all(isinstance(x, int) for x in row.parents)
        assert individuals[1].location is None
        assert individuals[1].parents is None

    def test_separately_converted_tables_equal(self):
        # Individuals from tskit have (empty) numpy arrays for location and parents,
        # which previously made comparing tables raise an error
        ts = msprime.sim_ancestry(2, sequence_length=10, random_seed=1)
        assert ts.num_individuals > 0
        assert gigl.Tables.from_tree_sequence(ts) == gigl.Tables.from_tree_sequence(ts)


class TestFlagsKeptByTableAlgorithms:
    """
    Table algorithms that build new iedges from valid ones keep the guarantees that their
    output satisfies by construction (see https://github.com/hyanwong/giglib/issues/181)
    """

    @staticmethod
    def ts_tables():
        ts = msprime.sim_ancestry(5, sequence_length=100, recombination_rate=0.01, random_seed=1)
        return gigl.Tables.from_tree_sequence(msprime.sim_mutations(ts, rate=0.01, random_seed=1))

    @staticmethod
    def run(name, tables):
        if name == "sample_resolve":
            tables.sample_resolve()
        else:
            # (a late enough cutoff to keep some iedges: an empty table satisfies everything)
            tables.decapitate(np.quantile(tables.nodes.time, 0.9))

    @pytest.mark.parametrize("name", ["sample_resolve", "decapitate"])
    def test_valid_input_keeps_all_flags(self, name, all_sv_types_2re_gig):
        for tables in (self.ts_tables(), all_sv_types_2re_gig.tables.copy()):
            assert tables.iedges.flags == ValidFlags.GIG
            self.run(name, tables)
            assert len(tables.iedges) > 0
            assert tables.iedges.flags == ValidFlags.GIG
            tables.validate(recheck=True)  # the flags are correct

    def test_find_homology_after_sample_resolve(self):
        # e.g. in a simulation that simplifies regularly, without validating again
        tables = self.ts_tables()
        tables.sample_resolve()
        u, v = tables.sample_nodes()[:2]
        assert len(tables.find_homology(u, v)) > 0

    @pytest.mark.parametrize("name", ["sample_resolve", "decapitate"])
    def test_flags_not_added(self, name):
        tables = self.ts_tables()
        tables.iedges.unset_bitflag(ValidFlags.IEDGES_SAME_PARENT_CHILD_FOR_EDGE)
        self.run(name, tables)
        assert len(tables.iedges) > 0
        # Not claimed, as the input didn't have it (the others are still kept)
        assert not tables.iedges.has_all_bitflags(ValidFlags.IEDGES_SAME_PARENT_CHILD_FOR_EDGE)
        assert tables.iedges.has_all_bitflags(ValidFlags.IEDGES_FOR_TRACING)

    @pytest.mark.parametrize("name", ["sample_resolve", "decapitate"])
    def test_stale_node_time_flags_rechecked(self, name):
        # If the node table changes after validation, the flag saying that parents are
        # older than children may be stale, so it is rechecked rather than kept
        tables = self.ts_tables()
        times = tables.nodes.time.copy()
        flags = tables.nodes.flags.copy()
        tables.nodes.clear()
        for t, f in zip(times.max() - times, flags):  # reverse the order of node times
            tables.nodes.add_row(t, flags=f)
        assert tables.iedges.has_all_bitflags(ValidFlags.IEDGES_PARENT_OLDER_THAN_CHILD)  # stale
        with pytest.raises(ValueError, match="not older than child"):
            self.run(name, tables)

    def test_sample_resolve_without_sorting(self):
        tables = self.ts_tables()
        tables.sample_resolve(sort=False)
        assert tables.iedges.has_all_bitflags(ValidFlags.IEDGES_ORDER_INDEPENDENT)
        tables.validate(ValidFlags.IEDGES_ORDER_INDEPENDENT, recheck=True)
        # Without the ordering flags, tracing needs the tables to be sorted first
        u, v = tables.sample_nodes()[:2]
        with pytest.raises(ValueError, match="FOR_CHILD_ADJACENT"):
            tables.find_homology(u, v)
        tables.sort()
        tables.find_homology(u, v)


class TestNodeTable:
    def test_times(self):
        nodes = gigl.tables.NodeTable()
        nodes.add_row(time=1)
        nodes.add_row(time=2)
        nodes.add_row(time=4)
        assert len(nodes.time.shape) == 1
        assert nodes.time.shape[0] == 3
        assert nodes.time.dtype == np.float64
        assert np.all(nodes.time == [1.0, 2.0, 4.0])
        new_nodes = nodes.copy()
        nodes.clear()
        assert len(nodes.time.shape) == 1
        assert nodes.time.shape[0] == 0
        assert nodes.time.dtype == np.float64
        assert len(new_nodes.time.shape) == 1
        assert new_nodes.time.shape[0] == 3
        assert new_nodes.time.dtype == np.float64
        assert np.all(new_nodes.time == [1.0, 2.0, 4.0])

    @pytest.mark.skip("add_rows() not yet reimplemented: see docs/design/storage.md")
    def test_add_rows(self):
        # Adding many rows in one call, broadcasting scalar values
        nodes = gigl.tables.NodeTable()
        ids = nodes.add_rows(time=[1, 2, 3], flags=0)
        assert np.all(ids == [0, 1, 2])
        assert np.all(nodes.time == [1, 2, 3])
        assert np.all(nodes.flags == 0)


class TestStringRepresentations:
    # Test string and html representations of tables
    def test_hardcoded_tskit_is_skipped(self, gig_from_degree2_ts):
        output = str(gig_from_degree2_ts)
        assert "tskit" not in output

    def test_identifiable_values_from_str(self):
        tables = gigl.Tables()
        nodes = [(0, 3.1451), (1, 7.4234)]
        iedge = [222, 777, 322, 877, 1, 0]
        for node in nodes:
            tables.nodes.add_row(time=node[1], flags=node[0])
        tables.iedges.add_row(
            iedge[4], iedge[5], child_left=iedge[0], child_right=iedge[1], parent_left=iedge[2], parent_right=iedge[3]
        )
        for node in nodes:
            assert str(node[0]) in str(tables)
            assert str(node[1]) in str(tables)
        for value in iedge:
            assert str(value) in str(tables)

    @pytest.mark.parametrize("num_rows", [0, 10, 40, 50])
    def test_repr_html(self, num_rows):
        # All the rows are shown, up to the maximum number of lines (40 by default),
        # beyond which some are skipped (with a note saying so)
        nodes = gigl.Tables().nodes
        for _ in range(num_rows):
            nodes.append({"time": 0, "flags": 0})
        html = nodes._repr_html_()
        if num_rows > 40:
            assert "10 rows skipped (giglib.set_print_options)" in html
        else:
            assert "skipped" not in html
            assert html.count("<tr") == num_rows + 1  # (including the header)


class TestIEdgeAttributes:
    def test_non_ts_attributes(self, simple_ts):
        tables = gigl.Tables.from_tree_sequence(simple_ts)
        assert tables.iedges.edge.dtype == np.int64
        assert np.all(tables.iedges.edge == gigl.NULL)

    @pytest.mark.parametrize(
        "name",
        ["parent", "child", "parent_left", "parent_right", "child_left", "child_right"],
    )
    def test_ts_attributes(self, simple_ts, name):
        gig = gigl.Graph.from_tree_sequence(simple_ts)
        tables = gig.tables
        assert getattr(tables.iedges, name).dtype == np.int64
        suffix = name.split("_")[-1]
        # Iedges are not in the same order as tskit edges, so compare sorted values
        assert np.array_equal(np.sort(getattr(tables.iedges, name)), np.sort(getattr(simple_ts, "edges_" + suffix)))


class TestIndividualAttributes:
    def test_asdict(self, simple_ts):
        tables = gigl.Tables.from_tree_sequence(simple_ts)
        for i, ind in enumerate(tables.individuals):
            assert np.array_equal(ind.parents, simple_ts.individual(i).parents)


class TestNodeAttributes:
    def test_flags(self, simple_ts):
        tables = gigl.Tables.from_tree_sequence(simple_ts)
        assert np.all(tables.nodes.flags == simple_ts.nodes_flags)

    def test_edges_match_tskit(self, simple_ts):
        gig = gigl.Graph.from_tree_sequence(simple_ts)
        iedges = {(ie.parent, ie.child, ie.child_left, ie.child_right) for ie in gig.iedges}
        edges = {(e.parent, e.child, e.left, e.right) for e in simple_ts.edges()}
        assert iedges == edges


class TestFindHomology:
    def test_single_tree(self):
        span = 123
        comb_ts = tskit.Tree.generate_comb(5, span=span).tree_sequence
        # make a gig with many unary nodes above node 0
        comb_ts = comb_ts.simplify([0, 4], keep_unary=True)
        gig = gigl.Graph.from_tree_sequence(comb_ts)
        whole = (0, span, 0)
        assert same_homology(gig.tables.find_homology(0, 1), {(whole, whole, comb_ts.first().root, whole)})

    def test_2_trees(self, degree2_2_tip_ts):
        assert degree2_2_tip_ts.num_trees == 2
        gig = gigl.Graph.from_tree_sequence(degree2_2_tip_ts)
        homology = gig.tables.find_homology(0, 1)
        expected = set()
        for tree in degree2_2_tip_ts.trees():
            interval = (int(tree.interval.left), int(tree.interval.right), 0)
            expected.add((interval, interval, tree.root, interval))
        assert same_homology(homology, expected)
        # sorted by position in u
        assert [h.u for h in homology] == sorted(h.u for h in homology)

    def test_time_cutoff(self, degree2_2_tip_ts):
        # set a cutoff so that the mrca in one tree is never visited
        assert degree2_2_tip_ts.num_trees == 2
        assert degree2_2_tip_ts.num_samples == 2
        T = degree2_2_tip_ts.nodes_time
        tree0 = degree2_2_tip_ts.first()
        tree1 = degree2_2_tip_ts.last()
        assert T[tree0.root] != T[tree1.root]
        gig = gigl.Graph.from_tree_sequence(degree2_2_tip_ts)
        cutoff = (T[[tree0.root, tree1.root]]).mean()
        homology = gig.tables.find_homology(0, 1, time_cutoff=cutoff)
        used_tree = tree0 if T[tree0.root] < T[tree1.root] else tree1
        assert [h.mrca for h in homology] == [used_tree.root]

    def test_simple_non_sv(self, simple_ts):
        assert simple_ts.num_trees > 1
        assert simple_ts.num_samples >= 2
        max_trees = 0
        gig = gigl.Graph.from_tree_sequence(simple_ts)
        for u in range(len(gig.tables.sample_nodes())):
            for v in range(u + 2, len(gig.tables.sample_nodes())):
                homology = homologous_positions(gig.tables.find_homology(u, v))
                equiv_ts = simple_ts.simplify([u, v], filter_nodes=False)
                max_trees = max(max_trees, equiv_ts.num_trees)
                for tree in equiv_ts.trees():
                    interval = (int(tree.interval.left), int(tree.interval.right), 0)
                    # No duplications, so each region has one homologue, at the same position
                    assert homologous_positions([(interval, interval, tree.get_mrca(u, v), interval)]) <= homology
        assert max_trees > 2  # at least some cases with 3 or more mrcas

    def test_double_inversion(self, double_inversion_gig):
        assert len(double_inversion_gig.sample_nodes()) == 2
        iedges = list(double_inversion_gig.iedges_for_child(0))
        assert len(iedges) == 1
        assert iedges[0].is_inversion()
        whole = (0, 100, 0)
        assert same_homology(double_inversion_gig.tables.find_homology(0, 1), {(whole, whole, 3, whole)})

    @pytest.mark.parametrize("sample_resolve", [True, False])
    def test_extended_inversion(self, extended_inversion_gig, sample_resolve):
        gig = extended_inversion_gig
        if sample_resolve:
            gig = gig.sample_resolve()
        assert set(gig.tables.sample_nodes()) == {0, 1}
        assert gig.min_pos_as_child(0) == 20
        assert gig.max_pos(0) == 155
        assert gig.min_pos_as_child(1) == 0
        assert gig.max_pos(1) == 100
        # The inversion only leaves 15..100 of the MRCA shared, which is 155 leftwards
        # (to 70) in u: so u's segment 70..155 runs backwards in v and in the MRCA
        homology = gig.tables.find_homology(0, 1)
        assert same_homology(homology, [((70, 155, 0), (100, 15, 0), 3, (100, 15, 0))])
        assert all(h.is_reversed for h in homology)
        assert sum(h.length for h in homology) == 85

    def test_inverted_duplicate(self, inverted_duplicate_gig):
        assert len(inverted_duplicate_gig.sample_nodes()) == 2
        assert same_homology(
            inverted_duplicate_gig.tables.find_homology(0, 1),
            {
                # The duplicated and inverted section of the MRCA (10..15) has two copies in u
                ((0, 5, 0), (0, 5, 0), 3, (10, 15, 0)),
                ((10, 15, 0), (5, 0, 0), 3, (15, 10, 0)),
                # Only the inverted section
                ((5, 10, 0), (10, 5, 0), 3, (20, 15, 0)),
            },
        )

    def test_inverted_duplicate_with_missing(self, inverted_duplicate_with_missing_gig):
        assert len(inverted_duplicate_with_missing_gig.sample_nodes()) == 2
        assert same_homology(
            inverted_duplicate_with_missing_gig.tables.find_homology(0, 1),
            {
                ((0, 5, 0), (0, 5, 0), 3, (10, 15, 0)),
                ((30, 35, 0), (5, 0, 0), 3, (15, 10, 0)),
                ((25, 30, 0), (10, 5, 0), 3, (20, 15, 0)),
            },
        )

    def test_unequal_crossover_first_mrca(self, unequal_crossover_gig):
        # X (node 4) = a B B B c and Y (node 5) = a B c. The lineages of X's first copy of B
        # and Y's copy meet at A (node 1), as do X's third copy and the end of Y, and stop
        assert unequal_crossover_gig.sequence_length(4) == 50
        assert unequal_crossover_gig.sequence_length(5) == 30
        assert same_homology(
            unequal_crossover_gig.tables.find_homology(4, 5),
            {
                ((0, 15, 0), (0, 15, 0), 1, (0, 15, 0)),
                ((35, 50, 0), (15, 30, 0), 1, (25, 40, 0)),
            },
        )

    @pytest.mark.parametrize("mode", ["closest_homologues", "all"])
    def test_unequal_crossover(self, unequal_crossover_gig, mode):
        # Each copy of B in X is homologous to Y's one copy: at A for the positions that
        # both inherit from the same position of A, and otherwise at R (node 0), through
        # the duplication. Each position of X has only one homologue in Y, so the closest
        # homologues are all the homology
        assert same_homology(
            unequal_crossover_gig.tables.find_homology(4, 5, mode=mode),
            {
                ((0, 15, 0), (0, 15, 0), 1, (0, 15, 0)),  # a, and the start of B
                ((15, 20, 0), (15, 20, 0), 0, (15, 20, 0)),  # the end of the first copy
                ((20, 30, 0), (10, 20, 0), 0, (10, 20, 0)),  # the second copy
                ((30, 35, 0), (10, 15, 0), 0, (10, 15, 0)),  # the start of the third copy
                ((35, 50, 0), (15, 30, 0), 1, (25, 40, 0)),  # the end of B, and c
            },
        )

    def test_no_recomb_sv_dup_del(self, all_sv_types_no_re_gig):
        assert same_homology(
            all_sv_types_no_re_gig.tables.find_homology(8, 10),
            {
                # (0, 50) is unchanged
                ((0, 50, 0), (0, 50, 0), 1, (0, 50, 0)),
                # (50, 150) of the MRCA is deleted in u, and (150, 200) duplicated in v
                ((50, 100, 0), (150, 200, 0), 1, (150, 200, 0)),
                ((50, 100, 0), (250, 300, 0), 1, (150, 200, 0)),
            },
        )

    def test_no_recomb_sv_dup_inv(self, all_sv_types_no_re_gig):
        assert same_homology(
            all_sv_types_no_re_gig.tables.find_homology(10, 12),
            {
                # "normal" region
                ((0, 20, 0), (0, 20, 0), 0, (0, 20, 0)),
                # start of inverted region
                ((20, 100, 0), (120, 40, 0), 0, (20, 100, 0)),
                # duplicated inverted region
                ((100, 120, 0), (40, 20, 0), 0, (100, 120, 0)),
                ((200, 220, 0), (40, 20, 0), 0, (100, 120, 0)),
                # duplicated non-inverted region
                ((120, 200, 0), (120, 200, 0), 0, (120, 200, 0)),
                ((220, 300, 0), (120, 200, 0), 0, (120, 200, 0)),
            },
        )

    def test_chromosomes(self):
        # u (node 0) and v (node 1) inherit from node 2: u on chromosome 1, and v on
        # chromosome 0, inverted
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1)
        tables.iedges.add_row(0, 2, child_right=10, child_chromosome=1, parent_chromosome=1)
        tables.iedges.add_row(
            1, 2, child_left=0, child_right=10, parent_left=10, parent_right=0, child_chromosome=0, parent_chromosome=1
        )
        tables.sort()
        tables.validate()
        expected = [gigl.Homology((0, 10, 1), (10, 0, 0), 2, (0, 10, 1))]
        assert same_homology(tables.find_homology(0, 1), expected)  # all chromosomes
        assert same_homology(tables.find_homology(0, 1, u_chromosome=1, v_chromosome=0), expected)

    @pytest.mark.parametrize("mode", ["first_mrca", "all"])
    def test_root_with_descendant(self, mode):
        # A root has no iedges above it, but has homology with its descendants
        tables = gigl.Tables()
        tables.nodes.add_row(1)
        tables.nodes.add_row(0)
        tables.iedges.add_row(1, 0, child_right=10)
        tables.sort()
        tables.validate()
        whole = gigl.Segment(0, 10, 0)
        expected = [gigl.Homology(whole, whole, 0, whole)]
        assert same_homology(tables.find_homology(0, 1, mode=mode), expected)
        assert same_homology(tables.find_homology(1, 0, mode=mode), expected)

    def test_chromosome_without_parent(self):
        # u (node 1) has a parent for chromosome 0, but not for chromosome 1, which its
        # child (node 2) inherits
        tables = gigl.Tables()
        tables.nodes.add_row(2)
        tables.nodes.add_row(1)
        tables.nodes.add_row(0)
        tables.iedges.add_row(1, 0, child_right=10)
        tables.iedges.add_row(2, 1, child_right=5, child_chromosome=1, parent_chromosome=1)
        tables.sort()
        tables.validate()
        segment = gigl.Segment(0, 5, 1)
        assert same_homology(tables.find_homology(1, 2), [gigl.Homology(segment, segment, 1, segment)])

    def test_graph_accepts_nodes(self, double_inversion_gig):
        gig = double_inversion_gig
        assert gig.find_homology(gig.nodes[0], gig.nodes[1]) == gig.tables.find_homology(0, 1)

    def test_bad_mode(self, double_inversion_gig):
        with pytest.raises(ValueError, match="Unknown mode"):
            double_inversion_gig.tables.find_homology(0, 1, mode="closest")

    def test_same_genome(self, double_inversion_gig):
        with pytest.raises(ValueError, match="different nodes"):
            double_inversion_gig.tables.find_homology(0, 0)


def copy_lost_tables():
    """
    A region B was duplicated in an ancestor A (node 1, from R, node 0), and u (node 2)
    inherits both copies, but v (node 3) only the first
    """
    tables = gigl.Tables()
    tables.nodes.add_row(10)  # R
    tables.nodes.add_row(5)  # A
    tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)  # u
    tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)  # v
    tables.iedges.add_row(1, 0, child_left=0, child_right=10, parent_left=0, parent_right=10)
    tables.iedges.add_row(1, 0, child_left=10, child_right=20, parent_left=0, parent_right=10)
    tables.iedges.add_row(2, 1, child_right=20)
    tables.iedges.add_row(3, 1, child_right=10)
    tables.sort()
    tables.validate()
    return tables


class TestFindHomologyModes:
    """
    The difference between "first_mrca" and "all": a region B was duplicated in an
    ancestor A (from R), and u inherits both copies, but v only the first
    """

    @pytest.fixture()
    def tables(self):
        return copy_lost_tables()

    def test_first_mrca(self, tables):
        # Once u's first copy and v's copy meet at A, v's lineage stops, so u's second
        # copy finds no homology
        assert same_homology(tables.find_homology(2, 3), [gigl.Homology((0, 10, 0), (0, 10, 0), 1, (0, 10, 0))])

    def test_all(self, tables):
        # u's second copy is homologous to v's copy through R (the closest homology of
        # u's second copy, as v has nothing closer)
        assert same_homology(
            tables.find_homology(2, 3, mode="all"),
            [
                gigl.Homology((0, 10, 0), (0, 10, 0), 1, (0, 10, 0)),
                gigl.Homology((10, 20, 0), (0, 10, 0), 0, (0, 10, 0)),
            ],
        )

    def test_all_reports_each_pair_once(self, tables):
        # u's first copy and v's copy also meet at R, but are reported at their MRCA, A
        homology = tables.find_homology(2, 3, mode="all")
        assert sum(h.length for h in homology) == 20

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_same_without_duplications(self, seed):
        # Without duplications, each position has at most one homologue
        ts = msprime.sim_ancestry(3, sequence_length=60, recombination_rate=0.03, random_seed=seed)
        tables = gigl.Tables.from_tree_sequence(ts)
        assert same_homology(tables.find_homology(0, 3), tables.find_homology(0, 3, mode="all"))


class TestFindClosestHomologues:
    def test_copy_lost(self):
        # As in TestFindHomologyModes: u's second copy's closest homologue is v's only
        # copy, through R, which "first_mrca" misses
        tables = copy_lost_tables()
        assert same_homology(
            tables.find_homology(2, 3, mode="closest_homologues"),
            [
                ((0, 10, 0), (0, 10, 0), 1, (0, 10, 0)),
                ((10, 20, 0), (0, 10, 0), 0, (0, 10, 0)),
            ],
        )

    def test_remote_homology_left_out(self):
        # A region B (positions 0-10 of R) was duplicated in A, and u and v both inherit
        # both copies. Each copy's closest homologue is the same copy in the other genome,
        # through A. The two copies are also homologous to each other through R, but that
        # is remote homology, found only by "all".
        tables = gigl.Tables()
        tables.nodes.add_row(10)  # R
        tables.nodes.add_row(5)  # A
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)  # u
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)  # v
        tables.iedges.add_row(1, 0, child_left=0, child_right=10, parent_left=0, parent_right=10)
        tables.iedges.add_row(1, 0, child_left=10, child_right=20, parent_left=0, parent_right=10)
        tables.iedges.add_row(2, 1, child_right=20)
        tables.iedges.add_row(3, 1, child_right=20)
        tables.sort()
        tables.validate()
        allelic = [((0, 20, 0), (0, 20, 0), 1, (0, 20, 0))]
        remote = [((0, 10, 0), (10, 20, 0), 0, (0, 10, 0)), ((10, 20, 0), (0, 10, 0), 0, (0, 10, 0))]
        assert same_homology(tables.find_homology(2, 3, mode="closest_homologues"), allelic)
        assert same_homology(tables.find_homology(2, 3, mode="all"), allelic + remote)

    @pytest.mark.parametrize("seed", range(10))
    def test_closest_part_of_all_homology(self, seed):
        # In a population with nested copies, and (unequal) crossovers: the closest
        # homologues are the pairs of positions, from all the homology, whose MRCA is the
        # most recent one for the position in u, or for the position in v
        tables = random_tables(seed, recombination=0.3)
        u, v = tables.sample_nodes()[:2]
        closest = homologous_positions(tables.find_homology(u, v, mode="closest_homologues"))
        everything = homologous_positions(tables.find_homology(u, v, mode="all"))
        time = tables.nodes.time
        # The time of the most recent MRCA of each position of u, and of v
        closest_in_u, closest_in_v = {}, {}
        for x, y, mrca, _ in everything:
            closest_in_u[x] = min(time[mrca], closest_in_u.get(x, np.inf))
            closest_in_v[y] = min(time[mrca], closest_in_v.get(y, np.inf))
        expected = set()
        for x, y, mrca, z in everything:
            if time[mrca] == closest_in_u[x] or time[mrca] == closest_in_v[y]:
                expected.add((x, y, mrca, z))
        assert closest == expected

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_same_without_duplications(self, seed):
        ts = msprime.sim_ancestry(3, sequence_length=60, recombination_rate=0.03, random_seed=seed)
        tables = gigl.Tables.from_tree_sequence(ts)
        assert same_homology(tables.find_homology(0, 3, mode="closest_homologues"), tables.find_homology(0, 3))


class TestHomology:
    def test_forward(self):
        h = gigl.Homology(gigl.Segment(10, 20, 0), gigl.Segment(30, 40, 1), 5, gigl.Segment(0, 10, 0))
        assert h.length == 10
        assert not h.is_reversed
        assert h.map == gigl.PositionMap(1, 20)
        assert h.map.position(15) == 35

    def test_reversed(self):
        h = gigl.Homology(gigl.Segment(10, 20, 0), gigl.Segment(40, 30, 0), 5, gigl.Segment(0, 10, 0))
        assert h.length == 10
        assert h.is_reversed
        assert h.map.position(10) == 39

    def test_unpack_segment_as_keywords(self, double_inversion_gig):
        gig = double_inversion_gig
        for h in gig.find_homology(0, 1):
            assert gig.haplotype(0, **h.u) == gig.haplotype(0)[h.u.left : h.u.right]


class TestHomologyRecords:
    """
    The properties that the records returned by find_homology promise, whatever the mode
    (but not e.g. where the homology is cut into records)
    """

    @staticmethod
    def examples():
        for seed in range(5):
            tables = random_tables(seed, recombination=0.3)
            u, v = tables.sample_nodes()[:2]
            yield tables, u, v
        tables = copy_lost_tables()
        yield tables, 2, 3

    @pytest.mark.parametrize("mode", ["first_mrca", "closest_homologues", "all"])
    def test_records(self, mode):
        for tables, u, v in self.examples():
            homology = tables.find_homology(u, v, mode=mode)
            assert homology == sorted(homology, key=lambda h: (h.u.chromosome, h.u.left))
            for h in homology:
                assert h.u.left < h.u.right
                assert abs(h.v.right - h.v.left) == abs(h.mrca_segment.right - h.mrca_segment.left) == h.length
                # The maps through the MRCA agree with the map from u to v
                u_to_mrca = gigl.PositionMap.from_intervals(h.u[:2], h.mrca_segment[:2])
                mrca_to_v = gigl.PositionMap.from_intervals(h.mrca_segment[:2], h.v[:2])
                assert u_to_mrca.then(mrca_to_v) == h.map

    def test_first_mrca_pairs_every_copy(self):
        # In mode "first_mrca", every copy in u of a segment of a common ancestor is
        # homologous through it to every copy in v (as random_homologous_positions needs)
        for tables, u, v in self.examples():
            records = tables.find_homology(u, v)
            # random_homologous_positions also needs the copies of a segment of a common
            # ancestor to be reported with the same segment, which it checks
            random_homologous_positions(records, np.random.default_rng(1))
            positions = homologous_positions(records)
            # The copies in u and in v of each position of a common ancestor, and the
            # pairs of them that are reported
            copies_in_u = collections.defaultdict(set)
            copies_in_v = collections.defaultdict(set)
            pairs = collections.defaultdict(set)
            for x, y, mrca, z in positions:
                copies_in_u[(mrca, z)].add(x)
                copies_in_v[(mrca, z)].add(y)
                pairs[(mrca, z)].add((x, y))
            for ancestral_position, reported in pairs.items():
                every_pair = {(x, y) for x in copies_in_u[ancestral_position] for y in copies_in_v[ancestral_position]}
                assert reported == every_pair


class TestSegment:
    # test the Segment class used in various places
    def test_segment_basic(self):
        seg = gigl.tables.Segment(10, 20, 1)
        assert seg.left == 10
        assert seg.right == 20
        assert seg.chromosome == 1
        assert str(seg) == "Segment(left=10, right=20, chromosome=1)"
        assert repr(seg) == "Segment(left=10, right=20, chromosome=1)"


class TestPosition:
    # Like Segment, the chromosome comes after the position
    def test_position_basic(self):
        pos = gigl.Position(10, 1)
        assert pos.position == 10
        assert pos.chromosome == 1
        assert pos == (10, 1)
        assert repr(pos) == "Position(position=10, chromosome=1)"

    @pytest.mark.parametrize(("cls", "values"), [("Position", (10, 1)), ("Segment", (10, 20, 1))])
    def test_still_a_tuple(self, cls, values):
        obj = getattr(gigl, cls)(*values)
        assert isinstance(obj, tuple)
        assert obj == values
        assert hash(obj) == hash(values)
        assert tuple(obj) == values  # iterating gives values, not field names
        assert obj[0] == values[0]
        assert obj[-1] == values[-1]
        assert obj[:2] == values[:2]
        assert dict(obj) == dict(zip(obj._fields, values))
        assert obj["chromosome"] == values[-1]
        with pytest.raises(AttributeError):
            obj["not_a_field"]

    def test_unpack_as_keywords(self):
        # Field names match argument names, so can be passed with **
        tables = gigl.Tables()
        tables.nodes.add_row(1)
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(1, 0, child_right=5, child_chromosome=1, parent_chromosome=1)
        for position, base in enumerate("ACGTA"):
            tables.mutations.add_row(0, **gigl.Position(position, 1), derived_state=base)
        tables.sort()
        gig = tables.graph()
        assert gig.allele_at(1, **gigl.Position(3, 1)) == "T"
        assert gig.haplotype(1, **gigl.Segment(1, 4, 1)) == "CGT"


class TestDecapitate:
    @staticmethod
    def three_generation_tables(with_individuals):
        # Grandparent (node 0, time 2), parent (node 1, time 1), child (node 2, time 0)
        tables = gigl.Tables()
        for t in (2, 1, 0):
            individual = tables.individuals.add_row() if with_individuals else gigl.NULL
            tables.nodes.add_row(t, individual=individual)
        tables.iedges.add_row(1, 0, child_right=10)
        tables.iedges.add_row(2, 1, child_right=10)
        tables.sort()
        return tables

    def test_no_individuals(self):
        # Previously failed with an IndexError
        tables = self.three_generation_tables(with_individuals=False)
        node_map = tables.decapitate(1.5)
        assert list(node_map) == [gigl.NULL, 0, 1]
        assert len(tables.nodes) == 2
        assert len(tables.individuals) == 0
        assert np.all(tables.nodes.individual == gigl.NULL)
        assert len(tables.iedges) == 1
        tables.graph()

    def test_individuals_without_parents(self):
        # Individuals added without parents have parents=None
        tables = self.three_generation_tables(with_individuals=True)
        assert tables.individuals[1].parents is None
        tables.decapitate(1.5)
        assert len(tables.individuals) == 2
        assert list(tables.nodes.individual) == [0, 1]

    def test_individual_parents_remapped(self):
        tables = gigl.Tables()
        old = tables.individuals.add_row()
        mid = tables.individuals.add_row(parents=(old,))
        young = tables.individuals.add_row(parents=(mid,))
        for t, i in ((2, old), (1, mid), (0, young)):
            tables.nodes.add_row(t, individual=i)
        tables.decapitate(1.5)
        assert len(tables.individuals) == 2
        # The parent of "mid" was removed, so becomes NULL
        assert tables.individuals[0].parents == (gigl.NULL,)
        assert tables.individuals[1].parents == (0,)


class TestMutationTable:
    def test_add_row(self):
        mutations = gigl.Tables().mutations
        assert mutations.add_row(1, 5, "A") == 0
        assert mutations.add_row(0, 3, "T", chromosome=2, time=1.5, metadata=b"x") == 1
        assert len(mutations) == 2
        assert list(mutations.node) == [1, 0]
        assert list(mutations.position) == [5, 3]
        assert list(mutations.chromosome) == [0, 2]
        assert mutations.derived_state == ["A", "T"]
        assert gigl.is_unknown_time(mutations.time[0])
        assert mutations.time[1] == 1.5
        row = mutations[1]
        assert isinstance(row, gigl.tables.MutationTableRow)
        assert row == (0, 3, 2, 1.5, "T", b"x")

    @pytest.mark.parametrize("state", [b"A", 1, None, ["A"]])
    def test_derived_state_must_be_string(self, state):
        with pytest.raises(TypeError, match="derived_state must be a string"):
            gigl.Tables().mutations.add_row(0, 0, state)

    @pytest.mark.parametrize("state", ["", "-", "ATTG"])
    def test_indel_alleles(self, state):
        mutations = gigl.Tables().mutations
        mutations.add_row(0, 0, state)
        assert mutations[0].derived_state == state

    def test_append(self):
        mutations = gigl.Tables().mutations
        mutations.add_row(1, 5, "A", time=2)
        mutations.append(mutations[0])
        assert mutations[0] == mutations[1]

    def test_copy_and_equality_with_unknown_times(self):
        tables = gigl.Tables()
        tables.mutations.add_row(0, 1, "A")  # unknown time is a NaN
        tables.mutations.add_row(0, 2, "C", time=1)
        copy = tables.copy()
        assert copy.mutations == tables.mutations
        assert copy == tables
        copy.mutations.add_row(0, 3, "G")
        assert copy.mutations != tables.mutations

    def test_inequality(self):
        a, b = gigl.Tables(), gigl.Tables()
        a.mutations.add_row(0, 1, "A")
        b.mutations.add_row(0, 1, "T")
        assert a.mutations != b.mutations
        b.mutations.clear()
        b.mutations.add_row(0, 1, "A", time=0)
        assert a.mutations != b.mutations

    def test_clear(self):
        mutations = gigl.Tables().mutations
        mutations.add_row(0, 1, "A")
        assert len(mutations.ids_for_node(0)) == 1
        mutations.clear()
        assert len(mutations) == 0
        assert len(mutations.ids_for_node(0)) == 0

    def test_str(self):
        tables = gigl.Tables()
        tables.mutations.add_row(0, 1, "A")
        assert "MUTATIONS" in str(tables)
        assert "derived_state" in str(tables.mutations)
        assert "mutations" in tables._repr_html_()

    def test_ids_for_node(self):
        mutations = gigl.Tables().mutations
        for node, chrom, pos in [(3, 0, 1), (3, 0, 1), (3, 0, 4), (3, 2, 0), (1, 0, 7)]:
            mutations.add_row(node, pos, "A", chromosome=chrom)
        assert list(mutations.ids_for_node(3)) == [0, 1, 2, 3]
        assert list(mutations.ids_for_node(3, chromosome=0)) == [0, 1, 2]
        assert list(mutations.ids_for_node(3, chromosome=2)) == [3]
        assert list(mutations.ids_for_node(3, chromosome=1)) == []
        assert list(mutations.ids_for_node(1)) == [4]
        assert list(mutations.ids_for_node(0)) == []
        # index is rebuilt after adding rows
        mutations.add_row(1, 8, "C")
        assert list(mutations.ids_for_node(1)) == [4, 5]

    def test_ids_for_node_not_adjacent(self):
        mutations = gigl.Tables().mutations
        for node in (0, 1, 0):
            mutations.add_row(node, 0, "A")
        with pytest.raises(ValueError, match="adjacent"):
            mutations.ids_for_node(0)

    @pytest.mark.parametrize("rows", [[(0, 0, 5), (0, 0, 4)], [(0, 1, 0), (0, 0, 9)]])
    def test_ids_for_node_unsorted(self, rows):
        mutations = gigl.Tables().mutations
        for node, chrom, pos in rows:
            mutations.add_row(node, pos, "A", chromosome=chrom)
        with pytest.raises(ValueError, match="ordered by chromosome, then position"):
            mutations.ids_for_node(0)

    def test_sort(self):
        tables = gigl.Tables()
        for t in (0, 2, 1):
            tables.nodes.add_row(t)
        # (node, chromosome, position, state), deliberately out of order
        rows = [(0, 0, 5, "a"), (2, 0, 1, "b"), (1, 0, 3, "c"), (1, 0, 3, "d"), (0, 1, 0, "e"), (0, 0, 5, "f")]
        for node, chrom, pos, state in rows:
            tables.mutations.add_row(node, pos, state, chromosome=chrom)
        tables.sort()
        # nodes in order of decreasing time (node 1 at time 2, node 2 at time 1, then
        # node 0), then chromosome, then position, and with mutations at the same node
        # and position ("c", "d" and "a", "f") kept in their original order
        assert tables.mutations.derived_state == ["c", "d", "b", "a", "f", "e"]
        # the sort is stable, so the same order is produced again
        tables.mutations.add_row(1, 3, "g")
        tables.sort()
        assert tables.mutations.derived_state == ["c", "d", "g", "b", "a", "f", "e"]

    def test_sort_bad_node(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0)
        tables.mutations.add_row(1, 0, "A")
        with pytest.raises(ValueError, match="not in the node table"):
            tables.sort()

    def test_change_times(self):
        tables = gigl.Tables()
        tables.nodes.add_row(1)
        tables.mutations.add_row(0, 0, "A", time=1.5)
        tables.mutations.add_row(0, 1, "C")
        tables.change_times(10)
        assert tables.nodes.time[0] == 11
        assert tables.mutations.time[0] == 11.5
        assert gigl.is_unknown_time(tables.mutations.time[1])


class TestMutationsInAlgorithms:
    @staticmethod
    def partly_ancestral_tables():
        """
        Root R (node 0, time 2) passes [0, 10) to A (node 1, time 1), which passes
        only [0, 5) to the sample S (node 2, time 0). So only [0, 5) of A and R is
        inherited by the sample.
        """
        tables = gigl.Tables()
        tables.nodes.add_row(2)
        tables.nodes.add_row(1)
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(1, 0, child_right=10)
        tables.iedges.add_row(2, 1, child_right=5)
        # (node, position, state)
        for node, pos, state in [(0, 2, "A"), (0, 8, "C"), (1, 3, "G"), (1, 7, "T"), (2, 4, "T"), (2, 9, "A")]:
            tables.mutations.add_row(node, pos, state)
        tables.sort()
        tables.validate()
        return tables

    def test_sample_resolve(self):
        tables = self.partly_ancestral_tables()
        before = tables.copy().graph().haplotype(2, right=10)
        tables.sample_resolve()
        # mutations at R:8 and A:7 are not inherited by the sample; S:9 is on a sample
        assert [(m.node, m.position) for m in tables.mutations] == [(0, 2), (1, 3), (2, 4), (2, 9)]
        assert tables.graph().haplotype(2, right=10) == before == "NNAGTNNNNA"

    def test_sample_resolve_on_graph(self):
        gig = self.partly_ancestral_tables().graph()
        new_gig = gig.sample_resolve()
        assert len(gig.mutations) == 6
        assert len(new_gig.mutations) == 4

    @pytest.mark.parametrize("seed", [1, 2])
    def test_sample_resolve_keeps_haplotypes(self, seed):
        ts = msprime.sim_ancestry(
            4, sequence_length=100, recombination_rate=0.01, record_full_arg=True, random_seed=seed
        )
        ts = msprime.sim_mutations(ts, rate=0.05, random_seed=seed)
        # Stop half the samples being samples, so that some of the ancestry (and
        # mutations) is no longer inherited by any sample
        tables = ts.dump_tables()
        flags = tables.nodes.flags
        flags[ts.samples()[::2]] = 0
        tables.nodes.flags = flags
        gig = gigl.Graph.from_tree_sequence(tables.tree_sequence())
        new_gig = gig.sample_resolve()
        assert len(new_gig.iedges) < len(gig.iedges)
        assert len(new_gig.mutations) < len(gig.mutations)
        for u in gig.sample_nodes():
            assert new_gig.haplotype(u, right=100) == gig.haplotype(u, right=100)

    def test_remove_non_ancestral_nodes(self):
        tables = self.partly_ancestral_tables()
        # Add an isolated node (3) with a mutation, which will be removed
        tables.nodes.add_row(0.5)
        tables.mutations.add_row(3, 1, "C")
        tables.sort()
        before = tables.copy().graph().haplotype(2, right=10)
        # Make node 0 isolated too, by giving A no parent: removed with its mutations
        tables.iedges.clear()
        tables.iedges.add_row(2, 1, child_right=5)
        tables.validate()
        kept = [m for m in tables.mutations if m.node != 0]
        tables.mutations.clear()
        for m in kept:
            tables.mutations.add_row(m.node, m.position, m.derived_state, chromosome=m.chromosome, time=m.time)
        mapping = tables.remove_non_ancestral_nodes()
        assert list(mapping) == [1, 2]
        assert len(tables.nodes) == 2
        # Nodes 1 and 2 are now nodes 0 and 1; node 3's mutation is removed
        assert [(m.node, m.position, m.derived_state) for m in tables.mutations] == [
            (0, 3, "G"),
            (0, 7, "T"),
            (1, 4, "T"),
            (1, 9, "A"),
        ]
        assert list(tables.mutations.ids_for_node(1)) == [2, 3]
        gig = tables.graph()
        assert gig.haplotype(1, right=10) == before.replace("A", "N", 1)  # lost R's mutation at 2

    def test_row_equality_with_unknown_times(self):
        mutations = gigl.Tables().mutations
        mutations.add_row(0, 0, "A")
        mutations.add_row(0, 0, "A", time=0)
        row = mutations[0]
        assert row == mutations[0]
        assert not (row != mutations[0])
        assert hash(row) == hash(mutations[0])
        assert len({row, mutations[0]}) == 1
        assert row != mutations[1]
        assert mutations[1] == (0, 0, 0, 0.0, "A", None)
        assert row != (0, 0, 0)

    def test_copy_keeps_index(self):
        mutations = gigl.Tables().mutations
        for node in (2, 2, 1):
            mutations.add_row(node, 0, "A")
        copy = mutations.copy()
        assert list(copy.ids_for_node(2)) == [0, 1]
        copy.add_row(2, 1, "C")  # makes node 2 non-adjacent in the copy only
        with pytest.raises(ValueError, match="adjacent"):
            copy.ids_for_node(2)
        assert list(mutations.ids_for_node(2)) == [0, 1]


class TestDecapitateWithMutations:
    @staticmethod
    def simulated_gig(seed):
        ts = msprime.sim_ancestry(4, sequence_length=60, recombination_rate=0.02, random_seed=seed)
        ts = msprime.sim_mutations(ts, rate=0.1, random_seed=seed)
        return gigl.Graph.from_tree_sequence(ts)

    @pytest.mark.parametrize("seed", [1, 2])
    @pytest.mark.parametrize("quantile", [0.1, 0.5, 0.9])
    def test_haplotypes_preserved(self, seed, quantile):
        gig = self.simulated_gig(seed)
        time = np.quantile(gig.tables.nodes.time, quantile)
        new_gig = gig.decapitate(time)
        node_map = gig.tables.copy().decapitate(time)
        assert np.all(new_gig.tables.mutations.time[~gigl.is_unknown_time(new_gig.tables.mutations.time)] < time)
        for u, new_u in enumerate(node_map):
            if new_u != gigl.NULL:
                assert new_gig.haplotype(new_u, right=60) == gig.haplotype(u, right=60)

    @pytest.mark.parametrize("seed", [1, 2])
    def test_discard_old_mutations(self, seed):
        gig = self.simulated_gig(seed)
        time = np.quantile(gig.tables.nodes.time, 0.5)
        new_gig = gig.decapitate(time, discard_old_mutations=True)
        assert len(new_gig.mutations) < len(gig.mutations)
        n_missing = 0
        for u in gig.sample_nodes():
            old, new = gig.haplotype(u, right=60), new_gig.haplotype(u, right=60)
            # each allele is either unchanged, or now missing
            assert all(a == b or b == "N" for a, b in zip(old, new))
            n_missing += sum(a != b for a, b in zip(old, new))
        assert n_missing > 0

    @staticmethod
    def three_generation_tables(child_iedge=(0, 10, 0, 10)):  # (cl, cr, pl, pr) of S's iedge
        """
        Root R (node 0, time 3), A (node 1, time 2), and sample S (node 2, time 0).
        """
        tables = gigl.Tables()
        tables.nodes.add_row(3)
        tables.nodes.add_row(2)
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(1, 0, child_right=10)
        coords = dict(zip(("child_left", "child_right", "parent_left", "parent_right"), child_iedge))
        tables.iedges.add_row(2, 1, **coords)
        for pos, state in enumerate("ACGTACGTAC"):
            tables.mutations.add_row(0, pos, state)
        tables.sort()
        return tables

    def test_inherited_alleles_precede_existing_mutations(self):
        tables = self.three_generation_tables()
        tables.mutations.add_row(2, 3, "C", time=0.5)  # S's own (young) mutation
        tables.mutations.add_row(1, 4, "G", time=2.5)  # old, on A's branch: removed
        tables.sort()
        gig = tables.graph()
        assert gig.haplotype(2) == "ACGCGCGTAC"
        new_gig = gig.decapitate(1)  # removes R and A
        assert len(new_gig.nodes) == 1
        # S now has no iedges, but its sequence length is given by its mutations
        assert new_gig.sequence_length(0, chromosome=0) == 10
        assert new_gig.haplotype(0) == "ACGCGCGTAC"
        # inherited alleles have unknown times, and come before S's own mutation at 3
        at_3 = [m for m in new_gig.mutations if m.position == 3]
        assert [m.derived_state for m in at_3] == ["T", "C"]
        assert not at_3[0].is_time_known()
        assert at_3[1].time == 0.5

    def test_old_mutation_on_kept_branch(self):
        # A mutation above S, older than the cutoff, is part of what S inherits
        tables = self.three_generation_tables()
        tables.mutations.add_row(2, 5, "T", time=1.5)
        tables.sort()
        gig = tables.graph()
        new_gig = gig.decapitate(1)
        assert new_gig.haplotype(0, right=10) == gig.haplotype(2) == "ACGTATGTAC"
        assert all(not m.is_time_known() for m in new_gig.mutations)

    def test_inversion_across_cutoff(self):
        # S inherits A's genome inverted
        gig = self.three_generation_tables(child_iedge=(0, 10, 10, 0)).graph()
        assert gig.haplotype(2) == "GTACGTACGT"
        new_gig = gig.decapitate(1)
        assert new_gig.haplotype(0, right=10) == "GTACGTACGT"
        # the new mutations are on S's own strand, so are not complemented again
        assert new_gig.allele_at(0, 0) == "G"

    def test_missing_alleles_not_added(self):
        tables = self.three_generation_tables()
        tables.mutations.clear()
        tables.mutations.add_row(0, 4, "A")
        tables.sort()
        new_gig = tables.graph().decapitate(1)
        assert [(m.node, m.position, m.derived_state) for m in new_gig.mutations] == [(0, 4, "A")]

    def test_no_change_if_cutoff_is_older(self):
        gig = self.three_generation_tables().graph()
        new_gig = gig.decapitate(10)
        assert new_gig.tables.mutations == gig.tables.mutations

    def test_unknown_time_mutation_before_removed_one(self):
        # S has an unknown-time mutation, followed (in table order) by an old one that is
        # removed. The old one overwrites the unknown-time one, so both are replaced.
        tables = self.three_generation_tables()
        tables.mutations.add_row(2, 3, "C")
        tables.mutations.add_row(2, 3, "G", time=1.5)
        tables.sort()
        gig = tables.graph()
        assert gig.allele_at(2, 3) == "G"
        new_gig = gig.decapitate(1)
        assert new_gig.haplotype(0, right=10) == gig.haplotype(2)
        assert [m.derived_state for m in new_gig.mutations_for_node(0) if m.position == 3] == ["G"]

    def test_old_mutation_above_kept_root(self):
        # A root younger than the cutoff loses its own (old) mutations
        tables = gigl.Tables()
        tables.nodes.add_row(2)
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(1, 0, child_right=10)
        tables.mutations.add_row(0, 4, "A", time=4)
        tables.mutations.add_row(0, 5, "C")
        tables.sort()
        gig = tables.graph()
        new_gig = gig.decapitate(3)
        assert len(new_gig.nodes) == 2
        assert new_gig.haplotype(1) == gig.haplotype(1) == "NNNNACNNNN"
        assert all(not m.is_time_known() for m in new_gig.mutations)

    def test_iedge_chromosomes_kept(self):
        tables = gigl.Tables()
        tables.nodes.add_row(3)
        tables.nodes.add_row(2)
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(1, 0, child_right=10, child_chromosome=1, parent_chromosome=1)
        tables.iedges.add_row(2, 1, child_right=10, child_chromosome=1, parent_chromosome=1)
        tables.mutations.add_row(0, 4, "A", chromosome=1)
        tables.sort()
        gig = tables.graph()
        for time in (2.5, 10):
            new_gig = gig.decapitate(time)
            node_map = gig.tables.copy().decapitate(time)
            assert len(new_gig.iedges) > 0
            assert np.all(new_gig.tables.iedges.child_chromosome == 1)
            assert np.all(new_gig.tables.iedges.parent_chromosome == 1)
            assert new_gig.allele_at(node_map[2], 4, chromosome=1) == "A"

    def test_long_iedge_with_few_mutations(self):
        # The time taken should depend on the number of mutations, not the iedge length
        L = 10**9
        tables = gigl.Tables()
        tables.nodes.add_row(3)
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(1, 0, child_right=L)
        tables.mutations.add_row(0, L - 1, "A")
        tables.sort()
        new_gig = tables.graph().decapitate(1)
        assert [(m.node, m.position, m.derived_state) for m in new_gig.mutations] == [(0, L - 1, "A")]


class TestNodeFlagConstants:
    @pytest.mark.parametrize("flag", [gigl.NODE_IS_SAMPLE, gigl.Const.NODE_IS_RE])
    def test_clear_flag_with_mask(self, flag):
        flags = np.array([flag | 2, 2], dtype=np.uint32)
        flags &= ~flag
        assert flags.tolist() == [2, 2]


# (child interval, parent interval) pairs, as in iedges
MAP_INTERVALS = [
    ((10, 20), (10, 20)),  # no change
    ((0, 10), (10, 20)),  # shifted
    ((10, 20), (20, 10)),  # inverted in place
    ((10, 20), (30, 20)),  # inverted and shifted
    ((0, 5), (5, 0)),  # inverted at the start of the genome
]


class TestPositionMap:
    def test_docstring_example(self):
        m = gigl.PositionMap.from_intervals((10, 20), (30, 20))
        assert m == gigl.PositionMap(-1, 40)
        assert m.interval(10, 20) == (30, 20)
        assert (m.position(10), m.position(19)) == (29, 20)

    @pytest.mark.parametrize(("child", "parent"), MAP_INTERVALS)
    def test_from_intervals(self, child, parent):
        m = gigl.PositionMap.from_intervals(child, parent)
        assert m.interval(*child) == parent
        assert m.is_reversing == ((child[0] < child[1]) != (parent[0] < parent[1]))
        assert m.inverse().interval(*parent) == child

    @pytest.mark.parametrize(("child", "parent"), MAP_INTERVALS)
    def test_positions(self, child, parent):
        # Each position maps to the position between the ends its own ends map to
        m = gigl.PositionMap.from_intervals(child, parent)
        positions = [m.position(x) for x in range(*child)]
        assert sorted(positions) == list(range(min(parent), max(parent)))
        for x in range(*child):
            assert m.position(x) == min(m.interval(x, x + 1))
            assert m.inverse().position(m.position(x)) == x

    def test_different_lengths(self):
        with pytest.raises(ValueError, match="different lengths"):
            gigl.PositionMap.from_intervals((0, 10), (0, 11))

    def test_from_iedge(self):
        iedges = gigl.tables.IEdgeTable()
        iedges.add_row(0, 1, child_left=10, child_right=20, parent_left=30, parent_right=20)
        assert gigl.PositionMap.from_iedge(iedges[0]) == gigl.PositionMap.from_intervals((10, 20), (30, 20))

    @pytest.mark.parametrize(("a", "b"), [(x, y) for x in MAP_INTERVALS for y in MAP_INTERVALS])
    def test_then(self, a, b):
        # Mapping through a chain is the same as mapping through each in turn
        a = gigl.PositionMap.from_intervals(*a)
        b = gigl.PositionMap.from_intervals(*b)
        both = a.then(b)
        for x in range(-5, 30):
            assert both.position(x) == b.position(a.position(x))
            assert both.interval(x, x + 3) == b.interval(*a.interval(x, x + 3))
        assert both.is_reversing == (a.is_reversing != b.is_reversing)
        assert a.then(a.inverse()) == gigl.PositionMap(1, 0)


class TestIncrementalValidation:
    """
    Validating the rows added since the last validation gives the same result as
    checking all the rows
    """

    @staticmethod
    def random_row(rng, num_nodes):
        # A row that may break any of the requirements (e.g. refer to a missing node)
        left, span = (int(x) for x in rng.integers(0, 20, size=2))
        parent_left = int(rng.integers(0, 20))
        parent_span = span if rng.random() < 0.8 else span + 1
        if rng.random() < 0.3:
            parent_left, parent_right = parent_left + parent_span, parent_left  # inversion
        else:
            parent_right = parent_left + parent_span
        return {
            "child": int(rng.integers(0, num_nodes + 1)),
            "parent": int(rng.integers(0, num_nodes + 1)),
            "child_left": left,
            "child_right": left + span if rng.random() < 0.9 else left,
            "parent_left": parent_left,
            "parent_right": parent_right,
            "child_chromosome": int(rng.integers(0, 2)),
            "edge": int(rng.choice([gigl.NULL, 0, 1])),
        }

    @staticmethod
    def error(tables, flags, **kwargs):
        try:
            tables.validate(flags, **kwargs)
        except ValueError as err:
            return str(err)
        return None

    @pytest.mark.parametrize("seed", range(10))
    def test_matches_full_validation(self, seed):
        rng = np.random.default_rng(seed)
        ts = msprime.sim_ancestry(4, sequence_length=40, recombination_rate=0.05, random_seed=seed + 1)
        source = gigl.Tables.from_tree_sequence(ts)
        rows = [row._asdict() for row in source.iedges]
        # Add the (valid) rows a few at a time, sometimes with an extra random row
        flag_sets = [*ValidFlags.IEDGES_ALL, ValidFlags.GIG]
        states = {}
        for flags in flag_sets:
            states[flags] = gigl.Tables()
            states[flags].nodes = source.nodes.copy()
        i = 0
        while i < len(rows):
            chunk = rows[i : i + int(rng.integers(1, 4))]
            i += len(chunk)
            if rng.random() < 0.1:
                chunk.append(self.random_row(rng, len(source.nodes)))
            for flags, tables in states.items():
                for row in chunk:
                    tables.iedges.add_row(**row)
                expected = self.error(tables.copy(), flags, recheck=True)
                if expected is None:
                    assert self.error(tables, flags) is None
                else:
                    assert self.error(tables, flags) is not None


class TestValidatedRows:
    """
    Methods that read the iedges of particular nodes can run if those iedges are
    among the validated rows, even if rows added since then have not been validated
    """

    @staticmethod
    def tables_with_new_rows():
        # Validated tables, plus new rows that are not validated (and are invalid)
        tables = gigl.Tables()
        for t in (2, 1, 1, 0):
            tables.nodes.add_row(t)
        tables.iedges.add_row(1, 0, child_right=10)
        tables.iedges.add_row(2, 0, child_right=10)
        tables.validate()
        tables.iedges.add_row(3, 1, child_right=10)
        tables.iedges.add_row(3, 3, child_right=5)  # overlapping, and its own parent
        return tables

    def test_num_validated(self):
        tables = self.tables_with_new_rows()
        assert tables.iedges.num_validated == 2
        assert tables.iedges.flags == 0
        tables.iedges.set_bitflag(ValidFlags.IEDGES_INTERVALS)  # a promise about all rows
        assert tables.iedges.num_validated == 4
        assert tables.iedges.flags == ValidFlags.IEDGES_INTERVALS

    def test_find_homology(self):
        tables = self.tables_with_new_rows()
        whole = gigl.Segment(0, 10, 0)
        assert same_homology(tables.find_homology(1, 2), [gigl.Homology(whole, whole, 0, whole)])

    def test_refuses_nodes_with_unvalidated_rows(self):
        tables = self.tables_with_new_rows()
        with pytest.raises(ValueError, match="node 3 include rows that have not been validated"):
            tables.find_homology(3, 2)
        with pytest.raises(ValueError, match="have not been validated"):
            tables.iedges.ids_for_child(3)

    def test_whole_table_methods_need_all_rows(self):
        tables = self.tables_with_new_rows()
        with pytest.raises(ValueError, match="requires the iedges to have these validity flags"):
            tables.sample_resolve()
        with pytest.raises(ValueError, match="not older than child 3"):
            tables.validate()

    def test_stale_node_flags_rechecked_for_validated_rows_only(self):
        tables = self.tables_with_new_rows()
        tables.change_times(1)  # the node-time flags must be rechecked
        assert [h.mrca for h in tables.find_homology(1, 2)] == [0]
        tables.nodes.clear()
        for t in (0, 1, 1, 0):
            tables.nodes.add_row(t)  # now the root is younger than its children
        with pytest.raises(ValueError, match="not older than child"):
            tables.find_homology(1, 2)


class TestValidationEdgeCases:
    def test_copied_node_tables_have_their_own_version(self):
        tables = gigl.Tables()
        tables.nodes.add_row(1)
        nodes_copy = tables.nodes.copy()
        tables.nodes.add_row(0)
        nodes_copy.add_row(2)  # the copy diverges, so is no longer valid for the iedges
        tables.iedges.add_row(1, 0, child_right=1)
        tables.validate()
        tables.nodes = nodes_copy
        with pytest.raises(ValueError, match="not older than child"):
            tables.validate()

    def test_sort_after_sample_resolve_keeps_flags(self):
        tables = TestValidate.valid_tables()
        tables.validate()
        tables.sample_resolve(sort=False)
        tables.sort()
        assert tables.iedges.has_all_bitflags(ValidFlags.IEDGES_FOR_TRACING)
        tables.find_homology(2, 3)

    def test_validate_frozen_empty_tables(self):
        tables = gigl.Tables()
        tables.freeze()
        tables.validate()
        tables.validate(recheck=True)

    def test_span_overflow(self):
        tables = gigl.Tables()
        tables.iedges.add_row(1, 0, child_left=0, child_right=1, parent_left=-(2**63), parent_right=2**63 - 1)
        with pytest.raises(ValueError, match="different parent and child spans"):
            tables.validate(ValidFlags.IEDGES_INTERVALS)

    def test_chromosome_order_overflow(self):
        tables = gigl.Tables()
        for chromosome in (-30000, 0, 30000, -30000):
            tables.iedges.add_row(1, 0, child_right=1, child_chromosome=chromosome)
        with pytest.raises(ValueError, match="not sorted by chromosome"):
            tables.validate(ValidFlags.IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC)

    @pytest.mark.parametrize("incremental", [False, True])
    def test_overlap_with_previous_iedge(self, incremental):
        # The third iedge starts after the second, but before it ends
        tables = gigl.Tables()
        tables.nodes.add_row(1)
        tables.nodes.add_row(0)
        tables.iedges.add_row(1, 0, child_left=0, child_right=10)
        tables.iedges.add_row(1, 0, child_left=10, child_right=12)
        if incremental:
            tables.validate()
        tables.iedges.add_row(1, 0, child_left=11, child_right=13)
        with pytest.raises(ValueError, match="multiple or duplicate parents at position 11"):
            tables.validate()
