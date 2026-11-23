import msprime
import numpy as np
import pytest
import tskit
from matplotlib import pyplot as plt

import giglib as gigl
from giglib.constants import Const, ValidFlags


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

    @staticmethod
    def valid_tables():
        # Valid as a GIG, but added without validation, so no iedge flags are set
        tables = gigl.Tables()
        tables.nodes.add_row(2)  # 0
        tables.nodes.add_row(1)  # 1
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)  # 2
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)  # 3
        tables.iedges.add_row(1, 0, child_right=10, edge=0)
        tables.iedges.add_row(2, 1, child_right=4, edge=1)
        tables.iedges.add_row(2, 0, child_left=4, child_right=10, parent_left=10, parent_right=4)
        tables.iedges.add_row(3, 1, child_right=10, child_chromosome=1, parent_chromosome=0)
        tables.iedges.unset_bitflag(ValidFlags.IEDGES_ALL)
        return tables

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
        tables.find_mrca_regions(2, 3)
        tables.nodes.clear()
        for t in (0, 1, 0, 0):
            tables.nodes.add_row(t)  # now the parent is no older than the children
        with pytest.raises(ValueError, match="not older than child"):
            tables.find_mrca_regions(2, 3)

    @staticmethod
    def record_checks(tables, monkeypatch):
        # Record the (flags, start) of each check of some iedges
        checks = []
        check = tables._check_iedges

        def recording_check(flags, start=0, stop=None):
            if flags and start < (len(tables.iedges) if stop is None else stop):
                checks.append((flags, start))
            check(flags, start, stop)

        monkeypatch.setattr(tables, "_check_iedges", recording_check)
        return checks

    def test_only_new_rows_are_checked(self, monkeypatch):
        tables = self.valid_tables()
        tables.validate()
        checks = self.record_checks(tables, monkeypatch)
        tables.validate()
        assert checks == []  # nothing new to check
        n = len(tables.iedges)
        tables.nodes.add_row(0)  # adding nodes doesn't affect the existing iedges
        tables.iedges.add_row(4, 1, child_right=10)
        tables.validate()
        assert checks == [(ValidFlags.GIG, n)]
        assert tables.iedges.flags == ValidFlags.GIG

    def test_newly_required_flags_checked_for_all_rows(self, monkeypatch):
        tables = self.valid_tables()
        tables.validate(ValidFlags.IEDGES_INTERVALS)
        checks = self.record_checks(tables, monkeypatch)
        tables.iedges.add_row(3, 1, child_left=10, child_right=12, child_chromosome=1)
        tables.validate(ValidFlags.IEDGES_INTERVALS | ValidFlags.IEDGES_INTEGERS)
        assert checks == [(ValidFlags.IEDGES_INTEGERS, 0), (ValidFlags.IEDGES_INTERVALS, len(tables.iedges) - 1)]

    def test_changed_nodes_rechecked_for_all_rows(self, monkeypatch):
        tables = self.valid_tables()
        tables.validate()
        checks = self.record_checks(tables, monkeypatch)
        tables.change_times(1)
        tables.validate()
        assert checks == [(ValidFlags.IEDGES_COMBO_NODE_TABLE, 0)]

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

    @pytest.mark.parametrize("method", ["find_mrca_regions", "sample_resolve"])
    def test_tracing_requires_all_its_flags(self, method):
        tables = self.valid_tables()
        tables.validate(ValidFlags.IEDGES_PARENT_OLDER_THAN_CHILD)  # but not e.g. adjacency
        args = (2, 3) if method == "find_mrca_regions" else ()
        with pytest.raises(ValueError, match="FOR_CHILD_ADJACENT"):
            getattr(tables, method)(*args)
        tables.validate(ValidFlags.IEDGES_FOR_TRACING)
        getattr(tables, method)(*args)

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

    def test_find_mrca_regions_after_sample_resolve(self):
        # e.g. in a simulation that simplifies regularly, without validating again
        tables = self.ts_tables()
        tables.sample_resolve()
        u, v = tables.sample_nodes()[:2]
        assert len(tables.find_mrca_regions(u, v)) > 0

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
            tables.find_mrca_regions(u, v)
        tables.sort()
        tables.find_mrca_regions(u, v)


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
        # Based on the corresponding test code in tskit
        nodes = gigl.Tables().nodes
        for _ in range(num_rows):
            nodes.append({"time": 0, "flags": 0})
        html = nodes._repr_html_()
        if num_rows == 50:
            assert len(html.splitlines()) == num_rows + 11
            assert "10 rows skipped (giglib.set_print_options)" in html.split("</tr>")[21]
        else:
            assert len(html.splitlines()) == num_rows + 20


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


class TestFindMrcas:
    def test_find_mrca_single_tree(self):
        span = 123
        comb_ts = tskit.Tree.generate_comb(5, span=span).tree_sequence
        # make a gig with many unary nodes above node 0
        comb_ts = comb_ts.simplify([0, 4], keep_unary=True)
        gig = gigl.Graph.from_tree_sequence(comb_ts)
        full_span = (0, span)
        shared_regions = gig.tables.find_mrca_regions(0, 1)
        assert len(shared_regions) == 1
        assert comb_ts.first().root in shared_regions
        mrca_intervals = shared_regions[comb_ts.first().root]
        assert len(mrca_intervals) == 1
        mrca, uv_interval_lists = mrca_intervals.popitem()
        assert mrca[:2] == full_span
        assert len(uv_interval_lists) == 2
        for interval_list in uv_interval_lists:
            assert len(interval_list) == 1
            assert full_span == interval_list[0][:2]

    def test_find_mrca_2_trees(self, degree2_2_tip_ts):
        num_trees = 2
        assert degree2_2_tip_ts.num_trees == num_trees
        gig = gigl.Graph.from_tree_sequence(degree2_2_tip_ts)
        shared_regions = gig.tables.find_mrca_regions(0, 1)
        internal_nodes = [node.id for node in degree2_2_tip_ts.nodes() if node.time > 0]
        assert len(shared_regions) == num_trees
        for mrca_id in internal_nodes:
            assert mrca_id in shared_regions
            val = shared_regions[mrca_id]
            assert len(val) == 1
            interval, (u, v) = val.popitem()
            assert len(u) == 1
            assert len(v) == 1
            assert interval in u
            assert interval in v

    def test_time_cutoff(self, degree2_2_tip_ts):
        # set a cutoff so that the mrca in one tree is never visited
        assert degree2_2_tip_ts.num_trees == 2
        assert degree2_2_tip_ts.num_samples == 2
        T = degree2_2_tip_ts.nodes_time
        tree0 = degree2_2_tip_ts.first()
        tree1 = degree2_2_tip_ts.last()
        assert T[tree0.root] != T[tree1.root]
        gig = gigl.Graph.from_tree_sequence(degree2_2_tip_ts)
        assert gig.sequence_length(0, chromosome=0) == degree2_2_tip_ts.sequence_length
        assert gig.sequence_length(1, chromosome=0) == degree2_2_tip_ts.sequence_length
        cutoff = (T[[tree0.root, tree1.root]]).mean()
        shared_regions = gig.tables.find_mrca_regions(0, 1, time_cutoff=cutoff)
        assert len(shared_regions) == 1
        used_tree = tree0 if T[tree0.root] < T[tree1.root] else tree1
        unused_tree = tree1 if T[tree0.root] < T[tree1.root] else tree0
        assert used_tree.root in shared_regions
        assert unused_tree.root not in shared_regions

    def test_simple_non_sv(self, simple_ts):
        chrom = 0
        assert simple_ts.num_trees > 1
        assert simple_ts.num_samples >= 2
        max_trees = 0
        gig = gigl.Graph.from_tree_sequence(simple_ts)
        for u in range(len(gig.tables.sample_nodes())):
            for v in range(u + 2, len(gig.tables.sample_nodes())):
                mrcas = gig.tables.find_mrca_regions(u, v)
                equiv_ts = simple_ts.simplify([u, v], filter_nodes=False)
                max_trees = max(max_trees, equiv_ts.num_trees)
                for tree in equiv_ts.trees():
                    interval = (
                        int(tree.interval.left),
                        int(tree.interval.right),
                        chrom,
                    )
                    mrca = tree.get_mrca(u, v)
                    assert interval in mrcas[mrca]
                    u_equivalent, v_equivalent = mrcas[mrca][interval]
                    assert len(u_equivalent) == 1  # No duplications
                    assert len(v_equivalent) == 1  # No duplications
                    assert interval in u_equivalent
                    assert interval in v_equivalent
        assert max_trees > 2  # at least some cases with 3 or more mrcas

    def test_double_inversion(self, double_inversion_gig):
        chrom = 0
        assert len(double_inversion_gig.sample_nodes()) == 2
        iedges = list(double_inversion_gig.iedges_for_child(0))
        assert len(iedges) == 1
        assert iedges[0].is_inversion()
        mrcas = double_inversion_gig.tables.find_mrca_regions(0, 1)
        assert len(mrcas) == 1
        assert 3 in mrcas
        mrca = mrcas[3]
        assert (0, 100, chrom) in mrca
        u, v = mrca[(0, 100, chrom)]
        assert u == [(0, 100, chrom)]
        assert v == [(0, 100, chrom)]

    @pytest.mark.parametrize("sample_resolve", [True, False])
    def test_extended_inversion(self, extended_inversion_gig, sample_resolve):
        chrom = 0
        gig = extended_inversion_gig
        if sample_resolve:
            gig = gig.sample_resolve()
        assert set(gig.tables.sample_nodes()) == {0, 1}
        assert gig.min_pos_as_child(0) == 20
        assert gig.max_pos(0) == 155
        assert gig.min_pos_as_child(1) == 0
        assert gig.max_pos(1) == 100
        mrcas = gig.tables.find_mrca_regions(0, 1)
        assert len(mrcas) == 1
        assert 3 in mrcas
        mrca = mrcas[3]
        assert (15, 100, chrom) in mrca  # the inverson only leaves 15..100 shared
        u, v = mrca[(15, 100, chrom)]
        assert len(v) == 1
        assert (15, 100, chrom) in v
        assert len(u) == 1
        assert (
            155,
            70,
            chrom,
        ) in u  # inverted region is span 85 from 155 leftward in sample

    def test_inverted_duplicate(self, inverted_duplicate_gig):
        chrom = 0
        assert len(inverted_duplicate_gig.sample_nodes()) == 2
        mrcas = inverted_duplicate_gig.tables.find_mrca_regions(0, 1)
        assert len(mrcas) == 1
        assert 3 in mrcas
        mrca = mrcas[3]
        assert (10, 15, chrom) in mrca  # the duplicated+inverted section
        u, v = mrca[(10, 15, chrom)]
        assert set(u) == {(0, 5, chrom), (15, 10, chrom)}
        assert v == [(0, 5, chrom)]

        assert (15, 20, chrom) in mrca  # only the inverted section
        u, v = mrca[(15, 20, chrom)]
        assert u == [(10, 5, chrom)]
        assert v == [(5, 10, chrom)]

    def test_inverted_duplicate_with_missing(self, inverted_duplicate_with_missing_gig):
        chrom = 0
        assert len(inverted_duplicate_with_missing_gig.sample_nodes()) == 2
        mrcas = inverted_duplicate_with_missing_gig.tables.find_mrca_regions(0, 1)
        assert len(mrcas) == 1
        assert 3 in mrcas
        mrca = mrcas[3]
        assert (10, 15, chrom) in mrca  # the duplicated+inverted section
        u, v = mrca[(10, 15, chrom)]
        assert set(u) == {(0, 5, chrom), (35, 30, chrom)}
        assert v == [(0, 5, chrom)]

        assert (15, 20, chrom) in mrca  # only the inverted section
        u, v = mrca[(15, 20, chrom)]
        assert u == [(30, 25, chrom)]
        assert v == [(5, 10, chrom)]

    def test_no_recomb_sv_dup_del(self, all_sv_types_no_re_gig):
        chrom = 0
        assert len(all_sv_types_no_re_gig.sample_nodes()) >= 2
        sample_u = 8
        sample_v = 10
        mrcas = all_sv_types_no_re_gig.tables.find_mrca_regions(sample_u, sample_v)
        assert len(mrcas) == 1
        assert 1 in mrcas

        # (0, 50) should be unchanged
        assert (0, 50, chrom) in mrcas[1]
        u, v = mrcas[1][(0, 50, chrom)]
        assert len(u) == len(v) == 1
        assert (0, 50, chrom) in u
        assert (0, 50, chrom) in v

        # (150, 50) should be duplicated
        assert (150, 200, chrom) in mrcas[1]
        u, v = mrcas[1][(150, 200, chrom)]
        assert len(u) == 1
        assert (50, 100, chrom) in u  # deletion
        assert len(v) == 2  # duplication
        assert (150, 200, chrom) in v  # duplication
        assert (250, 300, chrom) in v  # duplication

        # (50, 150) should be deleted, with no MRCA
        assert (50, 150) not in mrcas[1]
        assert len(mrcas[1]) == 2

    def test_no_recomb_sv_dup_inv(self, all_sv_types_no_re_gig):
        chrom = 0
        assert len(all_sv_types_no_re_gig.sample_nodes()) >= 2
        sample_u = 10
        sample_v = 12
        mrcas = all_sv_types_no_re_gig.tables.find_mrca_regions(sample_u, sample_v)
        assert len(mrcas) == 1
        assert 0 in mrcas
        mrca = mrcas[0]
        # "normal" region
        assert (0, 20, chrom) in mrca
        assert mrca[(0, 20, chrom)] == ([(0, 20, chrom)], [(0, 20, chrom)])

        # start of inverted region
        assert (20, 100, chrom) in mrca
        assert mrca[(20, 100, chrom)] == ([(20, 100, chrom)], [(120, 40, chrom)])

        # duplicated inverted region
        assert (100, 120, chrom) in mrca
        assert set(mrca[(100, 120, chrom)][0]) == {(100, 120, chrom), (200, 220, chrom)}
        assert mrca[(100, 120, chrom)][1] == [(40, 20, chrom)]

        # duplicated non-inverted region
        assert (120, 200, chrom) in mrca
        assert set(mrca[(120, 200, chrom)][0]) == {(120, 200, chrom), (220, 300, chrom)}
        assert mrca[(120, 200, chrom)][1] == [(120, 200, chrom)]

    def test_random_match_pos(self, simple_ts):
        rng = np.random.default_rng(1)
        ts = simple_ts.keep_intervals([(0, 2)]).trim()
        gig = gigl.Graph.from_tree_sequence(ts)
        assert gig.tables.sample_nodes()[0] == 0
        assert gig.tables.sample_nodes()[1] == 1
        mrcas = gig.tables.find_mrca_regions(0, 1)
        all_breaks = set()
        for _ in range(20):
            # in 20 replicates we should definitely have both 0 and 1
            breaks = mrcas.random_match_pos(rng)
            assert isinstance(breaks.u, gigl.Position)
            assert isinstance(breaks.v, gigl.Position)
            assert not breaks.opposite_orientations
            assert breaks.u == breaks.v
            all_breaks.add(breaks.u)
        assert all_breaks == {(0, 0), (1, 0)}


class TestMRCAdict:
    def test_plot(self):
        MRCAintervals = gigl.tables.MRCAdict.MRCAintervals
        mrcas = gigl.tables.MRCAdict()
        mrcas[0] = {
            (0, 10): MRCAintervals([(0, 10)], [(0, 10)]),
            (20, 30): MRCAintervals([(20, 30)], [(20, 30)]),
        }
        mrcas[1] = {(10, 20): MRCAintervals([(10, 20)], [(10, 20)])}
        mrcas._plot(highlight_position=5)

    def test_gig_plot_with_size(self, all_sv_types_no_re_gig):
        mrcas = all_sv_types_no_re_gig.tables.find_mrca_regions(11, 9)
        fig, ax = plt.subplots(1, figsize=(15, 5))
        mrcas._plot(highlight_position=110, ax=ax)
        # plt.savefig("test_gig_plot.png")  # uncomment to save a plot for inspection

    def test_random_match_pos_plain_tuple_keys(self):
        # The keys are documented as (left, right, chromosome) tuples, so plain tuples work
        mrcas = gigl.tables.MRCAdict({2: {(0, 10, 0): gigl.tables.MRCAdict.MRCAintervals([(0, 10, 0)], [(20, 30, 1)])}})
        match = mrcas.random_match_pos(np.random.default_rng(1))
        assert match.u.chromosome == 0
        assert match.v.chromosome == 1
        assert match.v.position - match.u.position == 20
        assert not match.opposite_orientations

    def test_random_match_pos_chromosomes(self):
        # u (node 0) and v (node 1) inherit from node 2: u on chromosome 1, and v
        # on chromosome 0, inverted
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1)
        tables.iedges.add_row(0, 2, child_right=10, child_chromosome=1, parent_chromosome=1)
        tables.iedges.add_row(
            1, 2, child_left=0, child_right=10, parent_left=10, parent_right=0, child_chromosome=0, parent_chromosome=1
        )
        tables.sort()
        mrcas = tables.graph().find_mrca_regions(0, 1, u_chromosomes=[1], v_chromosomes=[0])
        rng = np.random.default_rng(1)
        for _ in range(10):
            match = mrcas.random_match_pos(rng)
            assert match.u.chromosome == 1
            assert match.v.chromosome == 0
            assert match.opposite_orientations
            assert match.u.position + match.v.position == 9


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
        # S now has no iedges, so its sequence length (taken from iedges) is 0
        assert new_gig.sequence_length(0, chromosome=0) == 0
        assert new_gig.haplotype(0, right=10) == "ACGCGCGTAC"
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

    def test_find_mrca_regions(self):
        tables = self.tables_with_new_rows()
        mrcas = tables.find_mrca_regions(1, 2)
        assert list(mrcas.keys()) == [0]
        assert mrcas[0] == {gigl.Segment(0, 10, 0): ([gigl.Segment(0, 10, 0)], [gigl.Segment(0, 10, 0)])}

    def test_refuses_nodes_with_unvalidated_rows(self):
        tables = self.tables_with_new_rows()
        with pytest.raises(ValueError, match="node 3 include rows that have not been validated"):
            tables.find_mrca_regions(3, 2)
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
        assert list(tables.find_mrca_regions(1, 2).keys()) == [0]
        tables.nodes.clear()
        for t in (0, 1, 1, 0):
            tables.nodes.add_row(t)  # now the root is younger than its children
        with pytest.raises(ValueError, match="not older than child"):
            tables.find_mrca_regions(1, 2)
