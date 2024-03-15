import GeneticInheritanceGraphLibrary as gigl
import msprime
import numpy as np
import pytest
import tskit
from GeneticInheritanceGraphLibrary.constants import Const
from GeneticInheritanceGraphLibrary.constants import ValidFlags
from matplotlib import pyplot as plt


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
        tables = gigl.Tables.from_tree_sequence(simple_ts, timedelta=time)
        assert tables.nodes[0].time == simple_ts.node(0).time + time

    def test_mutations_not_implemented_error(self, simple_ts_with_mutations):
        ts_tables = simple_ts_with_mutations.tables
        assert ts_tables.mutations.num_rows > 0
        assert ts_tables.sites.num_rows > 0
        with pytest.raises(NotImplementedError):
            gigl.Tables.from_tree_sequence(simple_ts_with_mutations)

    def test_populations_not_implemented_error(self, ts_with_multiple_pops):
        # No need to test for migrations, since they necessarily
        # have multiple populations
        ts_tables = ts_with_multiple_pops.tables
        assert ts_tables.populations.num_rows > 1
        with pytest.raises(NotImplementedError):
            gigl.Tables.from_tree_sequence(ts_with_multiple_pops)

    def test_noninteger_positions(self):
        bad_ts = msprime.simulate(10, recombination_rate=10, random_seed=1)
        with pytest.raises(ValueError, match="an integer"):
            gigl.Tables.from_tree_sequence(bad_ts)


class TestFreeze:
    @pytest.mark.parametrize(
        "TableClass, params",
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
        with pytest.raises(AttributeError):
            table.add_row(**params)
        # Try replacing the data directly
        with pytest.raises(AttributeError, match="frozen"):
            table._data = []
        with pytest.raises(AttributeError, match="frozen"):
            table.clear()
        unfrozen = table.copy()
        unfrozen.add_row(**params)
        assert len(unfrozen) == 2
        unfrozen._data = []
        assert len(unfrozen) == 0
        unfrozen.add_row(**params)
        assert len(unfrozen) == 1
        unfrozen.clear()

    def test_freeze(self, trivial_gig):
        tables = trivial_gig.tables
        with pytest.raises(AttributeError):
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

    def test_iedges_cache(self, trivial_gig):
        tables = trivial_gig.tables
        assert tables.iedges.flags == ValidFlags.GIG
        sample = tables.samples()[0]
        chrom = 0
        tables_copy = tables.copy()
        assert tables_copy.iedges == tables.iedges
        # Warning: this is a hack to test the cache, and will invalidate it
        tables_copy.iedges._id_range_for_child[sample][chrom][0] += 1
        assert tables_copy.iedges != tables.iedges
        tables_copy.iedges._id_range_for_child[sample][chrom][0] -= 1  # Restore
        assert tables_copy.iedges == tables.iedges
        tables_copy.iedges.flags = ValidFlags.NONE
        assert tables_copy.iedges != tables.iedges


class TestExtractColumn:
    # Test extraction of columns from a gig
    def test_invalid_column_name(self, trivial_gig):
        tables = trivial_gig.tables
        with pytest.raises(AttributeError):
            tables.iedges.foobar

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
        assert np.array_equal(tables.samples(), simple_ts.samples())

    def test_sort(self):
        tables = gigl.Tables()
        tables.nodes.add_row(2)
        tables.nodes.add_row(1)
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(5, 0, 0, 5, parent=0, child=2)
        tables.iedges.add_row(0, 5, 0, 5, parent=1, child=3)
        tables.iedges.add_row(0, 5, 0, 5, parent=0, child=1)  # Out of order
        tables.sort()
        assert tables.iedges[0].parent == 0
        assert tables.iedges[1].parent == 0
        assert tables.iedges[2].parent == 1
        assert tables.iedges[0].child == 1
        assert tables.iedges[1].child == 2
        assert tables.iedges[2].child == 3

    def test_change_times(self, trivial_gig):
        tables = trivial_gig.tables.copy()
        times = tables.nodes.time
        tables.change_times(timedelta=1.5)
        assert np.isclose(tables.nodes.time, times + 1.5).all()


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
        tables.iedges.add_row(0, 1, 1, 0, child=0, parent=u)
        assert tables.iedges.parent_left[0] == 1
        assert tables.iedges.child_left[0] == 0
        assert tables.iedges[0].child_span == 1
        assert tables.iedges[0].parent_span == -1

    def test_append_bad_coord_type(self):
        tables = gigl.Tables()
        tables.nodes.add_row(flags=gigl.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(flags=gigl.NODE_IS_SAMPLE, time=0)
        with pytest.raises(TypeError, match="Could not convert"):
            tables.iedges.add_int_row(0, 1, None, 1, child=0, parent=1)

    def test_novalidate_add_row(self):
        tables = gigl.Tables()
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=0)
        assert tables.iedges.flags == ValidFlags.GIG
        tables.iedges.add_row(0, 1, 0, 1, child=0, parent=1)
        assert tables.iedges.flags != ValidFlags.GIG

    def test_validate_add_row(self):
        tables = gigl.Tables()
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=0)
        assert tables.iedges.flags == ValidFlags.GIG
        params = {"child": 1, "parent": 0, "validate": ValidFlags.IEDGES_ALL}
        tables.add_iedge_row(0, 1, 0, 1, **params)
        assert tables.iedges.flags == ValidFlags.GIG
        # Make a valid addition but simply assume that it's valid
        tables.add_iedge_row(1, 2, 1, 2, **params, skip_validate=True)
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

    def test_bad_child_iterator(self, all_sv_types_2re_gig):
        tables = all_sv_types_2re_gig.tables.copy()
        # Add a valid edge but don't check it
        tables.iedges.add_row(1000, 1001, 1000, 1001, child=1, parent=0)
        with pytest.raises(ValueError, match="Cannot use this method"):
            next(tables.iedges.ids_for_child(10))

    def test_good_child_iterator(self, all_sv_types_2re_gig):
        flags = (
            ValidFlags.IEDGES_FOR_CHILD_ADJACENT
            | ValidFlags.IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC
        )
        tables = all_sv_types_2re_gig.tables.copy()
        # Add a valid edge and check it
        last_child = tables.iedges[-1].child
        tables.iedges.add_row(
            1000,
            1001,
            1000,
            1001,
            child=last_child,
            parent=0,
            validate=flags,
        )
        # Can iterate even if not guaranteed in order
        _ = tables.iedges.ids_for_child(10)

    def test_unverified_child_iterator(self, all_sv_types_2re_gig):
        tables = all_sv_types_2re_gig.tables.copy()
        # Add a valid edge but don't check it
        last_child = tables.iedges[-1].child
        tables.iedges.add_row(1000, 1001, 1000, 1001, child=last_child, parent=0)
        with pytest.raises(ValueError, match="Cannot use this method"):
            _ = tables.iedges.ids_for_child(10)

    # TODO - add gigl.LEAFWARDS test
    @pytest.mark.parametrize("direction", [Const.ROOTWARDS])
    def test_notransform_interval(self, direction):
        flags = ValidFlags.IEDGES_COMBO_STANDALONE
        iedges = gigl.tables.IEdgeTable()
        iedges.add_row(10, 20, 10, 20, child=0, parent=1, validate=flags)
        assert iedges.transform_interval(0, (12, 17), direction) == (12, 17)
        assert iedges.transform_interval(0, (10, 20), direction) == (10, 20)
        nd_type = "child" if direction == Const.ROOTWARDS else "parent"
        with pytest.raises(ValueError, match=f"not in {nd_type} interval"):
            iedges.transform_interval(0, (9, 12), direction)
        with pytest.raises(ValueError, match=f"not in {nd_type} interval"):
            iedges.transform_interval(0, (17, 21), direction)

    def test_transform_interval_linear(self):
        flags = ValidFlags.IEDGES_COMBO_STANDALONE
        iedges = gigl.tables.IEdgeTable()
        iedges.add_row(0, 10, 10, 20, child=0, parent=1, validate=flags)
        assert iedges.transform_interval(0, (0, 10), Const.ROOTWARDS) == (10, 20)
        assert iedges.transform_interval(0, (2, 7), Const.ROOTWARDS) == (12, 17)
        with pytest.raises(ValueError, match="not in child interval"):
            iedges.transform_interval(0, (-1, 10), Const.ROOTWARDS)
        with pytest.raises(ValueError, match="not in child interval"):
            iedges.transform_interval(0, (1, 11), Const.ROOTWARDS)

    def test_transform_interval_inversion(self):
        flags = ValidFlags.IEDGES_COMBO_STANDALONE
        iedges = gigl.tables.IEdgeTable()
        iedges.add_row(10, 20, 30, 20, child=0, parent=1, validate=flags)
        assert iedges.transform_interval(0, (10, 20), Const.ROOTWARDS) == (30, 20)
        assert iedges.transform_interval(0, (11, 19), Const.ROOTWARDS) == (29, 21)
        with pytest.raises(ValueError, match="not in child interval"):
            iedges.transform_interval(0, (5, 21), Const.ROOTWARDS)

    def test_clear(self, trivial_gig):
        tables = trivial_gig.tables.copy()
        tables.iedges.add_row(10, 20, 10, 20, child=0, parent=4)
        assert tables.iedges.flags != ValidFlags.GIG
        assert len(tables.iedges._id_range_for_child) > 0
        tables.iedges.clear()
        assert tables.iedges.flags == ValidFlags.GIG
        assert len(tables.iedges._id_range_for_child) == 0

    def test_max_child_pos(self, trivial_gig):
        tables = trivial_gig.tables.copy()
        youngest_child = len(tables.nodes) - 1
        tables.add_iedge_row(
            10, 20, 10, 20, child=youngest_child, parent=0, validate=ValidFlags.GIG
        )
        assert tables.iedges.max_child_pos(youngest_child, chromosome=0) == 20

    def test_bad_max_child_pos(self, trivial_gig):
        tables = trivial_gig.tables.copy()
        # Don't validate
        tables.iedges.add_row(10, 20, 10, 20, child=0, parent=4)
        with pytest.raises(ValueError, match="Cannot use this method"):
            tables.iedges.max_child_pos(0, chromosome=0)


class TestIedgesValidation:
    """
    Tests for the validation of iedge tables during iedges.add_row()
    """

    @pytest.mark.parametrize(
        "flag, skip",
        [
            (f, s)
            for f in ValidFlags.iedges_combo_standalone_iter()
            for s in (True, False)
        ],
    )
    def test_flags(self, flag, skip):
        tables = gigl.Tables()
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=0)
        assert flag in tables.iedges.flags
        assert tables.iedges.flags != flag
        tables.add_iedge_row(
            0, 1, 0, 1, child=1, parent=0, validate=flag, skip_validate=skip
        )
        assert tables.iedges.flags == flag

    def test_bad_flags(self):
        flags = ValidFlags.IEDGES_COMBO_NODE_TABLE
        tables = gigl.Tables()
        for f in ValidFlags:
            if f != ValidFlags.NONE and f in flags:
                with pytest.raises(ValueError, match="involving the node table"):
                    tables.iedges.add_row(0, 1, 0, 1, child=1, parent=0, validate=f)

    def test_add_iedge_row_fail_integers(self):
        flags = ValidFlags.IEDGES_INTEGERS
        tables = gigl.Tables()
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=0)
        with pytest.raises(ValueError, match="Expected an integer"):
            tables.add_iedge_row(0, 1, 0, 1.2, child=1, parent=0, validate=flags)

    def test_add_iedge_row_fail_child_nonoverlapping(self):
        flags = ValidFlags.IEDGES_FOR_CHILD_NONOVERLAPPING
        tables = gigl.Tables()
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=0)
        tables.add_iedge_row(
            0,
            2,
            0,
            2,
            child=0,
            parent=1,
            validate=flags | ValidFlags.IEDGES_WITHIN_CHILD_SORTED,
        )
        with pytest.raises(ValueError, match="iedges overlap"):
            tables.add_iedge_row(1, 2, 2, 3, child=0, parent=1, validate=flags)

    def test_add_iedge_row_fail_child_adjacent(self):
        flags = ValidFlags.IEDGES_FOR_CHILD_ADJACENT
        tables = gigl.Tables()
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=0)
        tables.nodes.add_row(time=0)
        tables.add_iedge_row(0, 1, 0, 1, child=1, parent=0, validate=flags)
        tables.add_iedge_row(0, 1, 0, 1, child=2, parent=0, validate=flags)
        with pytest.raises(ValueError, match="non-adjacent"):
            tables.add_iedge_row(1, 2, 2, 3, child=1, parent=0, validate=flags)

    def test_add_iedge_row_fail_child_primary_order_chr_asc(self):
        flags = ValidFlags.IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC
        tables = gigl.Tables()
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=0)
        tables.nodes.add_row(time=0)
        tables.add_iedge_row(
            0, 1, 0, 1, child=1, parent=0, child_chromosome=1, validate=flags
        )
        with pytest.raises(ValueError, match="chromosome IDs out of order"):
            tables.add_iedge_row(
                0, 1, 0, 1, child=1, parent=0, child_chromosome=0, validate=flags
            )
        tables.add_iedge_row(
            0, 1, 0, 1, child=1, parent=0, child_chromosome=3, validate=flags
        )
        tables.add_iedge_row(
            0, 1, 0, 1, child=2, parent=0, child_chromosome=0, validate=flags
        )

    def test_add_iedge_row_fail_child_secondary_order_left_asc(self):
        flags = ValidFlags.IEDGES_FOR_CHILD_SECONDARY_ORDER_LEFT_ASC
        tables = gigl.Tables()
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=0)
        tables.add_iedge_row(1, 2, 1, 2, child=1, parent=0, validate=flags)
        with pytest.raises(ValueError, match="break edge_left ordering"):
            tables.add_iedge_row(0, 1, 2, 3, child=1, parent=0, validate=flags)

    def test_add_iedge_row_fail_intervals(self):
        flags = ValidFlags.IEDGES_INTERVALS
        tables = gigl.Tables()
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=0)
        with pytest.raises(ValueError, match="absolute spans differ"):
            tables.add_iedge_row(0, 1, 0, 2, child=1, parent=0, validate=flags)

    def test_add_iedge_row_fail_child_interval_positive(self):
        flags = ValidFlags.IEDGES_CHILD_INTERVAL_POSITIVE
        tables = gigl.Tables()
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=0)
        with pytest.raises(ValueError, match="child left must be < child right"):
            tables.add_iedge_row(2, 1, 1, 2, child=1, parent=0, validate=flags)

    def test_add_iedge_row_fail_no_nodes(self):
        flags = ValidFlags.IEDGES_COMBO_NODE_TABLE
        tables = gigl.Tables()
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=0)
        tables.add_iedge_row(0, 1, 0, 1, child=2, parent=0)
        with pytest.raises(ValueError, match="does not correspond to a node"):
            tables.add_iedge_row(0, 1, 0, 1, child=2, parent=0, validate=flags)

    def test_add_iedge_row_fail_parent_older_than_child(self):
        flags = ValidFlags.IEDGES_PARENT_OLDER_THAN_CHILD
        tables = gigl.Tables()
        tables.nodes.add_row(time=0)
        tables.nodes.add_row(time=1)
        with pytest.raises(ValueError, match="not less than parent time"):
            tables.add_iedge_row(0, 1, 0, 1, child=1, parent=0, validate=flags)

    def test_add_iedge_row_fail_primary_order_child_time(self):
        flags = ValidFlags.IEDGES_PRIMARY_ORDER_CHILD_TIME_DESC
        tables = gigl.Tables()
        tables.nodes.add_row(time=2)
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=0)
        tables.add_iedge_row(0, 1, 0, 1, child=2, parent=1, validate=flags)
        with pytest.raises(
            ValueError, match="older than the previous iedge child time"
        ):
            tables.add_iedge_row(0, 1, 0, 1, child=1, parent=0, validate=flags)

    def test_add_iedge_row_fail_secondary_order_child_id(self):
        flags = ValidFlags.IEDGES_SECONDARY_ORDER_CHILD_ID_ASC
        tables = gigl.Tables()
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=0)
        tables.nodes.add_row(time=0)
        tables.add_iedge_row(0, 1, 0, 1, child=2, parent=0, validate=flags)
        with pytest.raises(ValueError, match="lower child ID"):
            tables.add_iedge_row(0, 1, 0, 1, child=1, parent=0, validate=flags)

    def test_parent_child_at_inf(self):
        flags = ValidFlags.IEDGES_PARENT_OLDER_THAN_CHILD
        tables = gigl.Tables()
        tables.nodes.add_row(time=np.inf)
        tables.nodes.add_row(time=np.inf)
        tables.nodes.add_row(time=0)
        with pytest.raises(ValueError, match="not less than parent time"):
            tables.add_iedge_row(0, 1, 0, 1, child=0, parent=1, validate=flags)
        tables.add_iedge_row(0, 1, 0, 1, child=2, parent=1, validate=flags)


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

    def test_add_rows(self):
        # test if e.g. we can broadcast
        nodes = gigl.tables.NodeTable()
        nodes.add_row(time=123)
        ret = nodes.add_rows(np.arange(10).reshape(2, 5), flags=0, individual=gigl.NULL)
        assert len(nodes.time.shape) == 1
        assert np.all(ret == np.arange(10).reshape(2, 5) + 1)
        assert nodes.time.shape[0] == 11
        assert np.all(nodes.time == np.insert(np.arange(10), 0, 123))
        for i, n in enumerate(nodes):
            if i == 0:
                assert n.time == 123
            else:
                assert n.time == i - 1


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
            tables.nodes.add_row(time=node[0], flags=node[1])
        tables.iedges.add_row(
            child_left=iedge[0],
            child_right=iedge[1],
            parent_left=iedge[2],
            parent_right=iedge[3],
            child=iedge[4],
            parent=iedge[5],
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
            assert (
                "10 rows skipped (GeneticInheritanceGraphLibrary.set_print_options)"
                in html.split("</tr>")[21]
            )
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
        gig = gigl.from_tree_sequence(simple_ts)
        tables = gig.tables
        assert getattr(tables.iedges, name).dtype == np.int64
        suffix = name.split("_")[-1]
        ts_order = gig.iedge_map_sorted_by_parent
        assert np.all(
            getattr(tables.iedges, name)[ts_order]
            == getattr(simple_ts, "edges_" + suffix)
        )


class TestIndividualAttributes:
    def test_asdict(self, simple_ts):
        tables = gigl.Tables.from_tree_sequence(simple_ts)
        for i, ind in enumerate(tables.individuals):
            assert np.array_equal(
                ind.asdict()["parents"], simple_ts.individual(i).parents
            )


class TestNodeAttributes:
    def test_flags(self, simple_ts):
        tables = gigl.Tables.from_tree_sequence(simple_ts)
        assert np.all(tables.nodes.flags == simple_ts.nodes_flags)

    def test_child(self, simple_ts):
        gig = gigl.from_tree_sequence(simple_ts)
        tables = gig.tables
        ts_order = gig.iedge_map_sorted_by_parent
        assert np.all(tables.iedges.child[ts_order] == simple_ts.edges_child)


class TestFindMrcas:
    def test_find_mrca_single_tree(self):
        span = 123
        comb_ts = tskit.Tree.generate_comb(5, span=span).tree_sequence
        # make a gig with many unary nodes above node 0
        comb_ts = comb_ts.simplify([0, 4], keep_unary=True)
        gig = gigl.from_tree_sequence(comb_ts)
        full_span = (0, span)
        shared_regions = gig.tables.find_mrca_regions(0, 1)
        assert len(shared_regions) == 1
        assert comb_ts.first().root in shared_regions
        mrca_intervals = shared_regions[comb_ts.first().root]
        assert len(mrca_intervals) == 1
        mrca, uv_interval_lists = mrca_intervals.popitem()
        assert mrca == full_span
        assert len(uv_interval_lists) == 2
        for interval_list in uv_interval_lists:
            assert len(interval_list) == 1
            assert full_span in interval_list

    def test_find_mrca_2_trees(self, degree2_2_tip_ts):
        num_trees = 2
        assert degree2_2_tip_ts.num_trees == num_trees
        gig = gigl.from_tree_sequence(degree2_2_tip_ts)
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
        gig = gigl.from_tree_sequence(degree2_2_tip_ts)
        assert gig.sequence_length(0) == degree2_2_tip_ts.sequence_length
        assert gig.sequence_length(1) == degree2_2_tip_ts.sequence_length
        cutoff = (T[[tree0.root, tree1.root]]).mean()
        shared_regions = gig.tables.find_mrca_regions(0, 1, cutoff)
        assert len(shared_regions) == 1
        used_tree = tree0 if T[tree0.root] < T[tree1.root] else tree1
        unused_tree = tree1 if T[tree0.root] < T[tree1.root] else tree0
        assert used_tree.root in shared_regions
        assert unused_tree.root not in shared_regions

    def test_simple_non_sv(self, simple_ts):
        assert simple_ts.num_trees > 1
        assert simple_ts.num_samples >= 2
        max_trees = 0
        gig = gigl.from_tree_sequence(simple_ts)
        for u in range(len(gig.samples)):
            for v in range(u + 1, len(gig.samples)):
                mrcas = gig.tables.find_mrca_regions(u, v)
                equiv_ts = simple_ts.simplify([u, v], filter_nodes=False)
                max_trees = max(max_trees, equiv_ts.num_trees)
                for tree in equiv_ts.trees():
                    interval = (int(tree.interval.left), int(tree.interval.right))
                    mrca = tree.get_mrca(u, v)
                    assert interval in mrcas[mrca]
                    u_equivalent, v_equivalent = mrcas[mrca][interval]
                    assert len(u_equivalent) == 1  # No duplications
                    assert len(v_equivalent) == 1  # No duplications
                    assert interval in u_equivalent
                    assert interval in v_equivalent
        assert max_trees > 2  # at least some cases with 3 or more mrcas

    def test_double_inversion(self, double_inversion_gig):
        assert double_inversion_gig.num_samples == 2
        iedges = list(double_inversion_gig.iedges_for_child(0))
        assert len(iedges) == 1
        assert iedges[0].is_inversion()
        mrcas = double_inversion_gig.tables.find_mrca_regions(0, 1)
        assert len(mrcas) == 1
        assert 3 in mrcas
        mrca = mrcas[3]
        assert (0, 100) in mrca
        u, v = mrca[(0, 100)]
        assert u == [(0, 100)]
        assert v == [(0, 100)]

    @pytest.mark.parametrize("sample_resolve", [True, False])
    def test_extended_inversion(self, extended_inversion_gig, sample_resolve):
        gig = extended_inversion_gig
        if sample_resolve:
            gig = gig.sample_resolve()
        assert gig.min_position(0) == 20
        assert gig.max_position(0) == 155
        assert gig.min_position(1) == 0
        assert gig.max_position(1) == 100
        mrcas = gig.tables.find_mrca_regions(0, 1)
        assert len(mrcas) == 1
        assert 3 in mrcas
        mrca = mrcas[3]
        assert (15, 100) in mrca  # the inverson only leaves 15..100 shared
        u, v = mrca[(15, 100)]
        assert len(v) == 1
        assert (15, 100) in v
        assert len(u) == 1
        assert (155, 70) in u  # inverted region is span 85 from 155 leftward in sample

    def test_inverted_duplicate(self, inverted_duplicate_gig):
        assert inverted_duplicate_gig.num_samples == 2
        mrcas = inverted_duplicate_gig.tables.find_mrca_regions(0, 1)
        assert len(mrcas) == 1
        assert 3 in mrcas
        mrca = mrcas[3]
        assert (10, 15) in mrca  # the duplicated+inverted section
        u, v = mrca[(10, 15)]
        assert set(u) == {(0, 5), (15, 10)}
        assert v == [(0, 5)]

        assert (15, 20) in mrca  # only the inverted section
        u, v = mrca[(15, 20)]
        assert u == [(10, 5)]
        assert v == [(5, 10)]

    def test_inverted_duplicate_with_missing(self, inverted_duplicate_with_missing_gig):
        assert inverted_duplicate_with_missing_gig.num_samples == 2
        mrcas = inverted_duplicate_with_missing_gig.tables.find_mrca_regions(0, 1)
        assert len(mrcas) == 1
        assert 3 in mrcas
        mrca = mrcas[3]
        assert (10, 15) in mrca  # the duplicated+inverted section
        u, v = mrca[(10, 15)]
        assert set(u) == {(0, 5), (35, 30)}
        assert v == [(0, 5)]

        assert (15, 20) in mrca  # only the inverted section
        u, v = mrca[(15, 20)]
        assert u == [(30, 25)]
        assert v == [(5, 10)]

    def test_no_recomb_sv_dup_del(self, all_sv_types_no_re_gig):
        assert all_sv_types_no_re_gig.num_samples >= 2
        sample_u = 8
        sample_v = 10
        mrcas = all_sv_types_no_re_gig.tables.find_mrca_regions(sample_u, sample_v)
        assert len(mrcas) == 1
        assert 1 in mrcas

        # (0, 50) should be unchanged
        assert (0, 50) in mrcas[1]
        u, v = mrcas[1][(0, 50)]
        assert len(u) == len(v) == 1
        assert (0, 50) in u
        assert (0, 50) in v

        # (150, 50) should be duplicated
        assert (150, 200) in mrcas[1]
        u, v = mrcas[1][(150, 200)]
        assert len(u) == 1
        assert (50, 100) in u  # deletion
        assert len(v) == 2  # duplication
        assert (150, 200) in v  # duplication
        assert (250, 300) in v  # duplication

        # (50, 150) should be deleted, with no MRCA
        assert (50, 150) not in mrcas[1]
        assert len(mrcas[1]) == 2

    def test_no_recomb_sv_dup_inv(self, all_sv_types_no_re_gig):
        assert all_sv_types_no_re_gig.num_samples >= 2
        sample_u = 10
        sample_v = 12
        mrcas = all_sv_types_no_re_gig.tables.find_mrca_regions(sample_u, sample_v)
        assert len(mrcas) == 1
        assert 0 in mrcas
        mrca = mrcas[0]
        # "normal" region
        assert (0, 20) in mrca
        assert mrca[(0, 20)] == ([(0, 20)], [(0, 20)])

        # start of inverted region
        assert (20, 100) in mrca
        assert mrca[(20, 100)] == ([(20, 100)], [(120, 40)])

        # duplicated inverted region
        assert (100, 120) in mrca
        assert set(mrca[(100, 120)][0]) == {(100, 120), (200, 220)}
        assert mrca[(100, 120)][1] == [(40, 20)]

        # duplicated non-inverted region
        assert (120, 200) in mrca
        assert set(mrca[(120, 200)][0]) == {(120, 200), (220, 300)}
        assert mrca[(120, 200)][1] == [(120, 200)]

    def test_random_match_pos(self, simple_ts):
        rng = np.random.default_rng(1)
        ts = simple_ts.keep_intervals([(0, 2)]).trim()
        gig = gigl.from_tree_sequence(ts)
        assert gig.samples[0] == 0
        assert gig.samples[1] == 1
        mrcas = gig.tables.find_mrca_regions(0, 1)
        all_breaks = set()
        for _ in range(20):
            # in 20 replicates we should definitely have both 0 and 1
            breaks = mrcas.random_match_pos(rng)
            assert len(breaks) == 3
            assert not breaks.opposite_orientations
            assert breaks.u == breaks.v
            all_breaks.add(breaks.u)
        assert all_breaks == {0, 1}


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
