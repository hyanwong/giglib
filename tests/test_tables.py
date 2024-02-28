import GeneticInheritanceGraphLibrary as gigl
import msprime
import numpy as np
import pytest


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
        assert np.array_equal(tables.nodes.time, [0, 0, 0, 1, 2])
        assert np.array_equal(tables.nodes.flags, [1, 1, 1, 0, 0])
        assert np.array_equal(tables.iedges.parent, [4, 3, 3, 4, 4])
        assert np.array_equal(tables.iedges.child, [3, 0, 0, 1, 2])
        assert np.array_equal(tables.iedges.child_left, [0, 0, 3, 0, 0])


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
