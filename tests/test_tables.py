import GeneticInheritanceGraph as gig
import msprime
import numpy as np
import pytest


class TestCreation:
    # Test creation of a set of GiGtables
    def test_simple_from_tree_sequence(self, simple_ts):
        assert simple_ts.num_trees > 1
        tables = gig.Tables.from_tree_sequence(simple_ts)
        assert len(tables.nodes) == simple_ts.num_nodes
        assert len(tables.iedges) == simple_ts.num_edges
        # test conversion from float to int for genomic coordinates
        assert simple_ts.edges_left.dtype == np.float64
        assert tables.iedges.parent_left.dtype == np.int64
        assert tables.iedges.child_left.dtype == np.int64
        assert simple_ts.edges_right.dtype == np.float64
        assert tables.iedges.parent_right.dtype == np.int64
        assert tables.iedges.child_right.dtype == np.int64

    @pytest.mark.parametrize("time", [0, 1])
    def test_simple_from_tree_sequence_with_timedelta(self, simple_ts, time):
        assert simple_ts.num_trees > 1
        tables = gig.Tables.from_tree_sequence(simple_ts, timedelta=time)
        assert tables.nodes[0].time == simple_ts.node(0).time + time

    def test_mutations_not_implemented_error(self, simple_ts_with_mutations):
        ts_tables = simple_ts_with_mutations.tables
        assert ts_tables.mutations.num_rows > 0
        assert ts_tables.sites.num_rows > 0
        with pytest.raises(NotImplementedError):
            gig.Tables.from_tree_sequence(simple_ts_with_mutations)

    def test_populations_not_implemented_error(self, ts_with_multiple_pops):
        # No need to test for migrations, since they necessarily
        # have multiple populations
        ts_tables = ts_with_multiple_pops.tables
        assert ts_tables.populations.num_rows > 1
        with pytest.raises(NotImplementedError):
            gig.Tables.from_tree_sequence(ts_with_multiple_pops)

    def test_noninteger_positions(self):
        bad_ts = msprime.simulate(10, recombination_rate=10, random_seed=1)
        with pytest.raises(ValueError, match="not an integer"):
            gig.Tables.from_tree_sequence(bad_ts)


class TestExtractColumn:
    # Test extraction of columns from a gig
    def test_invalid_column_name(self, trivial_gig):
        with pytest.raises(AttributeError):
            trivial_gig.iedges.foobar

    def test_column_from_empty_table(self, trivial_gig):
        assert len(trivial_gig.individuals.parents) == 0

    def test_extracted_columns(self, trivial_gig):
        assert np.array_equal(trivial_gig.nodes.time, [0, 0, 0, 1, 2])
        assert np.array_equal(trivial_gig.nodes.flags, [1, 1, 1, 0, 0])
        assert np.array_equal(trivial_gig.iedges.parent, [4, 3, 3, 4, 4])
        assert np.array_equal(trivial_gig.iedges.child_left, [0, 3, 3, 0, 0])


class TestMethods:
    def test_samples(self, simple_ts):
        tables = gig.Tables.from_tree_sequence(simple_ts)
        assert np.array_equal(tables.samples(), simple_ts.samples())


class TestBaseTable:
    # Test various basic table methods
    def test_append_row_values(self, trivial_gig):
        assert len(trivial_gig.nodes) == 5
        trivial_gig.nodes.append({"time": 3, "flags": 0})
        assert len(trivial_gig.nodes) == 6
        assert trivial_gig.nodes.time[-1] == 3
        assert trivial_gig.nodes.flags[-1] == 0

    def test_iteration(self, trivial_gig):
        assert len(trivial_gig.nodes) > 0
        for row in trivial_gig.nodes:
            assert isinstance(row, gig.tables.NodeTableRow)
        assert len(trivial_gig.iedges) > 0
        for row in trivial_gig.iedges:
            assert isinstance(row, gig.tables.IEdgeTableRow)


class TestIEdgeTable:
    def test_append_integer_coords(trivial_gig):
        tables = gig.Tables()
        u = tables.nodes.add_row(flags=gig.NODE_IS_SAMPLE, time=0)
        tables.iedges.add_row(0, u, 0, 1, 1, 0)
        assert tables.iedges.parent_left[0] == 0
        assert tables.iedges.child_left[0] == 1
        assert tables.iedges[0].child_span == -1
        assert tables.iedges[0].parent_span == 1


class TestStringRepresentations:
    # Test string and html representations of tables
    def test_hardcoded_tskit_is_skipped(self, gig_from_degree2_ts):
        output = str(gig_from_degree2_ts)
        assert "tskit" not in output

    def test_identifiable_values_from_str(self):
        tables = gig.Tables()
        nodes = [(0, 3.1451), (1, 7.4234)]
        iedge = [1, 0, 0.9876, 6.7890, 0.9876, 6.7890]
        for node in nodes:
            tables.nodes.add_row(time=node[0], flags=node[1])
        tables.iedges.add_row(
            parent=iedge[0],
            child=iedge[1],
            child_left=iedge[2],
            child_right=iedge[3],
            parent_left=iedge[4],
            parent_right=iedge[5],
        )
        for node in nodes:
            assert str(node[0]) in str(tables)
            assert str(node[1]) in str(tables)
        for value in iedge:
            assert str(value) in str(tables)

    @pytest.mark.parametrize("num_rows", [0, 10, 40, 50])
    def test_repr_html(self, num_rows):
        # Based on the corresponding test code in tskit
        nodes = gig.Tables().nodes
        for _ in range(num_rows):
            nodes.append({"time": 0, "flags": 0})
        html = nodes._repr_html_()
        if num_rows == 50:
            assert len(html.splitlines()) == num_rows + 11
            assert (
                "10 rows skipped (GeneticInheritanceGraph.set_print_options)"
                in html.split("</tr>")[21]
            )
        else:
            assert len(html.splitlines()) == num_rows + 20


class TestIEdgeAttributes:
    def test_non_ts_attributes(self, simple_ts):
        tables = gig.Tables.from_tree_sequence(simple_ts)
        assert tables.iedges.edge.dtype == np.int64
        assert np.all(tables.iedges.edge == gig.NULL)

    @pytest.mark.parametrize(
        "name",
        ["parent", "child", "parent_left", "parent_right", "child_left", "child_right"],
    )
    def test_ts_attributes(self, simple_ts, name):
        tables = gig.Tables.from_tree_sequence(simple_ts)
        assert getattr(tables.iedges, name).dtype == np.int64
        suffix = name.split("_")[-1]
        assert np.all(
            getattr(tables.iedges, name) == getattr(simple_ts, "edges_" + suffix)
        )


class TestNodeAttributes:
    def test_flags(self, simple_ts):
        tables = gig.Tables.from_tree_sequence(simple_ts)
        assert np.all(tables.nodes.flags == simple_ts.nodes_flags)

    def test_child(self, simple_ts):
        tables = gig.Tables.from_tree_sequence(simple_ts)
        assert np.all(tables.iedges.child == simple_ts.edges_child)
