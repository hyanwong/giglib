import GeneticInheritanceGraph as gig
import numpy as np
import pytest


class TestCreation:
    # Test creation of a set of GiGtables
    def test_simple_from_tree_sequence(self, simple_ts):
        assert simple_ts.num_trees > 1
        g = gig.TableGroup.from_tree_sequence(simple_ts)
        assert len(g.nodes) == simple_ts.num_nodes
        assert len(g.intervals) == simple_ts.num_edges

    @pytest.mark.parametrize("time", [0, 1])
    def test_simple_from_tree_sequence_with_timedelta(self, simple_ts, time):
        assert simple_ts.num_trees > 1
        g = gig.TableGroup.from_tree_sequence(simple_ts, timedelta=time)
        assert g.nodes[0].time == simple_ts.node(0).time + time

    def test_mutations_not_implemented_error(self, simple_ts_with_mutations):
        tables = simple_ts_with_mutations.tables
        assert tables.mutations.num_rows > 0
        assert tables.sites.num_rows > 0
        with pytest.raises(NotImplementedError):
            gig.TableGroup.from_tree_sequence(simple_ts_with_mutations)

    def test_populations_not_implemented_error(self, ts_with_multiple_pops):
        # No need to test for migrations, since they necessarily
        # have multiple populations
        tables = ts_with_multiple_pops.tables
        assert tables.populations.num_rows > 1
        with pytest.raises(NotImplementedError):
            gig.TableGroup.from_tree_sequence(ts_with_multiple_pops)


class TestExtractColumn:
    # Test extraction of columns from a gig
    def test_invalid_column_name(self, trivial_gig):
        with pytest.raises(AttributeError):
            trivial_gig.intervals.foobar

    def test_column_from_empty_table(self, trivial_gig):
        assert len(trivial_gig.individuals.parents) == 0

    def test_extracted_columns(self, trivial_gig):
        assert np.array_equal(trivial_gig.nodes.time, [0, 0, 0, 1, 2])
        assert np.array_equal(trivial_gig.nodes.flags, [1, 1, 1, 0, 0])
        assert np.array_equal(trivial_gig.intervals.parent, [4, 3, 3, 4, 4])
        assert np.array_equal(trivial_gig.intervals.child_left, [0, 3, 3, 0, 0])


class TestBaseTable:
    # Test various basic table methods
    def test_append_row_values(self, trivial_gig):
        assert len(trivial_gig.nodes) == 5
        trivial_gig.nodes.append({"time": 3, "flags": 0})
        assert len(trivial_gig.nodes) == 6
        assert trivial_gig.nodes.time[-1] == 3
        assert trivial_gig.nodes.flags[-1] == 0


class TestStringRepresentations:
    # Test string and html representations of tables
    def test_hardcoded_tskit_is_skipped(self, gig_from_degree2_ts):
        output = str(gig_from_degree2_ts)
        assert "tskit" not in output

    def test_identifiable_values_from_str(self):
        table_group = gig.TableGroup()
        nodes = [(0, 3.1451), (1, 7.4234)]
        interval = [1, 0, 0.9876, 6.7890, 0.9876, 6.7890]
        for node in nodes:
            table_group.nodes.add_row(time=node[0], flags=node[1])
        table_group.intervals.add_row(
            parent=interval[0],
            child=interval[1],
            child_left=interval[2],
            child_right=interval[3],
            parent_left=interval[4],
            parent_right=interval[5],
        )
        for node in nodes:
            assert str(node[0]) in str(table_group)
            assert str(node[1]) in str(table_group)
        for value in interval:
            assert str(value) in str(table_group)

    @pytest.mark.parametrize("num_rows", [0, 10, 40, 50])
    def test_repr_html(self, num_rows):
        # Based on the corresponding test code in tskit
        nodes = gig.TableGroup().nodes
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
