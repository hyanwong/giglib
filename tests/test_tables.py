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


class TestExtractColumn:
    # Test extraction of columns from a gig

    def test_incorrect_column_error(self, trivial_gig):
        with pytest.raises(AttributeError):
            trivial_gig.nodes.foo

    def test_column_from_empty_table(self, trivial_gig):
        assert len(trivial_gig.individuals.parents) == 0

    def test_extracted_columns(self, trivial_gig):
        assert np.array_equal(trivial_gig.nodes.time, [0, 0, 0, 1, 2])
        assert np.array_equal(trivial_gig.nodes.flags, [1, 1, 1, 0, 0])
        assert np.array_equal(trivial_gig.intervals.parent, [4, 3, 3, 4, 4])
        assert np.array_equal(trivial_gig.intervals.child_left, [0, 3, 3, 0, 0])
