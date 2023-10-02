import GeneticInheritanceGraph as gig
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
