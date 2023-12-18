import GeneticInheritanceGraph as gigl
import numpy as np
import pytest


class TestFunctions:
    # Test high level functions
    def test_simple_from_tree_sequence(self, simple_ts):
        assert simple_ts.num_trees > 1
        gig = gigl.from_tree_sequence(simple_ts)
        assert np.all(gig.samples() == gig.tables.samples())


class TestConstructor:
    def test_from_empty_tables(self):
        tables = gigl.Tables()
        gig = tables.graph()
        assert gig.num_nodes == 0
        assert gig.num_iedges == 0

    def test_from_tables(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1, flags=0)
        tables.iedges.add_row(
            parent=1,
            child=0,
            child_left=0,
            child_right=1,
            parent_left=1,
            parent_right=0,
        )
        gig = tables.graph()
        assert gig.num_nodes == 2
        assert gig.num_iedges == 1
        assert gig.num_samples == 1

    def test_no_edges(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1, flags=0)
        tables.nodes.add_row(1, flags=0)
        gig = tables.graph()
        assert gig.num_nodes == 3
        assert gig.num_iedges == 0
        assert gig.num_samples == 1
        # Cached values should be defined but empty
        assert gig.parent_range.shape == (gig.num_nodes, 2)
        assert np.all(gig.parent_range == gigl.NULL)
        assert gig.child_range.shape == (gig.num_nodes, 2)
        assert np.all(gig.child_range == gigl.NULL)
        assert gig.iedge_map_sorted_by_child.shape == (0,)

    def test_from_bad(self):
        with pytest.raises(
            ValueError, match="must be a GeneticInheritanceGraph.Tables"
        ):
            gigl.Graph("not a tree sequence")

    def test_bad_parent_child_time(self, simple_ts):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(
            parent=0,
            child=0,
            child_left=0,
            child_right=1,
            parent_left=1,
            parent_right=0,
        )
        with pytest.raises(ValueError, match="0 not older than child 0"):
            tables.graph()

    def test_negative_parent_child_id(self, simple_ts):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(
            parent=1,
            child=-1,
            child_left=0,
            child_right=1,
            parent_left=1,
            parent_right=0,
        )
        with pytest.raises(ValueError, match="negative"):
            tables.graph()

    def test_bigger_parent_child_id(self, simple_ts):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(
            parent=10,
            child=0,
            child_left=0,
            child_right=1,
            parent_left=1,
            parent_right=0,
        )
        with pytest.raises(ValueError, match="ID >= num nodes"):
            tables.graph()

    def test_unmatched_parent_child_spans(self, simple_ts):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(
            parent=1,
            child=0,
            child_left=0,
            child_right=2,
            parent_left=1,
            parent_right=0,
        )
        with pytest.raises(ValueError, match="spans"):
            tables.graph()

    def test_nonmatching_spans(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1, flags=0)
        tables.iedges.add_row(
            parent=1,
            child=0,
            child_left=0,
            child_right=0,
            parent_left=1,
            parent_right=0,
        )
        with pytest.raises(ValueError, match="spans"):
            tables.graph()

    def test_id_unsorted(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1)
        tables.nodes.add_row(2)
        tables.iedges.add_row(
            parent=1,
            child=0,
            child_left=0,
            child_right=1,
            parent_left=1,
            parent_right=0,
        )
        tables.iedges.add_row(
            parent=2,
            child=0,
            child_left=0,
            child_right=1,
            parent_left=1,
            parent_right=0,
        )
        tables.iedges.add_row(
            parent=1,
            child=0,
            child_left=1,
            child_right=2,
            parent_left=1,
            parent_right=2,
        )
        with pytest.raises(ValueError, match="not sorted by parent ID"):
            tables.graph()

    def test_time_unsorted(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1)
        tables.nodes.add_row(2)
        tables.iedges.add_row(
            parent=2,
            child=1,
            child_left=0,
            child_right=1,
            parent_left=1,
            parent_right=0,
        )
        tables.iedges.add_row(
            parent=1,
            child=0,
            child_left=0,
            child_right=1,
            parent_left=1,
            parent_right=0,
        )
        tables.iedges.add_row(
            parent=1,
            child=0,
            child_left=1,
            child_right=2,
            parent_left=1,
            parent_right=2,
        )
        with pytest.raises(ValueError, match="not sorted by parent time"):
            tables.graph()

    def test_duplicate_parents(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1)
        tables.nodes.add_row(2)
        tables.iedges.add_row(
            parent=1,
            child=0,
            child_left=0,
            child_right=1,
            parent_left=1,
            parent_right=0,
        )
        tables.iedges.add_row(
            parent=1,
            child=0,
            child_left=0,
            child_right=1,
            parent_left=1,
            parent_right=0,
        )
        with pytest.raises(ValueError, match="multiple or duplicate parents"):
            tables.graph()

    def test_multiple_parents(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1)
        tables.nodes.add_row(2)
        tables.iedges.add_row(
            parent=1,
            child=0,
            child_left=0,
            child_right=1,
            parent_left=1,
            parent_right=0,
        )
        tables.iedges.add_row(
            parent=2,
            child=0,
            child_left=0,
            child_right=1,
            parent_left=1,
            parent_right=0,
        )
        with pytest.raises(ValueError, match="multiple or duplicate parents"):
            tables.graph()


class TestMethods:
    def test_edge_iterator(self, simple_ts):
        gig = gigl.from_tree_sequence(simple_ts)
        assert gig.num_iedges == simple_ts.num_edges
        assert len(gig.iedges) == simple_ts.num_edges  # __len__ should work
        i = 0
        for iedge, edge in zip(gig.iedges, simple_ts.edges()):
            assert isinstance(iedge, gigl.graph.IEdge)
            assert iedge.id == edge.id == i
            i += 1
        assert i == simple_ts.num_edges

    def test_iedges_for_child(self, simple_ts):
        gig = gigl.from_tree_sequence(simple_ts)
        edges = set()
        for u in range(gig.num_nodes):
            for iedge in gig.iedges_for_child(u):
                assert isinstance(iedge, gigl.graph.IEdge)
                assert iedge.child == u
                edges.add(iedge.id)
        assert len(edges) == simple_ts.num_edges

    def test_sequence_length(self, simple_ts):
        gig = gigl.from_tree_sequence(simple_ts)
        for u in gig.samples():
            assert gig.sequence_length(u) == simple_ts.sequence_length

    def test_sequence_length_root(self, all_sv_types_gig):
        # root will have no upward edges
        root = all_sv_types_gig.num_nodes - 1
        assert all_sv_types_gig.sequence_length(root) == 200

    def test_no_sequence_length(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        gig = tables.graph()
        for u in gig.samples():
            assert gig.sequence_length(u) == 0


class TestSampleResolving:
    """
    Sample resolving is complicated enough that we want a whole class to test
    """

    def test_simple_sample_resolve(self, simple_ts):
        gig = gigl.from_tree_sequence(simple_ts.simplify())
        new_gig = gig.sample_resolve()  # Shouldn't make any changes
        assert gig.tables == new_gig.tables

    def test_all_svs_sample_resolve(self, all_sv_types_gig):
        """
        The all_sv_types_gig is not sample resolved, so we should see changes
        (in particular, 10->11 should be split into two edges)
        """
        new_gig = all_sv_types_gig.sample_resolve()  # Shouldn't make any changes
        assert new_gig.num_iedges - all_sv_types_gig.num_iedges == 1
        new_edges = iter(new_gig.tables.iedges)
        for iedge_row in all_sv_types_gig.tables.iedges:
            if iedge_row.parent == 11 and iedge_row.child == 10:
                assert iedge_row.parent_left == iedge_row.child_left == 0
                assert iedge_row.parent_right == iedge_row.child_right == 200
                new_iedge = next(new_edges)
                assert new_iedge.parent_left == new_iedge.child_left == 0
                assert new_iedge.parent_right == new_iedge.child_right == 50
                new_iedge = next(new_edges)
                assert new_iedge.parent_left == new_iedge.child_left == 150
                assert new_iedge.parent_right == new_iedge.child_right == 200
            else:
                assert iedge_row == next(new_edges)


class TestIEdge:
    """
    Test the wrapper around a single iedge
    """

    @pytest.mark.parametrize(
        "name",
        ["parent", "child", "parent_left", "parent_right", "child_left", "child_right"],
    )
    def test_edge_accessors(self, simple_ts, name):
        gig = gigl.from_tree_sequence(simple_ts)
        suffix = name.split("_")[-1]

        ie = gig.iedges[0]
        assert getattr(ie, name) == getattr(simple_ts.edge(0), suffix)

    def test_span(self, all_sv_types_gig):
        for ie in all_sv_types_gig.iedges:
            assert ie.span == np.abs(ie.child_right - ie.child_left)
            assert ie.span == np.abs(ie.parent_right - ie.parent_left)

    def test_is_inversion(self, all_sv_types_gig):
        num_inversions = 0
        for ie in all_sv_types_gig.iedges:
            if ie.is_inversion():
                num_inversions += 1
                assert ie.parent_right - ie.parent_left < 0
        assert num_inversions == 1

    def test_notransform(self):
        ie = gigl.graph.IEdge(
            0, 1, parent_left=10, parent_right=20, child_left=10, child_right=20, id=0
        )
        assert ie.transform_to_parent(child_position=12) == 12
        assert ie.transform_to_parent(child_position=17) == 17

    def test_transform_linear(self):
        ie = gigl.graph.IEdge(
            0, 1, parent_left=10, parent_right=20, child_left=0, child_right=10, id=0
        )
        assert ie.transform_to_parent(child_position=2) == 12
        assert ie.transform_to_parent(child_position=7) == 17

    def test_transform_inversion(self):
        ie = gigl.graph.IEdge(
            0, 1, parent_left=20, parent_right=10, child_left=10, child_right=20, id=0
        )
        assert ie.transform_to_parent(child_position=10) == 19
        assert ie.transform_to_parent(child_position=11) == 18
        assert ie.transform_to_parent(child_position=12) == 17
        assert ie.transform_to_parent(child_position=17) == 12
        assert ie.transform_to_parent(child_position=18) == 11
        assert ie.transform_to_parent(child_position=19) == 10
