import GeneticInheritanceGraphLibrary as gigl
import numpy as np
import pytest
import tskit
from GeneticInheritanceGraphLibrary.constants import Const


class TestConstructor:
    def test_from_empty_tables(self):
        tables = gigl.Tables()
        gig = tables.graph()
        assert len(gig.nodes) == 0
        assert len(gig.iedges) == 0

    def test_from_tables(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1, flags=0)
        tables.iedges.add_row(0, 1, 1, 0, child=0, parent=1)
        gig = tables.graph()
        assert len(gig.nodes) == 2
        assert len(gig.iedges) == 1
        assert len(gig.samples) == 1

    def test_no_edges(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1, flags=0)
        tables.nodes.add_row(1, flags=0)
        gig = tables.graph()
        assert len(gig.nodes) == 3
        assert len(gig.iedges) == 0
        assert len(gig.samples) == 1
        # Cached values should be defined but empty
        assert gig.child_range.shape == (len(gig.nodes), 2)
        assert np.all(gig.child_range == gigl.NULL)
        assert gig.iedge_map_sorted_by_parent.shape == (0,)

    def test_from_bad(self):
        with pytest.raises(
            ValueError,
            match="must be a GeneticInheritanceGraphLibrary.Tables",
        ):
            gigl.Graph("not a tree sequence")

    def test_bad_parent_child_time(self, simple_ts):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(0, 1, 1, 0, child=0, parent=0)
        with pytest.raises(ValueError, match="0 not older than child 0"):
            tables.graph()

    def test_negative_parent_child_id(self, simple_ts):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(0, 1, 1, 0, child=-1, parent=1)
        with pytest.raises(ValueError, match="negative"):
            tables.graph()

    def test_bigger_parent_child_id(self, simple_ts):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(0, 1, 1, 0, child=0, parent=10)
        with pytest.raises(ValueError, match="ID >= num nodes"):
            tables.graph()

    def test_unmatched_parent_child_spans(self, simple_ts):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(0, 1, 2, 0, child=0, parent=1)
        with pytest.raises(ValueError, match="spans"):
            tables.graph()
        tables.iedges.clear()
        assert len(tables.iedges) == 0
        # test something more standard than an inversion
        tables.iedges.add_row(1, 2, 0, 2, child=0, parent=1)
        with pytest.raises(ValueError, match="spans"):
            tables.graph()

    def test_bad_spans(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1, flags=0)
        tables.iedges.add_row(0, 0, 0, 1, child=0, parent=1)
        with pytest.raises(ValueError, match="spans"):
            tables.graph()

    def test_id_unsorted(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1)
        tables.iedges.add_row(0, 1, 0, 1, child=0, parent=2)
        tables.iedges.add_row(1, 2, 0, 1, child=1, parent=2)
        tables.iedges.add_row(2, 3, 0, 1, child=0, parent=2)
        with pytest.raises(ValueError, match="not sorted by child"):
            tables.graph()

    def test_time_unsorted(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1)
        tables.nodes.add_row(2)
        tables.iedges.add_row(0, 1, 1, 0, child=0, parent=2)
        tables.iedges.add_row(0, 1, 1, 0, child=1, parent=2)
        with pytest.raises(ValueError, match="not sorted by child time, descending"):
            tables.graph()

    def test_duplicate_parents(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1)
        tables.nodes.add_row(2)
        tables.iedges.add_row(0, 2, 0, 2, child=0, parent=1)
        tables.iedges.add_row(1, 3, 1, 3, child=0, parent=1)
        with pytest.raises(ValueError, match="multiple or duplicate parents"):
            tables.graph()

    def test_multiple_parents(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1)
        tables.nodes.add_row(2)
        tables.iedges.add_row(0, 2, 0, 2, child=0, parent=1)
        tables.iedges.add_row(1, 3, 1, 3, child=0, parent=2)
        with pytest.raises(ValueError, match="multiple or duplicate parents"):
            tables.graph()


class TestMethods:
    def test_edge_iterator(self, simple_ts):
        gig = gigl.from_tree_sequence(simple_ts)
        assert len(gig.iedges) == simple_ts.num_edges
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
        for u in range(len(gig.nodes)):
            for iedge in gig.iedges_for_child(u):
                assert isinstance(iedge, gigl.graph.IEdge)
                assert iedge.child == u
                edges.add(iedge.id)
        assert len(edges) == simple_ts.num_edges

    def test_sequence_length(self, simple_ts):
        gig = gigl.from_tree_sequence(simple_ts)
        for u in gig.samples:
            assert gig.sequence_length(u) == simple_ts.sequence_length

    def test_sequence_length_root(self, all_sv_types_no_re_gig):
        # root will have no upward edges
        root = len(all_sv_types_no_re_gig.nodes) - 1
        assert all_sv_types_no_re_gig.sequence_length(root) == 200

    def test_no_sequence_length(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        gig = tables.graph()
        for u in gig.samples:
            assert gig.sequence_length(u) == 0


class TestTskit:
    """
    Methods that involve tree_sequence conversion
    """

    # Test high level functions
    def test_simple_from_tree_sequence(self, simple_ts):
        assert simple_ts.num_trees > 1
        gig = gigl.from_tree_sequence(simple_ts)
        assert np.all(gig.samples == gig.tables.samples())

    def test_to_tree_sequence(self, degree2_2_tip_ts):
        gig = gigl.from_tree_sequence(degree2_2_tip_ts)
        L = degree2_2_tip_ts.sequence_length + 100
        ts = gig.to_tree_sequence(sequence_length=L)
        assert ts.num_samples == 2
        assert ts.num_trees == 3
        assert ts.at_index(2).num_edges == 0  # empty region at end
        assert ts.sequence_length == L

    def test_to_tree_sequence_bad_length(self, degree2_2_tip_ts):
        gig = gigl.from_tree_sequence(degree2_2_tip_ts)
        L = degree2_2_tip_ts.sequence_length - 1
        with pytest.raises(tskit.LibraryError):
            gig.to_tree_sequence(sequence_length=L)

    def test_roundtrip(self, simple_ts):
        gig = gigl.from_tree_sequence(simple_ts)
        ts = gig.to_tree_sequence()
        ts.tables.assert_equals(simple_ts.tables, ignore_provenance=True)

    def test_bad_ts(self, inverted_duplicate_gig):
        with pytest.raises(ValueError, match="Cannot convert to tree sequence"):
            inverted_duplicate_gig.to_tree_sequence()


class TestSampleResolving:
    """
    Sample resolving is complicated enough that we want a whole class to test
    """

    def test_simple_sample_resolve(self, simple_ts):
        gig = gigl.from_tree_sequence(simple_ts.simplify())
        new_gig = gig.sample_resolve()  # Shouldn't make any changes
        assert gig.tables == new_gig.tables

    def test_all_svs_no_re_sample_resolve(self, all_sv_types_no_re_gig):
        """
        The all_sv_types_no_re_gig is not sample resolved, so we should see changes (in
        particular, 1->2 should be split into two edges, either side of the deletion)
        """
        new_gig = all_sv_types_no_re_gig.sample_resolve()  # Should only split one iedge
        assert len(new_gig.iedges) - len(all_sv_types_no_re_gig.iedges) == 1
        new_edges = iter(new_gig.tables.iedges)
        for iedge_row in all_sv_types_no_re_gig.tables.iedges:
            if iedge_row.parent == 1 and iedge_row.child == 2:
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

    def test_all_svs_re_sample_resolve(self, all_sv_types_re_gig):
        """
        An even more complicated gig. In particular, recombination means that some of
        the iedges leading into the RE nodes should be trimmed. We should still split
        the 1->2 iedge but also we have an internal edge from 5->7 that contains
        a missing section
        """
        new_gig = all_sv_types_re_gig.sample_resolve()
        assert len(new_gig.iedges) - len(all_sv_types_re_gig.iedges) == 2
        iedge_rows = list(all_sv_types_re_gig.iedges_for_child(2))
        assert len(iedge_rows) == 1
        assert iedge_rows[0].parent == 1
        assert iedge_rows[0].parent_left == iedge_rows[0].child_left == 0
        assert iedge_rows[0].parent_right == iedge_rows[0].child_right == 200

        iedge_rows = list(new_gig.iedges_for_child(2))
        assert len(iedge_rows) == 2
        assert iedge_rows[0].parent == 1
        assert iedge_rows[0].parent_left == iedge_rows[0].child_left == 0
        assert iedge_rows[0].parent_right == iedge_rows[0].child_right == 50
        assert iedge_rows[1].parent == 1
        assert iedge_rows[1].parent_left == iedge_rows[1].child_left == 150
        assert iedge_rows[1].parent_right == iedge_rows[1].child_right == 200

        # One of the recombination nodes
        iedge_rows = list(all_sv_types_re_gig.iedges_for_child(7))
        assert len(iedge_rows) == 1
        assert iedge_rows[0].parent == 5
        assert iedge_rows[0].parent_left == iedge_rows[0].child_left == 0
        assert iedge_rows[0].parent_right == iedge_rows[0].child_right == 300
        iedge_rows = list(new_gig.iedges_for_child(7))
        assert len(iedge_rows) == 2
        assert iedge_rows[0].parent == 5
        assert iedge_rows[0].parent_left == iedge_rows[0].child_left == 0
        assert iedge_rows[0].parent_right == iedge_rows[0].child_right == 150
        assert iedge_rows[1].parent == 5
        assert iedge_rows[1].parent_left == iedge_rows[1].child_left == 170
        assert iedge_rows[1].parent_right == iedge_rows[1].child_right == 300

    def test_sample_resolve_no_edges(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        gig = tables.graph()
        new_gig = gig.sample_resolve()
        assert len(new_gig.iedges) == 0
        assert len(new_gig.nodes) == 1

    def test_sample_resolve_no_samples(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0)
        gig = tables.graph()
        new_gig = gig.sample_resolve()
        assert len(new_gig.iedges) == 0
        # Sample resolution does not delete unconnected nodes
        assert len(new_gig.nodes) == 1

    def test_sample_resolve_inversion(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1)
        tables.nodes.add_row(2)
        tables.iedges.add_row(0, 100, 0, 100, child=1, parent=2)
        tables.iedges.add_row(1, 2, 2, 1, child=0, parent=1)
        gig = tables.graph()
        assert gig.iedges[0].parent == 2
        assert gig.iedges[0].parent_left == 0
        assert gig.iedges[0].parent_right == 100
        new_gig = gig.sample_resolve()
        assert len(new_gig.iedges) == 2
        assert new_gig.iedges[0].parent == 2
        assert new_gig.iedges[0].parent_left == 1
        assert new_gig.iedges[0].parent_right == 2

    def test_sample_resolve_double_inversion(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1)
        tables.nodes.add_row(2)
        tables.iedges.add_row(0, 100, 100, 0, child=1, parent=2)
        tables.iedges.add_row(2, 5, 5, 2, child=0, parent=1)
        gig = tables.graph()
        new_gig = gig.sample_resolve()
        assert len(new_gig.iedges) == 2
        assert new_gig.iedges[0].parent == 2
        assert new_gig.iedges[0].parent_left == 98
        assert new_gig.iedges[0].parent_right == 95
        # TODO - test a transformation of e.g. 3 in the original sample

    def test_extended_inversion(self, extended_inversion_gig):
        assert extended_inversion_gig.max_position(0) == 155
        assert extended_inversion_gig.min_position(0) == 20

        assert extended_inversion_gig.nodes[0].is_sample()
        iedges = list(extended_inversion_gig.iedges_for_child(0))
        assert len(iedges) == 1
        iedges = list(extended_inversion_gig.iedges_for_child(iedges[0].parent))
        assert len(iedges) == 1
        ie = iedges[0]
        assert ie.child_left == 10
        assert ie.child_right == 160
        assert ie.is_simple_inversion()
        gig = extended_inversion_gig.sample_resolve()

        # Now check sample resolving has correctly trimmed the inversion edge
        iedges = list(gig.iedges_for_child(0))
        assert len(iedges) == 1
        iedges = list(gig.iedges_for_child(iedges[0].parent))
        assert len(iedges) == 1
        ie = iedges[0]
        assert ie.is_inversion()
        assert ie.child_left == 20
        assert ie.child_right == 155
        assert ie.parent_left == 150
        assert ie.parent_right == 15


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

        ie = gig.iedges[gig.iedge_map_sorted_by_parent[0]]  # tskit order
        assert getattr(ie, name) == getattr(simple_ts.edge(0), suffix)

    def test_span(self, all_sv_types_no_re_gig):
        for ie in all_sv_types_no_re_gig.iedges:
            assert ie.span == np.abs(ie.child_right - ie.child_left)
            assert ie.span == np.abs(ie.parent_right - ie.parent_left)

    def test_is_inversion(self, all_sv_types_no_re_gig):
        num_inversions = 0
        for ie in all_sv_types_no_re_gig.iedges:
            if ie.is_inversion():
                num_inversions += 1
                assert ie.parent_right - ie.parent_left < 0
        assert num_inversions == 1

    @pytest.mark.parametrize("direction", (Const.ROOTWARDS, Const.LEAFWARDS))
    def test_notransform_position(self, direction):
        ie = gigl.graph.IEdge(10, 20, 10, 20, child=0, parent=1, id=0)
        assert ie.transform_position(10, direction) == 10
        assert ie.transform_position(12, direction) == 12
        assert ie.transform_position(17, direction) == 17
        nd_type = "child" if direction == Const.ROOTWARDS else "parent"
        with pytest.raises(ValueError, match=f"not in {nd_type} interval"):
            ie.transform_position(9, direction)
        with pytest.raises(ValueError, match=f"not in {nd_type} interval"):
            ie.transform_position(20, direction)

    def test_transform_position_linear(self):
        ie = gigl.graph.IEdge(0, 10, 10, 20, child=0, parent=1, id=0)
        assert ie.transform_position(0, Const.ROOTWARDS) == 10
        assert ie.transform_position(2, Const.ROOTWARDS) == 12
        assert ie.transform_position(7, Const.ROOTWARDS) == 17
        with pytest.raises(ValueError, match="not in child interval"):
            ie.transform_position(-1, Const.ROOTWARDS)
        with pytest.raises(ValueError, match="not in child interval"):
            ie.transform_position(10, Const.ROOTWARDS)

        assert ie.transform_position(10, Const.LEAFWARDS) == 0
        assert ie.transform_position(12, Const.LEAFWARDS) == 2
        assert ie.transform_position(17, Const.LEAFWARDS) == 7
        with pytest.raises(ValueError, match="not in parent interval"):
            ie.transform_position(9, Const.LEAFWARDS)
        with pytest.raises(ValueError, match="not in parent interval"):
            ie.transform_position(20, Const.LEAFWARDS)

    @pytest.mark.parametrize("direction", (Const.ROOTWARDS, Const.LEAFWARDS))
    def test_transform_position_inversion(self, direction):
        ie = gigl.graph.IEdge(10, 20, 20, 10, child=0, parent=1, id=0)
        assert ie.transform_position(10, direction) == 19
        assert ie.transform_position(11, direction) == 18
        assert ie.transform_position(12, direction) == 17
        assert ie.transform_position(17, direction) == 12
        assert ie.transform_position(18, direction) == 11
        assert ie.transform_position(19, direction) == 10
        nd_type = "child" if direction == Const.ROOTWARDS else "parent"
        with pytest.raises(ValueError, match=f"not in {nd_type} interval"):
            ie.transform_position(20, direction)

    def test_transform_position_moved_inversion(self):
        ie = gigl.graph.IEdge(10, 20, 30, 20, child=0, parent=1, id=0)
        assert ie.transform_position(10, Const.ROOTWARDS) == 29
        assert ie.transform_position(11, Const.ROOTWARDS) == 28
        assert ie.transform_position(12, Const.ROOTWARDS) == 27
        assert ie.transform_position(17, Const.ROOTWARDS) == 22
        assert ie.transform_position(18, Const.ROOTWARDS) == 21
        assert ie.transform_position(19, Const.ROOTWARDS) == 20
