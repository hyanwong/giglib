import GeneticInheritanceGraphLibrary as gigl
import numpy as np
import pytest


class TestFunctions:
    # Test high level functions
    def test_simple_from_tree_sequence(self, simple_ts):
        assert simple_ts.num_trees > 1
        gig = gigl.from_tree_sequence(simple_ts)
        assert np.all(gig.samples == gig.tables.samples())


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
        assert gig.parent_range.shape == (len(gig.nodes), 2)
        assert np.all(gig.parent_range == gigl.NULL)
        assert gig.child_range.shape == (len(gig.nodes), 2)
        assert np.all(gig.child_range == gigl.NULL)
        assert gig.iedge_map_sorted_by_child.shape == (0,)

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
        tables.nodes.add_row(1)
        tables.nodes.add_row(2)
        tables.iedges.add_row(0, 1, 0, 1, child=0, parent=1)
        tables.iedges.add_row(1, 2, 0, 1, child=0, parent=2)
        tables.iedges.add_row(2, 3, 0, 1, child=0, parent=1)
        with pytest.raises(ValueError, match="not sorted by parent ID"):
            tables.graph()

    def test_time_unsorted(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1)
        tables.nodes.add_row(2)
        tables.iedges.add_row(0, 1, 1, 0, child=1, parent=2)
        tables.iedges.add_row(0, 1, 1, 0, child=0, parent=1)
        with pytest.raises(ValueError, match="not sorted by parent time"):
            tables.graph()

    def test_duplicate_parents(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1)
        tables.nodes.add_row(2)
        tables.iedges.add_row(0, 2, 0, 2, child=0, parent=1)
        tables.iedges.add_row(0, 1, 0, 1, child=0, parent=1)
        with pytest.raises(ValueError, match="multiple or duplicate parents"):
            tables.graph()

    def test_multiple_parents(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1)
        tables.nodes.add_row(2)
        tables.iedges.add_row(0, 2, 0, 2, child=0, parent=1)
        tables.iedges.add_row(0, 1, 0, 1, child=0, parent=2)
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

    def test_sequence_length_root(self, all_sv_types_gig):
        # root will have no upward edges
        root = len(all_sv_types_gig.nodes) - 1
        assert all_sv_types_gig.sequence_length(root) == 200

    def test_no_sequence_length(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        gig = tables.graph()
        for u in gig.samples:
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
        assert len(new_gig.iedges) - len(all_sv_types_gig.iedges) == 1
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
        tables.iedges.add_row(1, 2, 2, 1, child=0, parent=1)
        tables.iedges.add_row(0, 100, 0, 100, child=1, parent=2)
        gig = tables.graph()
        assert gig.iedges[1].parent == 2
        assert gig.iedges[1].parent_left == 0
        assert gig.iedges[1].parent_right == 100
        new_gig = gig.sample_resolve()
        assert len(new_gig.iedges) == 2
        assert new_gig.iedges[1].parent == 2
        assert new_gig.iedges[1].parent_left == 1
        assert new_gig.iedges[1].parent_right == 2

    def test_sample_resolve_double_inversion(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1)
        tables.nodes.add_row(2)
        tables.iedges.add_row(2, 5, 5, 2, child=0, parent=1)
        tables.iedges.add_row(0, 100, 100, 0, child=1, parent=2)
        gig = tables.graph()
        new_gig = gig.sample_resolve()
        assert len(new_gig.iedges) == 2
        assert new_gig.iedges[1].parent == 2
        assert new_gig.iedges[1].parent_left == 98
        assert new_gig.iedges[1].parent_right == 95
        # TODO - test a transformation of e.g. 3 in the original sample


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

    @pytest.mark.parametrize("direction", (gigl.ROOTWARDS, gigl.LEAFWARDS))
    def test_notransform_position(self, direction):
        ie = gigl.graph.IEdge(10, 20, 10, 20, child=0, parent=1, id=0)
        assert ie.transform_position(10, direction) == 10
        assert ie.transform_position(12, direction) == 12
        assert ie.transform_position(17, direction) == 17
        nd_type = "child" if direction == gigl.ROOTWARDS else "parent"
        with pytest.raises(ValueError, match=f"not in {nd_type} interval"):
            ie.transform_position(9, direction)
        with pytest.raises(ValueError, match=f"not in {nd_type} interval"):
            ie.transform_position(20, direction)

    def test_transform_position_linear(self):
        ie = gigl.graph.IEdge(0, 10, 10, 20, child=0, parent=1, id=0)
        assert ie.transform_position(0, gigl.ROOTWARDS) == 10
        assert ie.transform_position(2, gigl.ROOTWARDS) == 12
        assert ie.transform_position(7, gigl.ROOTWARDS) == 17
        with pytest.raises(ValueError, match="not in child interval"):
            ie.transform_position(-1, gigl.ROOTWARDS)
        with pytest.raises(ValueError, match="not in child interval"):
            ie.transform_position(10, gigl.ROOTWARDS)

        assert ie.transform_position(10, gigl.LEAFWARDS) == 0
        assert ie.transform_position(12, gigl.LEAFWARDS) == 2
        assert ie.transform_position(17, gigl.LEAFWARDS) == 7
        with pytest.raises(ValueError, match="not in parent interval"):
            ie.transform_position(9, gigl.LEAFWARDS)
        with pytest.raises(ValueError, match="not in parent interval"):
            ie.transform_position(20, gigl.LEAFWARDS)

    @pytest.mark.parametrize("direction", (gigl.ROOTWARDS, gigl.LEAFWARDS))
    def test_transform_position_inversion(self, direction):
        ie = gigl.graph.IEdge(10, 20, 20, 10, child=0, parent=1, id=0)
        assert ie.transform_position(10, direction) == 19
        assert ie.transform_position(11, direction) == 18
        assert ie.transform_position(12, direction) == 17
        assert ie.transform_position(17, direction) == 12
        assert ie.transform_position(18, direction) == 11
        assert ie.transform_position(19, direction) == 10
        nd_type = "child" if direction == gigl.ROOTWARDS else "parent"
        with pytest.raises(ValueError, match=f"not in {nd_type} interval"):
            ie.transform_position(20, direction)

    def test_transform_position_moved_inversion(self):
        ie = gigl.graph.IEdge(10, 20, 30, 20, child=0, parent=1, id=0)
        assert ie.transform_position(10, gigl.ROOTWARDS) == 29
        assert ie.transform_position(11, gigl.ROOTWARDS) == 28
        assert ie.transform_position(12, gigl.ROOTWARDS) == 27
        assert ie.transform_position(17, gigl.ROOTWARDS) == 22
        assert ie.transform_position(18, gigl.ROOTWARDS) == 21
        assert ie.transform_position(19, gigl.ROOTWARDS) == 20

    # TODO - add gigl.LEAFWARDS test
    @pytest.mark.parametrize("direction", [gigl.ROOTWARDS])
    def test_notransform_interval(self, direction):
        ie = gigl.graph.IEdge(10, 20, 10, 20, child=0, parent=1, id=0)
        assert ie.transform_interval((12, 17), direction) == (12, 17)
        assert ie.transform_interval((10, 20), direction) == (10, 20)
        nd_type = "child" if direction == gigl.ROOTWARDS else "parent"
        with pytest.raises(ValueError, match=f"not in {nd_type} interval"):
            ie.transform_interval((9, 12), direction)
        with pytest.raises(ValueError, match=f"not in {nd_type} interval"):
            ie.transform_interval((17, 21), direction)

    def test_transform_interval_linear(self):
        ie = gigl.graph.IEdge(0, 10, 10, 20, child=0, parent=1, id=0)
        assert ie.transform_interval((0, 10), gigl.ROOTWARDS) == (10, 20)
        assert ie.transform_interval((2, 7), gigl.ROOTWARDS) == (12, 17)
        with pytest.raises(ValueError, match="not in child interval"):
            ie.transform_interval((-1, 10), gigl.ROOTWARDS)
        with pytest.raises(ValueError, match="not in child interval"):
            ie.transform_interval((1, 11), gigl.ROOTWARDS)

    def test_transform_interval_inversion(self):
        ie = gigl.graph.IEdge(10, 20, 30, 20, child=0, parent=1, id=0)
        assert ie.transform_interval((10, 20), gigl.ROOTWARDS) == (30, 20)
        assert ie.transform_interval((11, 19), gigl.ROOTWARDS) == (29, 21)
        with pytest.raises(ValueError, match="not in child interval"):
            ie.transform_interval((5, 21), gigl.ROOTWARDS)


class TestFindMrcas:
    def test_simple_non_sv(self, simple_ts):
        assert simple_ts.num_trees > 1
        assert simple_ts.num_samples >= 2
        gig = gigl.from_tree_sequence(simple_ts)
        mrcas = gig.find_mrca_regions(0, 1)
        for tree in simple_ts.simplify([0, 1], filter_nodes=False).trees():
            interval = (int(tree.interval.left), int(tree.interval.right))
            mrca = tree.get_mrca(0, 1)
            assert interval in mrcas[mrca]
            u_equivalent, v_equivalent = mrcas[mrca][interval]
            assert len(u_equivalent) == 1  # No duplications
            assert len(v_equivalent) == 1  # No duplications
            assert u_equivalent[0] == interval
            assert v_equivalent[0] == interval

    def test_no_recomb_sv_dup_del(self, all_sv_types_gig):
        assert all_sv_types_gig.num_samples >= 2
        sample_u = 0
        sample_v = 2
        mrcas = all_sv_types_gig.find_mrca_regions(sample_u, sample_v)
        assert len(mrcas) == 1
        assert 11 in mrcas

        # (0, 50) should be unchanged
        assert (0, 50) in mrcas[11]
        u, v = mrcas[11][(0, 50)]
        assert len(u) == len(v) == 1
        assert u[0] == (0, 50)
        assert v[0] == (0, 50)

        # (150, 50) should be duplicated
        assert (150, 200) in mrcas[11]
        u, v = mrcas[11][(150, 200)]
        assert len(u) == 2  # duplication
        assert (150, 200) in u  # duplication
        assert (250, 300) in u  # duplication
        assert len(v) == 1
        assert v[0] == (50, 100)  # deletion

        # (50, 150) should be deleted, with no MRCA
        assert (50, 150) not in mrcas[11]
        assert len(mrcas[11]) == 2

    def test_no_recomb_sv_dup_inv(self, all_sv_types_gig):
        assert all_sv_types_gig.num_samples >= 2
        sample_u = 0
        sample_v = 4
        mrcas = all_sv_types_gig.find_mrca_regions(sample_u, sample_v)
        assert len(mrcas) == 1
        assert 12 in mrcas
        for k, v in mrcas[12].items():
            print(k, ":", v)
        # TODO - test inversions
