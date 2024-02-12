import GeneticInheritanceGraphLibrary as gigl
import numpy as np
import pytest
import tskit


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
    def test_find_mrca_single_tree(self):
        span = 123
        comb_ts = tskit.Tree.generate_comb(5, span=span).tree_sequence
        # make a gig with many unary nodes above node 0
        comb_ts = comb_ts.simplify([0, 4], keep_unary=True)
        gig = gigl.from_tree_sequence(comb_ts)
        full_span = (0, span)
        shared_regions = gig.find_mrca_regions(0, 1)
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
        shared_regions = gig.find_mrca_regions(0, 1)
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
        shared_regions = gig.find_mrca_regions(0, 1, cutoff)
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
                mrcas = gig.find_mrca_regions(u, v)
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
        mrcas = double_inversion_gig.find_mrca_regions(0, 1)
        assert len(mrcas) == 1
        assert 3 in mrcas
        mrca = mrcas[3]
        assert (0, 100) in mrca
        u, v = mrca[(0, 100)]
        assert u == {(0, 100)}
        assert v == {(0, 100)}

    @pytest.mark.parametrize("sample_resolve", [True, False])
    def test_extended_inversion(self, extended_inversion_gig, sample_resolve):
        gig = extended_inversion_gig
        if sample_resolve:
            gig = gig.sample_resolve()
        assert gig.min_position(0) == 20
        assert gig.max_position(0) == 155
        assert gig.min_position(1) == 0
        assert gig.max_position(1) == 100
        mrcas = gig.find_mrca_regions(0, 1)
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
        mrcas = inverted_duplicate_gig.find_mrca_regions(0, 1)
        assert len(mrcas) == 1
        assert 3 in mrcas
        mrca = mrcas[3]
        assert (10, 15) in mrca  # the duplicated+inverted section
        u, v = mrca[(10, 15)]
        assert u == {(0, 5), (15, 10)}
        assert v == {(0, 5)}

        assert (15, 20) in mrca  # only the inverted section
        u, v = mrca[(15, 20)]
        assert u == {(10, 5)}
        assert v == {(5, 10)}

    def test_inverted_duplicate_with_missing(self, inverted_duplicate_with_missing_gig):
        assert inverted_duplicate_with_missing_gig.num_samples == 2
        mrcas = inverted_duplicate_with_missing_gig.find_mrca_regions(0, 1)
        assert len(mrcas) == 1
        assert 3 in mrcas
        mrca = mrcas[3]
        assert (10, 15) in mrca  # the duplicated+inverted section
        u, v = mrca[(10, 15)]
        assert u == {(0, 5), (35, 30)}
        assert v == {(0, 5)}

        assert (15, 20) in mrca  # only the inverted section
        u, v = mrca[(15, 20)]
        assert u == {(30, 25)}
        assert v == {(5, 10)}

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
        assert (0, 50) in u
        assert (0, 50) in v

        # (150, 50) should be duplicated
        assert (150, 200) in mrcas[11]
        u, v = mrcas[11][(150, 200)]
        assert len(u) == 2  # duplication
        assert (150, 200) in u  # duplication
        assert (250, 300) in u  # duplication
        assert len(v) == 1
        assert (50, 100) in v  # deletion

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
        mrca = mrcas[12]
        # "normal" region
        assert (0, 80) in mrca
        assert mrca[(0, 80)] == ({(0, 80)}, {(0, 80)})

        # start of inverted region
        assert (80, 100) in mrca
        assert mrca[(80, 100)] == ({(80, 100)}, {(160, 140)})

        # duplicated inverted region
        assert (100, 160) in mrca
        assert mrca[(100, 160)] == ({(100, 160), (200, 260)}, {(140, 80)})

        # duplicated non-inverted region
        assert (160, 200) in mrca
        assert mrca[(160, 200)] == ({(160, 200), (260, 300)}, {(160, 200)})
