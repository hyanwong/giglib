import GeneticInheritanceGraph as gigl
import numpy as np
import portion as P
import pytest
import tskit


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


class TestMRCARegions:
    """
    Test the ability to locate closes matching regions between two sequences
    """

    def test_binary_interval_stack(self):
        """
        Test the BinaryIntervalStack class
        """
        BIStack = gigl.graph.BinaryIntervalStack
        comb_ts = tskit.Tree.generate_comb(5).tree_sequence
        # make a gig with many unary nodes above node 0
        comb_ts = comb_ts.simplify([0, 4], keep_unary=True)
        gig = gigl.from_tree_sequence(comb_ts)
        stack = BIStack(gig, 0, 1)
        for i in gig.iedges:
            stack.add(
                0 if i.child == 0 else 1,
                i.parent,
                P.closedopen(0, np.inf),
                gigl.graph.OriginalInterval(0),
            )

        for i, (idx, node_id, ivls) in enumerate(stack.reduce()):
            assert node_id == i  # nodes are in time order, and all are visited
            if node_id != comb_ts.first().root and node_id != 0:
                assert idx == BIStack.B
                assert ivls.domain() == P.closedopen(0, np.inf)
        # The last item popped off should be the shared root
        assert idx == BIStack.SHARED
        assert i == gig.num_iedges

    def test_binary_interval_stack_equal_times(self):
        """
        When we add intervals to one of the stacks, we need to make sure that they
        are keep on the stack in the correct reversed order (first by node time
        descending, then by ID descending)
        """
        BIStack = gigl.graph.BinaryIntervalStack
        balanced_ts = tskit.Tree.generate_balanced(8).tree_sequence
        gig = gigl.from_tree_sequence(balanced_ts)
        stack = BIStack(gig, 0, 7)
        t = gig.tables.nodes.time
        assert t[8] == t[9] == t[11] == t[12]
        assert t[10] > t[8]
        # Although they aren't in the inheritance path, add other intermediate nodes in
        # an arbitrary order, just as a hack to check the order is correctly maintained
        stack.add(
            BIStack.B, 12, P.closedopen(0, np.inf), gigl.graph.OriginalInterval(0)
        )
        stack.add(
            BIStack.B, 10, P.closedopen(0, np.inf), gigl.graph.OriginalInterval(0)
        )
        stack.add(BIStack.B, 9, P.closedopen(0, np.inf), gigl.graph.OriginalInterval(0))
        stack.add(
            BIStack.B, 11, P.closedopen(0, np.inf), gigl.graph.OriginalInterval(0)
        )
        stack.add(BIStack.B, 8, P.closedopen(0, np.inf), gigl.graph.OriginalInterval(0))
        expected_B_order = [10, 12, 11, 9, 8, 7]
        for expected, node_id in zip(expected_B_order, stack.stackB):
            assert expected == node_id

    def test_find_mrca_single_tree(self):
        span = 123
        comb_ts = tskit.Tree.generate_comb(5, span=span).tree_sequence
        # make a gig with many unary nodes above node 0
        comb_ts = comb_ts.simplify([0, 4], keep_unary=True)
        gig = gigl.from_tree_sequence(comb_ts)
        shared_regions = gig.find_mrca_regions(0, 1)
        assert len(shared_regions) == 1
        assert comb_ts.first().root in shared_regions
        val = shared_regions[comb_ts.first().root]
        assert len(val) == 2
        for i in [0, 1]:
            assert len(val[i]) == 1
            interval, origin = val[i].popitem()
            assert interval == P.closedopen(0, span)
            assert origin.inverted is False

    def test_find_mrca_2_trees(self, degree2_2_tip_ts):
        gig = gigl.from_tree_sequence(degree2_2_tip_ts)
        shared_regions = gig.find_mrca_regions(0, 1)
        assert len(shared_regions) == 2
        assert gig.tables.nodes.time[2] in shared_regions
        val = shared_regions[gig.tables.nodes.time[2]]
        assert len(val) == 2
        for i in [0, 1]:
            assert len(val[i]) == 1
            interval, origin = val[i].popitem()
            assert interval == P.closedopen(0, gig.sequence_length(0))
            assert origin.inverted is False

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
