import numpy as np
import pytest
import tskit

import giglib as gigl
from giglib.constants import Const


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
        assert len(gig.tables.sample_nodes()) == 1

    def test_no_edges(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1, flags=0)
        tables.nodes.add_row(1, flags=0)
        gig = tables.graph()
        assert len(gig.nodes) == 3
        assert len(gig.iedges) == 0
        assert len(gig.tables.sample_nodes()) == 1
        # Cached values should be defined but empty
        assert len(gig._id_range_for_parent) == 0
        assert gig._iedge_map_sorted_by_parent.shape == (0,)

    def test_from_bad(self):
        with pytest.raises(
            ValueError,
            match="must be a giglib.Tables",
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
        gig = gigl.Graph.from_tree_sequence(simple_ts)
        assert len(gig.iedges) == simple_ts.num_edges
        assert len(gig.iedges) == simple_ts.num_edges  # __len__ should work
        i = 0
        for iedge, edge in zip(gig.iedges, simple_ts.edges()):
            assert isinstance(iedge, gigl.graph.IEdge)
            assert iedge.id == edge.id == i
            i += 1
        assert i == simple_ts.num_edges

    def test_iedges_for_child(self, simple_ts):
        gig = gigl.Graph.from_tree_sequence(simple_ts)
        edges = set()
        for u in range(len(gig.nodes)):
            for iedge in gig.iedges_for_child(u):
                assert isinstance(iedge, gigl.graph.IEdge)
                assert iedge.child == u
                edges.add(iedge.id)
        assert len(edges) == simple_ts.num_edges

    def test_iedges_for_child_view(self, multi_chromosome_gig):
        gig = multi_chromosome_gig
        view = gig.iedges_for_child(1)
        assert len(view) == 3
        assert view[0].id == next(iter(view)).id
        assert [ie.child_chromosome for ie in view] == [1, 3, 3]
        assert len(gig.iedges_for_child(1, chromosome=3)) == 2
        assert len(gig.iedges_for_child(1, chromosome=99)) == 0
        assert len(gig.iedges_for_child(0)) == 0  # root

    def test_iedges_for_parent(self, multi_chromosome_gig):
        gig = multi_chromosome_gig
        for u in range(len(gig.nodes)):
            expected = {ie.id for ie in gig.iedges if ie.parent == u}
            view = gig.iedges_for_parent(u)
            assert len(view) == len(expected)
            assert {ie.id for ie in view} == expected
            assert all(isinstance(ie, gigl.graph.IEdge) for ie in view)
            chroms = [ie.parent_chromosome for ie in view]
            assert chroms == sorted(chroms)
            for c in gig.chromosomes_as_parent(u):
                expected_c = {ie.id for ie in gig.iedges if ie.parent == u and ie.parent_chromosome == c}
                assert {ie.id for ie in gig.iedges_for_parent(u, c)} == expected_c
        assert len(gig.iedges_for_parent(1, chromosome=99)) == 0
        assert len(gig.iedges_for_parent(5)) == 0  # sample: not a parent

    def test_total_sequence_length_multi_chromosome(self, multi_chromosome_gig):
        gig = multi_chromosome_gig
        assert gig.sequence_length(1, 1) == 10
        assert gig.sequence_length(1, 3) == 5
        assert gig.total_sequence_length(1) == 15
        assert gig.total_sequence_length(0) == 20

    def test_max_pos(self):
        ts = tskit.Tree.generate_balanced(2, span=4).tree_sequence
        tables = gigl.Tables.from_tree_sequence(ts.keep_intervals([[1, 3]]))
        unconnected_node = tables.nodes.add_row(time=2)
        gig = tables.graph()
        for n in ts.nodes():  # Will omit the unconnected one
            assert gig.max_pos(n.id) == 3
            if n.is_sample():
                assert gig.max_pos_as_child(n.id) == 3
                assert gig.min_pos_as_child(n.id) == 1
            else:
                assert gig.max_pos_as_child(n.id) is None
                assert gig.min_pos_as_child(n.id) is None
        assert gig.max_pos(unconnected_node) is None
        assert gig.min_pos_as_child(unconnected_node) is None

    def test_sequence_length(self, simple_ts):
        gig = gigl.Graph.from_tree_sequence(simple_ts)
        for u in gig.tables.sample_nodes():
            assert gig.sequence_length(u, chromosome=0) == simple_ts.sequence_length

    def test_sequence_length_root(self, all_sv_types_no_re_gig):
        # root will have no upward edges
        root = len(all_sv_types_no_re_gig.nodes) - 1
        for chrom in all_sv_types_no_re_gig.chromosomes(root):
            assert all_sv_types_no_re_gig.sequence_length(root, chrom) == 200
        assert all_sv_types_no_re_gig.total_sequence_length(root) == 200

    def test_no_sequence_length(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        gig = tables.graph()
        for u in gig.tables.sample_nodes():
            # any chromosomes reported as length 0
            assert gig.sequence_length(u, 0) == 0
            assert gig.sequence_length(u, 1) == 0
            assert gig.sequence_length(u, 2) == 0

    def test_samples(self, trivial_gig):
        assert np.all(trivial_gig.tables.sample_nodes() > 0)  # samples are at the end
        n = 0
        for u, sample in zip(trivial_gig.tables.sample_nodes(), trivial_gig.sample_nodes()):
            assert u == sample.id
            assert sample.time == trivial_gig.tables.nodes.time[u]
            assert trivial_gig.nodes[u].is_sample() is True
            n += 1
        assert n == np.sum(trivial_gig.tables.nodes.flags & gigl.NODE_IS_SAMPLE > 0)
        assert len(trivial_gig.sample_nodes()) == n
        assert trivial_gig.sample_nodes()[0].id == trivial_gig.tables.sample_nodes()[0]
        assert any(u.is_sample() is False for u in trivial_gig.nodes)

    def test_samples_time(self):
        tables = gigl.Tables()
        for t in (0, 0, 1, 2.5, 3):
            tables.nodes.add_row(t, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1, flags=0)
        gig = tables.graph()
        assert list(tables.sample_nodes()) == [0, 1, 2, 3, 4]
        assert list(tables.sample_nodes(time=0)) == [0, 1]
        assert list(tables.sample_nodes(time=1)) == [2]  # node 5 is not a sample
        assert list(tables.sample_nodes(time=4)) == []
        assert list(tables.sample_nodes(time=(1, 3))) == [2, 3]
        assert [u.id for u in gig.sample_nodes(time=(0, 2))] == [0, 1, 2]
        assert len(gig.sample_nodes(time=99)) == 0
        with pytest.raises(ValueError, match="start <= end"):
            tables.sample_nodes(time=(3, 1))

    def test_no_samples(self):
        tables = gigl.Tables()
        assert len(tables.sample_nodes()) == 0
        gig = tables.graph()
        assert len(gig.sample_nodes()) == 0
        assert len(gig.sample_nodes().ids) == 0

    @pytest.fixture()
    def flagged_gig(self):
        # node: time, flags
        tables = gigl.Tables()
        S, X = gigl.NODE_IS_SAMPLE, gigl.Const.NODE_IS_RE
        for t, f in ((0, S), (0, S | X), (1, X), (1, S), (2, 0)):
            tables.nodes.add_row(t, flags=f)
        return tables.graph()

    def test_filter_nodes(self, flagged_gig):
        gig = flagged_gig
        S, X = gigl.NODE_IS_SAMPLE, gigl.Const.NODE_IS_RE
        assert list(gig.filter_nodes().ids) == [0, 1, 2, 3, 4]
        assert list(gig.filter_nodes(flags=X).ids) == [1, 2]
        assert list(gig.filter_nodes(flags=S | X).ids) == [1]  # all bits must be set
        assert list(gig.filter_nodes(flags=S, time=0).ids) == [0, 1]
        assert list(gig.filter_nodes(time=1).ids) == [2, 3]
        assert list(gig.filter_nodes(time=(1, 3)).ids) == [2, 3, 4]
        assert list(gig.filter_nodes(flags=0).ids) == [0, 1, 2, 3, 4]
        view = gig.filter_nodes(flags=X)
        assert len(view) == 2
        assert all(isinstance(u, gigl.graph.Node) for u in view)
        assert view[-1].id == 2
        # the tables method returns the same IDs
        for kwargs in ({}, {"flags": X}, {"flags": S | X}, {"time": (1, 3)}, {"flags": S, "time": 1}):
            assert np.array_equal(gig.filter_nodes(**kwargs).ids, gig.tables.filter_nodes(**kwargs))

    def test_sample_nodes_is_filter_alias(self, flagged_gig):
        gig = flagged_gig
        for time in (None, 0, 1, (0, 2)):
            expected = gig.filter_nodes(flags=gigl.NODE_IS_SAMPLE, time=time).ids
            assert np.array_equal(gig.sample_nodes(time=time).ids, expected)
            assert np.array_equal(gig.tables.sample_nodes(time=time), expected)

    def test_ids_read_only(self, flagged_gig):
        gig = flagged_gig
        # the unfiltered sample IDs are cached, so must not be modifiable
        assert gig.sample_nodes().ids is gig.sample_nodes().ids
        for view in (gig.nodes, gig.sample_nodes(), gig.filter_nodes(time=1), gig.iedges):
            with pytest.raises(ValueError, match="read-only"):
                view.ids[...] = 0
        assert list(gig.sample_nodes().ids) == [0, 1, 3]

    def test_nodes_ids(self, flagged_gig):
        assert list(flagged_gig.nodes.ids) == list(range(len(flagged_gig.nodes)))
        assert flagged_gig.nodes[-1].id == len(flagged_gig.nodes) - 1

    def test_node_objects_as_arguments(self, multi_chromosome_gig):
        gig = multi_chromosome_gig
        u = gig.nodes[1]
        assert [ie.id for ie in gig.iedges_for_child(u)] == [ie.id for ie in gig.iedges_for_child(1)]
        assert [ie.id for ie in gig.iedges_for_parent(u)] == [ie.id for ie in gig.iedges_for_parent(1)]
        assert gig.total_sequence_length(u) == gig.total_sequence_length(1)
        assert gig.chromosomes(u) == gig.chromosomes(1)
        with pytest.raises(TypeError, match="integer ID"):
            gig.iedges_for_child("1")


class TestTskitNames:
    @pytest.mark.parametrize(
        ("attr", "expected"),
        [
            ("num_nodes", lambda g: len(g.nodes)),
            ("num_edges", lambda g: len(g.iedges)),
            ("num_individuals", lambda g: len(g.individuals)),
            ("num_samples", lambda g: len(g.sample_nodes())),
        ],
    )
    def test_num_properties(self, all_sv_types_1re_gig, attr, expected):
        with pytest.warns(gigl.TskitNameWarning, match=attr):
            assert getattr(all_sv_types_1re_gig, attr) == expected(all_sv_types_1re_gig)

    def test_samples_returns_ids(self, all_sv_types_1re_gig):
        gig = all_sv_types_1re_gig
        with pytest.warns(gigl.TskitNameWarning, match="sample_nodes"):
            assert np.array_equal(gig.samples(), gig.sample_nodes().ids)

    def test_node_and_individual(self, all_sv_types_1re_gig):
        gig = all_sv_types_1re_gig
        with pytest.warns(gigl.TskitNameWarning, match=r"gig\.nodes\[u\]"):
            assert gig.node(1) == gig.nodes[1]
        tables = gigl.Tables()
        tables.individuals.add_row()
        gig = tables.graph()
        with pytest.warns(gigl.TskitNameWarning, match=r"gig\.individuals\[i\]"):
            assert gig.individual(0) == gig.individuals[0]


class TestTskit:
    """
    Methods that involve tree_sequence conversion
    """

    # Test high level functions
    def test_simple_from_tree_sequence(self, simple_ts):
        assert simple_ts.num_trees > 1
        gig = gigl.Graph.from_tree_sequence(simple_ts)
        assert np.all(gig.tables.sample_nodes() == simple_ts.samples())
        assert [u.id for u in gig.sample_nodes()] == list(simple_ts.samples())

    def test_to_tree_sequence(self, degree2_2_tip_ts):
        gig = gigl.Graph.from_tree_sequence(degree2_2_tip_ts)
        L = degree2_2_tip_ts.sequence_length + 100
        ts = gig.to_tree_sequence(sequence_length=L)
        assert ts.num_samples == 2
        assert ts.num_trees == 3
        assert ts.at_index(2).num_edges == 0  # empty region at end
        assert ts.sequence_length == L

    def test_to_tree_sequence_bad_length(self, degree2_2_tip_ts):
        gig = gigl.Graph.from_tree_sequence(degree2_2_tip_ts)
        L = degree2_2_tip_ts.sequence_length - 1
        with pytest.raises(tskit.LibraryError):
            gig.to_tree_sequence(sequence_length=L)

    def test_roundtrip(self, simple_ts):
        gig = gigl.Graph.from_tree_sequence(simple_ts)
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
        gig = gigl.Graph.from_tree_sequence(simple_ts.simplify())
        new_gig = gig.sample_resolve()  # Shouldn't make any changes
        assert gig.tables == new_gig.tables

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
        assert extended_inversion_gig.max_pos(0) == 155
        assert extended_inversion_gig.min_pos_as_child(0) == 20

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

    def test_all_svs_1re_sample_resolve(self, all_sv_types_1re_gig):
        """
        A gig with one re node. Some of the edges will be trimmed
        """
        new_gig = all_sv_types_1re_gig.sample_resolve()
        iedge_rows = list(all_sv_types_1re_gig.iedges_for_child(2))
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

        re_nodes = np.where(new_gig.tables.nodes.flags & Const.NODE_IS_RE)[0]
        assert len(re_nodes) == 1

    def test_all_svs_2re_sample_resolve(self, all_sv_types_2re_gig):
        """
        An even more complicated gig. In particular, recombination means that some of
        the iedges leading into the RE nodes should be trimmed. We should still split
        the 1->2 iedge but also we have an internal edge from 5->7 that contains
        a missing section
        """
        new_gig = all_sv_types_2re_gig.sample_resolve()
        assert len(new_gig.iedges) - len(all_sv_types_2re_gig.iedges) == 2
        iedge_rows = list(all_sv_types_2re_gig.iedges_for_child(2))
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

        re_nodes = np.where(new_gig.tables.nodes.flags & Const.NODE_IS_RE)[0]
        assert len(re_nodes) == 2

        # One of the recombination nodes
        iedge_rows = list(all_sv_types_2re_gig.iedges_for_child(7))
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

    def test_sample_resolve_with_chromosomes(self, multi_chromosome_gig):
        new_gig = multi_chromosome_gig.sample_resolve()
        assert len(new_gig.iedges) == len(multi_chromosome_gig.iedges)
        assert new_gig.iedges != multi_chromosome_gig.iedges
        assert new_gig.iedges[0].child_chromosome == 1
        assert new_gig.iedges[0].parent_chromosome == 0
        assert new_gig.iedges[0].child_left == 0
        assert new_gig.iedges[0].child_right == 5
        assert new_gig.iedges[0].parent_left == 0
        assert new_gig.iedges[0].parent_right == 5
        # last sample is not attached to any other chromosomes
        assert new_gig.iedges[-1] == multi_chromosome_gig.iedges[-1]

        # check all iedges simply differ by lef/right
        for ie1, ie2 in zip(new_gig.iedges, multi_chromosome_gig.iedges):
            assert ie1.child == ie2.child
            assert ie1.parent == ie2.parent
            assert ie1.child_chromosome == ie2.child_chromosome
            assert ie1.parent_chromosome == ie2.parent_chromosome


class TestIEdge:
    """
    Test the wrapper around a single iedge
    """

    def basic_edge(self, *args):
        return gigl.graph.IEdge(
            *args,
            child=0,
            parent=1,
            edge=-1,
            id=0,
            child_chromosome=0,
            parent_chromosome=0,
        )

    @pytest.mark.parametrize(
        "name",
        ["parent", "child", "parent_left", "parent_right", "child_left", "child_right"],
    )
    def test_edge_accessors(self, simple_ts, name):
        gig = gigl.Graph.from_tree_sequence(simple_ts)
        suffix = name.split("_")[-1]

        ie = gig.iedges[gig._iedge_map_sorted_by_parent[0]]  # tskit order
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

    @pytest.mark.parametrize("direction", [Const.ROOTWARDS, Const.LEAFWARDS])
    def test_notransform_position(self, direction):
        ie = self.basic_edge(10, 20, 10, 20)
        assert ie.transform_position(10, direction) == 10
        assert ie.transform_position(12, direction) == 12
        assert ie.transform_position(17, direction) == 17
        nd_type = "child" if direction == Const.ROOTWARDS else "parent"
        with pytest.raises(ValueError, match=f"not in {nd_type} interval"):
            ie.transform_position(9, direction)
        with pytest.raises(ValueError, match=f"not in {nd_type} interval"):
            ie.transform_position(20, direction)

    def test_transform_position_linear(self):
        ie = self.basic_edge(0, 10, 10, 20)
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

    @pytest.mark.parametrize("direction", [Const.ROOTWARDS, Const.LEAFWARDS])
    def test_transform_position_inversion(self, direction):
        ie = self.basic_edge(10, 20, 20, 10)
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
        ie = self.basic_edge(10, 20, 30, 20)
        assert ie.transform_position(10, Const.ROOTWARDS) == 29
        assert ie.transform_position(11, Const.ROOTWARDS) == 28
        assert ie.transform_position(12, Const.ROOTWARDS) == 27
        assert ie.transform_position(17, Const.ROOTWARDS) == 22
        assert ie.transform_position(18, Const.ROOTWARDS) == 21
        assert ie.transform_position(19, Const.ROOTWARDS) == 20
