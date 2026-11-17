import msprime
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
        tables.iedges.add_row(0, 1, child_left=0, child_right=1, parent_left=1, parent_right=0)
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
        assert len(gig.iedges_for_parent(1)) == 0

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
        tables.iedges.add_row(0, 0, child_left=0, child_right=1, parent_left=1, parent_right=0)
        with pytest.raises(ValueError, match="0 not older than child 0"):
            tables.graph()

    def test_negative_parent_child_id(self, simple_ts):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(-1, 1, child_left=0, child_right=1, parent_left=1, parent_right=0)
        with pytest.raises(ValueError, match="negative"):
            tables.graph()

    def test_bigger_parent_child_id(self, simple_ts):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(0, 10, child_left=0, child_right=1, parent_left=1, parent_right=0)
        with pytest.raises(ValueError, match="ID >= num nodes"):
            tables.graph()

    def test_unmatched_parent_child_spans(self, simple_ts):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1, flags=gigl.NODE_IS_SAMPLE)
        tables.iedges.add_row(0, 1, child_left=0, child_right=1, parent_left=2, parent_right=0)
        with pytest.raises(ValueError, match="spans"):
            tables.graph()
        tables.iedges.clear()
        assert len(tables.iedges) == 0
        # test something more standard than an inversion
        tables.iedges.add_row(0, 1, child_left=1, child_right=2, parent_left=0, parent_right=2)
        with pytest.raises(ValueError, match="spans"):
            tables.graph()

    def test_bad_spans(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1, flags=0)
        tables.iedges.add_row(0, 1, child_left=0, child_right=0, parent_left=0, parent_right=1)
        with pytest.raises(ValueError, match="spans"):
            tables.graph()

    def test_id_unsorted(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1)
        tables.iedges.add_row(0, 2, child_right=1)
        tables.iedges.add_row(1, 2, child_left=1, child_right=2, parent_left=0, parent_right=1)
        tables.iedges.add_row(0, 2, child_left=2, child_right=3, parent_left=0, parent_right=1)
        with pytest.raises(ValueError, match="not sorted by child"):
            tables.graph()

    def test_time_unsorted(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1)
        tables.nodes.add_row(2)
        tables.iedges.add_row(0, 2, child_left=0, child_right=1, parent_left=1, parent_right=0)
        tables.iedges.add_row(1, 2, child_left=0, child_right=1, parent_left=1, parent_right=0)
        with pytest.raises(ValueError, match="not sorted by child time, descending"):
            tables.graph()

    def test_duplicate_parents(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1)
        tables.nodes.add_row(2)
        tables.iedges.add_row(0, 1, child_right=2)
        tables.iedges.add_row(0, 1, child_left=1, child_right=3)
        with pytest.raises(ValueError, match="multiple or duplicate parents"):
            tables.graph()

    def test_multiple_parents(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1)
        tables.nodes.add_row(2)
        tables.iedges.add_row(0, 1, child_right=2)
        tables.iedges.add_row(0, 2, child_left=1, child_right=3)
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
                assert {ie.id for ie in gig.iedges_for_parent(u, chromosome=c)} == expected_c
        assert len(gig.iedges_for_parent(1, chromosome=99)) == 0
        assert len(gig.iedges_for_parent(5)) == 0  # sample: not a parent

    def test_total_sequence_length_multi_chromosome(self, multi_chromosome_gig):
        gig = multi_chromosome_gig
        assert gig.sequence_length(1, chromosome=1) == 10
        assert gig.sequence_length(1, chromosome=3) == 5
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
            assert all_sv_types_no_re_gig.sequence_length(root, chromosome=chrom) == 200
        assert all_sv_types_no_re_gig.total_sequence_length(root) == 200

    def test_no_sequence_length(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        gig = tables.graph()
        for u in gig.tables.sample_nodes():
            # any chromosomes reported as length 0
            assert gig.sequence_length(u, chromosome=0) == 0
            assert gig.sequence_length(u, chromosome=1) == 0
            assert gig.sequence_length(u, chromosome=2) == 0

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

    @pytest.mark.parametrize("filtered", [False, True])
    def test_view_index_bounds(self, flagged_gig, filtered):
        def make_view():
            return flagged_gig.filter_nodes(time=(0, 2)) if filtered else flagged_gig.nodes

        # Use fresh views throughout: accessing .ids on an unfiltered view
        # creates an ID array, which changes the indexing code path
        n = len(make_view())
        assert n > 1
        for bad in (n, n + 1, -n - 1, -2 * n):
            with pytest.raises(IndexError):
                make_view()[bad]
        ids = make_view().ids
        for i in range(-n, n):
            assert make_view()[i].id == ids[i]

    def test_iedges_index_bounds(self, multi_chromosome_gig):
        iedges = multi_chromosome_gig.iedges
        n = len(iedges)
        assert iedges[-n].id == 0
        assert iedges[-1].id == n - 1
        with pytest.raises(IndexError):
            iedges[-n - 1]

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

    def test_edges(self, all_sv_types_1re_gig):
        gig = all_sv_types_1re_gig
        with pytest.warns(gigl.TskitNameWarning, match=r"gig\.edges\(\) .* try gig\.iedges"):
            edges = gig.edges()
        assert len(edges) == len(gig.iedges)
        assert list(edges.ids) == list(gig.iedges.ids)

    def test_tables_edges(self):
        tables = gigl.Tables()
        tables.nodes.add_row(1)
        tables.nodes.add_row(0)
        with pytest.warns(gigl.TskitNameWarning, match=r"tables\.edges .* try tables\.iedges"):
            edges = tables.edges
        assert edges is tables.iedges
        # Still works (and warns) once the tables are frozen in a graph
        gig = tables.graph()
        with pytest.warns(gigl.TskitNameWarning):
            assert gig.tables.edges is gig.tables.iedges


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
        tables.iedges.add_row(1, 2, child_right=100)
        tables.iedges.add_row(0, 1, child_left=1, child_right=2, parent_left=2, parent_right=1)
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
        tables.iedges.add_row(1, 2, child_left=0, child_right=100, parent_left=100, parent_right=0)
        tables.iedges.add_row(0, 1, child_left=2, child_right=5, parent_left=5, parent_right=2)
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

    def basic_edge(self, *coords):
        # coords are (child_left, child_right, parent_left, parent_right)
        return gigl.graph.IEdge(
            **dict(zip(("child_left", "child_right", "parent_left", "parent_right"), coords)),
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
        # The first tskit edge has the lowest parent time, then parent ID
        edge = simple_ts.edge(0)
        matches = [ie for ie in gig.iedges_for_parent(edge.parent) if ie.child == edge.child]
        (ie,) = (ie for ie in matches if ie.child_left == edge.left)
        assert getattr(ie, name) == getattr(edge, suffix)

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


class TestMutations:
    @staticmethod
    def make_tables():
        """
        Node 1 inherits [0, 5) from node 2 (time 1) and [5, 10) from node 3 (time 2),
        so the parent above node 1 depends on the position
        """
        tables = gigl.Tables()
        for t, f in ((0, gigl.NODE_IS_SAMPLE), (0, gigl.NODE_IS_SAMPLE), (1, 0), (2, 0)):
            tables.nodes.add_row(t, flags=f)
        tables.add_iedge_row(0, 2, child_right=10)
        tables.add_iedge_row(1, 2, child_right=5)
        tables.add_iedge_row(1, 3, child_left=5, child_right=10)
        tables.add_iedge_row(2, 3, child_right=10)
        tables.sort()
        return tables

    def test_no_mutations(self):
        gig = self.make_tables().graph()
        assert len(gig.mutations) == 0
        assert len(gig.mutations_for_node(0)) == 0

    def test_accessors(self):
        tables = self.make_tables()
        tables.mutations.add_row(1, 7, "T", time=1.5)
        tables.mutations.add_row(3, 2, "A")  # above the root: ancestral state
        tables.mutations.add_row(1, 2, "G", time=0.5)
        tables.mutations.add_row(0, 0, "C", chromosome=1)  # no iedges on chr 1: allowed
        tables.sort()
        gig = tables.graph()
        assert len(gig.mutations) == 4
        assert all(isinstance(m, gigl.graph.Mutation) for m in gig.mutations)
        assert [m.id for m in gig.mutations] == [0, 1, 2, 3]
        view = gig.mutations_for_node(1)
        assert len(view) == 2
        assert [m.position for m in view] == [2, 7]
        assert [m.node for m in view] == [1, 1]
        assert view.ids[0] == view[0].id
        assert len(gig.mutations_for_node(1, chromosome=1)) == 0
        assert len(gig.mutations_for_node(2)) == 0
        assert [m.derived_state for m in gig.mutations_for_node(gig.nodes[3])] == ["A"]
        assert [m.chromosome for m in gig.mutations_for_node(0)] == [1]
        root_mutation = gig.mutations_for_node(3)[0]
        assert not root_mutation.is_time_known()
        assert view[0].is_time_known()

    @pytest.mark.parametrize("column", ["node", "chromosome", "position", "time"])
    def test_columns_read_only(self, column):
        # Editing a column directly would leave a stale index, so is refused
        tables = gigl.Tables()
        tables.nodes.add_row(0)
        tables.nodes.add_row(0)
        tables.mutations.add_row(0, 0, "A", time=0)
        with pytest.raises(ValueError, match="read-only"):
            getattr(tables.mutations, column)[0] = 1
        gig = tables.graph()
        assert len(gig.mutations_for_node(0)) == 1
        assert gig.allele_at(0, 0) == "A"

    def test_ids_match_tables_layer(self):
        tables = self.make_tables()
        tables.mutations.add_row(1, 2, "G")
        tables.mutations.add_row(1, 7, "T")
        gig = tables.graph()
        assert np.array_equal(gig.mutations_for_node(1).ids, gig.tables.mutations.ids_for_node(1))

    def test_bad_node(self):
        tables = self.make_tables()
        tables.mutations.add_row(4, 0, "A")
        with pytest.raises(ValueError, match="not in the node table"):
            tables.graph()

    def test_negative_position(self):
        tables = self.make_tables()
        tables.mutations.add_row(0, -1, "A")
        with pytest.raises(ValueError, match="negative positions"):
            tables.graph()

    @pytest.mark.parametrize("time", [np.inf, -np.inf, np.nan])
    def test_bad_time(self, time):
        # a plain NaN is not the same as UNKNOWN_TIME
        tables = self.make_tables()
        tables.mutations.add_row(3, 0, "A", time=time)
        with pytest.raises(ValueError, match="neither finite nor UNKNOWN_TIME"):
            tables.graph()

    def test_unsorted(self):
        tables = self.make_tables()
        tables.mutations.add_row(0, 5, "A")
        tables.mutations.add_row(0, 4, "A")
        with pytest.raises(ValueError, match="tables.sort"):
            tables.graph()
        tables.sort()
        assert [m.position for m in tables.graph().mutations] == [4, 5]

    def test_younger_than_node(self):
        tables = self.make_tables()
        tables.mutations.add_row(2, 0, "A", time=0.5)
        with pytest.raises(ValueError, match="younger than its node"):
            tables.graph()

    def test_time_equal_to_node_allowed(self):
        tables = self.make_tables()
        tables.mutations.add_row(2, 0, "A", time=1)
        assert len(tables.graph().mutations) == 1

    @pytest.mark.parametrize(("position", "valid"), [(2, False), (4, False), (5, True), (9, True)])
    def test_older_than_parent_at_position(self, position, valid):
        # node 1's parent is node 2 (time 1) for [0, 5), node 3 (time 2) for [5, 10)
        tables = self.make_tables()
        tables.mutations.add_row(1, position, "A", time=1.5)
        if valid:
            assert len(tables.graph().mutations) == 1
        else:
            with pytest.raises(ValueError, match="not younger than the parent"):
                tables.graph()

    def test_time_equal_to_parent_invalid(self):
        tables = self.make_tables()
        tables.mutations.add_row(0, 0, "A", time=1)
        with pytest.raises(ValueError, match="not younger than the parent"):
            tables.graph()

    def test_above_root_has_no_upper_limit(self):
        tables = self.make_tables()
        tables.mutations.add_row(3, 0, "A", time=1e9)
        assert len(tables.graph().mutations) == 1

    def test_time_order_at_same_position(self):
        tables = self.make_tables()
        tables.mutations.add_row(0, 3, "A", time=0.2)
        tables.mutations.add_row(0, 3, "C", time=0.7)  # older, but later in the table
        with pytest.raises(ValueError, match="must be in time order"):
            tables.graph()

    def test_time_order_only_within_position(self):
        tables = self.make_tables()
        tables.mutations.add_row(0, 3, "A", time=0.2)
        tables.mutations.add_row(0, 4, "C", time=0.7)
        assert len(tables.graph().mutations) == 2

    def test_mixed_known_and_unknown_times(self):
        tables = self.make_tables()
        tables.mutations.add_row(3, 3, "A")  # unknown: e.g. ancestral state
        tables.mutations.add_row(3, 3, "C", time=5)
        tables.mutations.add_row(3, 3, "G")
        tables.mutations.add_row(3, 3, "T", time=4)
        gig = tables.graph()
        assert [m.derived_state for m in gig.mutations] == ["A", "C", "G", "T"]
        # known times must still be in order, even with unknown times in between
        tables = self.make_tables()
        tables.mutations.add_row(3, 3, "C", time=4)
        tables.mutations.add_row(3, 3, "G")
        tables.mutations.add_row(3, 3, "T", time=5)
        with pytest.raises(ValueError, match="must be in time order"):
            tables.graph()

    def test_tskit_names(self):
        tables = self.make_tables()
        tables.mutations.add_row(3, 0, "A")
        gig = tables.graph()
        with pytest.warns(gigl.TskitNameWarning, match=r"len\(gig\.mutations\)"):
            assert gig.num_mutations == 1
        with pytest.warns(gigl.TskitNameWarning, match=r"gig\.mutations\[i\]"):
            assert gig.mutation(0) == gig.mutations[0]

    def test_mutation_equality_with_unknown_times(self):
        tables = self.make_tables()
        tables.mutations.add_row(3, 0, "A")
        gig = tables.graph()
        assert gig.mutations[0] == gig.mutations[0]
        assert hash(gig.mutations[0]) == hash(gig.mutations[0])
        assert gig.mutations[0] != gig.tables.mutations[0]  # no id in a table row


def _all_nodes_as_samples(ts):
    tables = ts.dump_tables()
    tables.nodes.flags = np.full(tables.nodes.num_rows, tskit.NODE_IS_SAMPLE, dtype=np.uint32)
    return tables.tree_sequence()


def _remove_mutation_times(ts):
    tables = ts.dump_tables()
    tables.mutations.time = np.full(tables.mutations.num_rows, tskit.UNKNOWN_TIME)
    return tables.tree_sequence()


class TestHaplotypesVsTskit:
    """
    Compare haplotypes with tskit's alignments, which (with no reference sequence)
    use "N" for positions with no site, and for isolated (missing) samples
    """

    @staticmethod
    def assert_haplotypes_match(ts):
        gig = gigl.Graph.from_tree_sequence(ts)
        L = int(ts.sequence_length)
        n = 0
        for u, alignment in zip(ts.samples(), ts.alignments()):
            assert gig.haplotype(u, right=L) == alignment
            n += 1
        assert n == ts.num_samples

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_simple(self, seed):
        ts = msprime.sim_ancestry(4, sequence_length=100, recombination_rate=0.01, random_seed=seed)
        ts = msprime.sim_mutations(ts, rate=0.02, random_seed=seed)
        assert ts.num_sites > 0
        self.assert_haplotypes_match(ts)

    def test_recurrent_mutations(self):
        # A high rate gives multiple mutations at the same site (and node)
        ts = msprime.sim_ancestry(3, sequence_length=20, recombination_rate=0.05, random_seed=4)
        ts = msprime.sim_mutations(ts, rate=2, random_seed=4)
        assert ts.num_mutations > 2 * ts.num_sites
        self.assert_haplotypes_match(ts)

    def test_binary_alleles(self):
        ts = msprime.sim_ancestry(3, sequence_length=50, random_seed=5)
        ts = msprime.sim_mutations(ts, rate=0.05, model=msprime.BinaryMutationModel(), random_seed=5)
        self.assert_haplotypes_match(ts)

    def test_unknown_times(self):
        ts = msprime.sim_ancestry(3, sequence_length=50, recombination_rate=0.02, random_seed=6)
        ts = _remove_mutation_times(msprime.sim_mutations(ts, rate=0.5, random_seed=6))
        self.assert_haplotypes_match(ts)

    def test_missing_data(self):
        ts = msprime.sim_ancestry(3, sequence_length=50, random_seed=7)
        ts = msprime.sim_mutations(ts, rate=0.1, random_seed=7)
        ts = ts.delete_intervals([[10, 20]], simplify=False)
        assert "N" * 10 in next(ts.alignments())
        self.assert_haplotypes_match(ts)

    @staticmethod
    def short_span_ts():
        # Sample 0's only edge ends at 5, before the end of the tree sequence (10)
        tables = tskit.TableCollection(sequence_length=10)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(time=1)
        tables.edges.add_row(0, 5, 2, 0)
        tables.edges.add_row(0, 10, 2, 1)
        tables.sites.add_row(position=2, ancestral_state="A")
        tables.sites.add_row(position=7, ancestral_state="C")
        tables.sort()
        return tables.tree_sequence()

    def test_shorter_node_span(self):
        ts = self.short_span_ts()
        gig = gigl.Graph.from_tree_sequence(ts)
        alignment = next(ts.alignments())
        assert alignment == "NNANNNNNNN"
        assert gig.haplotype(0) == "NNANN"  # stops at the end of the node's edges
        assert gig.haplotype(0, right=int(ts.sequence_length)) == alignment

    def test_reference_sequence_not_converted(self):
        tables = self.short_span_ts().dump_tables()
        tables.reference_sequence.data = "GGGGGGGGGG"
        ts = tables.tree_sequence()
        gig = gigl.Graph.from_tree_sequence(ts)
        # tskit uses the reference at positions without sites; giglib has missing data
        assert list(ts.alignments())[1] == "GGAGGGGCGG"
        assert gig.haplotype(1) == "NNANNNNCNN"

    def test_internal_nodes(self):
        ts = msprime.sim_ancestry(3, sequence_length=50, recombination_rate=0.02, random_seed=8)
        ts = _all_nodes_as_samples(msprime.sim_mutations(ts, rate=0.1, random_seed=8))
        self.assert_haplotypes_match(ts)


class TestHaplotypes:
    @staticmethod
    def two_node_tables(*iedges, root_mutations=()):
        """
        Parent node 0 (time 1) and child node 1 (time 0), with iedges given as
        (child_left, child_right, parent_left, parent_right), and mutations above the
        parent given as (position, state)
        """
        tables = gigl.Tables()
        tables.nodes.add_row(1)
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        for cl, cr, pl, pr in iedges:
            tables.add_iedge_row(1, 0, child_left=cl, child_right=cr, parent_left=pl, parent_right=pr)
        for pos, state in root_mutations:
            tables.mutations.add_row(0, pos, state)
        tables.sort()
        return tables

    def test_identity(self):
        tables = self.two_node_tables((0, 5, 0, 5), root_mutations=[(0, "A"), (2, "C"), (4, "G")])
        gig = tables.graph()
        assert gig.haplotype(0) == "ANCNG"
        assert gig.haplotype(1) == "ANCNG"
        assert gig.haplotype(1, left=2, right=4) == "CN"
        assert gig.haplotype(1, left=3, right=3) == ""
        assert gig.haplotype(1, missing_data_character="?") == "A?C?G"
        assert gig.allele_at(1, 2) == "C"
        assert gig.allele_at(1, 1) is None
        assert gig.allele_at(gig.nodes[1], 2) == "C"

    def test_bad_interval(self):
        gig = self.two_node_tables((0, 5, 0, 5)).graph()
        for left, right in [(-1, 3), (3, 2)]:
            with pytest.raises(ValueError, match="Invalid interval"):
                gig.haplotype(1, left=left, right=right)
        with pytest.raises(ValueError, match="non-negative"):
            gig.allele_at(1, -1)

    def test_no_mutations(self):
        gig = self.two_node_tables((0, 5, 0, 5)).graph()
        assert gig.haplotype(1) == "NNNNN"
        assert gig.haplotype(1, chromosome=1) == ""  # no material on chromosome 1

    def test_most_recent_mutation_wins(self):
        tables = self.two_node_tables((0, 3, 0, 3), root_mutations=[(1, "A"), (1, "C")])
        tables.mutations.add_row(1, 2, "G")
        tables.mutations.add_row(1, 2, "T")
        tables.sort()
        gig = tables.graph()
        assert gig.haplotype(0) == "NCN"
        assert gig.haplotype(1) == "NCT"

    def test_shifted_coordinates(self):
        # child [0, 3) inherits parent [5, 8)
        tables = self.two_node_tables((0, 3, 5, 8), root_mutations=[(5, "A"), (7, "T"), (0, "G")])
        gig = tables.graph()
        assert gig.haplotype(1) == "ANT"

    def test_deletion(self):
        # child loses parent positions 2 and 3
        tables = self.two_node_tables((0, 2, 0, 2), (2, 4, 4, 6), root_mutations=list(enumerate("ACGTAC")))
        gig = tables.graph()
        assert gig.haplotype(0) == "ACGTAC"
        assert gig.haplotype(1) == "ACAC"

    def test_duplication(self):
        # child has two copies of parent [0, 3)
        tables = self.two_node_tables((0, 3, 0, 3), (3, 6, 0, 3), root_mutations=list(enumerate("ACG")))
        tables.mutations.add_row(1, 4, "T")  # only in the second copy
        tables.sort()
        gig = tables.graph()
        assert gig.haplotype(1) == "ACGATG"

    def test_insertion(self):
        # child positions [2, 4) are not inherited from the parent
        tables = self.two_node_tables((0, 2, 0, 2), (4, 6, 2, 4), root_mutations=list(enumerate("ACGT")))
        gig = tables.copy().graph()
        assert gig.haplotype(1) == "ACNNGT"
        tables.mutations.add_row(1, 2, "T")
        tables.mutations.add_row(1, 3, "T")
        tables.sort()
        assert tables.graph().haplotype(1) == "ACTTGT"

    def test_indel_alleles(self):
        tables = self.two_node_tables((0, 4, 0, 4), root_mutations=list(enumerate("ACGT")))
        tables.mutations.add_row(1, 1, "CTTG")  # insertion of TTG after the C
        tables.mutations.add_row(1, 2, "")  # deletion of G
        tables.sort()
        gig = tables.graph()
        assert gig.haplotype(1) == "ACTTGT"
        assert gig.allele_at(1, 2) == ""

    def test_multiple_generations(self):
        ts = msprime.sim_ancestry(2, sequence_length=10, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.5, random_seed=1)
        gig = gigl.Graph.from_tree_sequence(ts)
        for u in gig.sample_nodes():
            hap = gig.haplotype(u)
            assert [gig.allele_at(u, x) or "N" for x in range(len(hap))] == list(hap)


class TestInversions:
    @staticmethod
    def inverted_tables(*root_mutations, complements=None):
        # child [0, 10) inherits parent [0, 10) inverted, so position x in the parent
        # is position 9 - x in the child
        tables = TestHaplotypes.two_node_tables((0, 10, 10, 0), root_mutations=root_mutations)
        if complements is not None:
            tables.allele_complements = complements
        return tables

    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            ("X", "X"),  # not in the map: left unchanged, without a warning
            ("A", "T"),
            ("C", "G"),
            ("R", "Y"),
            ("N", "N"),
            ("a", "t"),
            ("ATTG", "CAAT"),
            ("-", "-"),
            ("", ""),
            ("1", "1"),
        ],
    )
    def test_complement(self, state, expected):
        gig = self.inverted_tables((2, state)).graph()
        assert gig.allele_at(0, 2) == state
        assert gig.allele_at(1, 7) == expected

    def test_haplotype(self):
        gig = self.inverted_tables(*enumerate("AACGTTGCAC")).graph()
        assert gig.haplotype(0) == "AACGTTGCAC"
        assert gig.haplotype(1) == "GTGCAACGTT"  # reverse complement

    def test_partial_inversion(self):
        tables = TestHaplotypes.two_node_tables(
            (0, 2, 0, 2), (2, 5, 5, 2), (5, 7, 5, 7), root_mutations=list(enumerate("ACGTACG"))
        )
        gig = tables.graph()
        # parent positions 2-4 ("GTA") are reverse complemented to "TAC"
        assert gig.haplotype(1) == "AC" + "TAC" + "CG"

    def test_double_inversion(self):
        tables = TestHaplotypes.two_node_tables((0, 4, 4, 0), root_mutations=list(enumerate("ACGG")))
        tables.nodes.add_row(-1, flags=gigl.NODE_IS_SAMPLE)  # grandchild, node 2
        tables.add_iedge_row(2, 1, child_left=0, child_right=4, parent_left=4, parent_right=0)
        tables.sort()
        gig = tables.graph()
        assert gig.haplotype(1) == "CCGT"
        assert gig.haplotype(2) == "ACGG"

    def test_mutation_below_inversion(self):
        tables = self.inverted_tables((2, "A"))
        tables.mutations.add_row(1, 3, "G")  # on the child's own strand
        tables.sort()
        gig = tables.graph()
        assert gig.allele_at(1, 3) == "G"
        assert gig.allele_at(1, 7) == "T"

    def test_complementing_turned_off(self):
        with pytest.warns(UserWarning, match="ATTG"):
            gig = self.inverted_tables((2, "A"), (3, "ATTG"), complements={}).graph()
        assert gig.allele_at(1, 7) == "A"
        assert gig.allele_at(1, 6) == "ATTG"

    def test_explicit_entry_wins(self):
        gig = self.inverted_tables((2, "ins1"), complements={"ins1": "ins1_rev"}).graph()
        assert gig.allele_at(1, 7) == "ins1_rev"

    def test_warning_for_uncomplementable_alleles(self):
        # "AT" is its own reverse complement, so should not be listed
        with pytest.warns(UserWarning, match=r"\['AXY', 'B\?'\]"):
            self.inverted_tables((2, "AXY"), (3, "AT"), (4, "B?")).graph()

    def test_no_warning(self, recwarn):
        # complementable, single character, explicitly mapped, or no inversion
        self.inverted_tables((2, "AT"), (3, "1"), (4, "AXY"), complements={"AXY": "AXY", "A": "T", "T": "A"}).graph()
        TestHaplotypes.two_node_tables((0, 5, 0, 5), root_mutations=[(2, "AXY")]).graph()
        assert len([w for w in recwarn if issubclass(w.category, UserWarning)]) == 0


class TestAlleleComplements:
    def test_default(self):
        tables = gigl.Tables()
        assert tables.allele_complements["A"] == "T"
        assert tables.allele_complements["-"] == "-"
        assert tables.allele_complements["y"] == "r"
        assert tables.allele_complements is not gigl.Tables().allele_complements

    def test_copy_and_equality(self):
        tables = gigl.Tables()
        tables.allele_complements = {"X": "Y"}
        copy = tables.copy()
        assert copy.allele_complements == {"X": "Y"}
        assert copy == tables
        copy.allele_complements["Y"] = "X"
        assert copy != tables
        assert tables.allele_complements == {"X": "Y"}

    def test_frozen(self):
        gig = gigl.Tables().graph()
        with pytest.raises(TypeError):
            gig.tables.allele_complements["A"] = "A"
        assert gig.tables.copy().allele_complements["A"] == "T"


class TestToTreeSequenceWithMutations:
    @staticmethod
    def assert_round_trip(ts, exact_order=True):
        gig = gigl.Graph.from_tree_sequence(ts)
        ts2 = gig.to_tree_sequence(sequence_length=ts.sequence_length)
        ts.tables.sites.assert_equals(ts2.tables.sites)
        if exact_order:
            ts.tables.mutations.assert_equals(ts2.tables.mutations)
        else:
            # Without times, the order of mutations within a site (for different
            # nodes) cannot be recovered, so just compare their contents
            def contents(ts):
                return sorted((m.site, m.node, m.derived_state) for m in ts.mutations())

            assert contents(ts2) == contents(ts)
        assert list(ts2.alignments()) == list(ts.alignments())

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_round_trip(self, seed):
        ts = msprime.sim_ancestry(4, sequence_length=80, recombination_rate=0.02, random_seed=seed)
        ts = msprime.sim_mutations(ts, rate=0.05, random_seed=seed)
        self.assert_round_trip(ts)

    def test_round_trip_recurrent_mutations(self):
        ts = msprime.sim_ancestry(4, sequence_length=50, recombination_rate=0.02, random_seed=4)
        ts = msprime.sim_mutations(ts, rate=1, random_seed=4)
        assert ts.num_mutations > 2 * ts.num_sites
        self.assert_round_trip(ts)

    def test_round_trip_binary(self):
        ts = msprime.sim_ancestry(3, sequence_length=50, random_seed=5)
        ts = msprime.sim_mutations(ts, rate=0.1, model=msprime.BinaryMutationModel(), random_seed=5)
        self.assert_round_trip(ts)

    def test_round_trip_unknown_times(self):
        ts = msprime.sim_ancestry(3, sequence_length=50, recombination_rate=0.02, random_seed=6)
        ts = _remove_mutation_times(msprime.sim_mutations(ts, rate=0.5, random_seed=6))
        assert np.all(tskit.is_unknown_time(ts.tables.mutations.time))
        self.assert_round_trip(ts, exact_order=False)

    def test_round_trip_missing_data(self):
        ts = msprime.sim_ancestry(3, sequence_length=50, random_seed=7)
        ts = msprime.sim_mutations(ts, rate=0.1, random_seed=7)
        ts = ts.delete_intervals([[10, 20]], simplify=False)
        self.assert_round_trip(ts)

    def test_sites_without_mutations(self):
        tables = tskit.TableCollection(sequence_length=10)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(time=1)
        tables.edges.add_row(0, 10, 2, 0)
        tables.edges.add_row(0, 10, 2, 1)
        tables.sites.add_row(position=3, ancestral_state="G")
        tables.sort()
        self.assert_round_trip(tables.tree_sequence())

    @staticmethod
    def two_root_tables():
        """
        Two separate trees, with roots 2 and 5 (time 1) above samples 0, 1 and 3, 4
        """
        tables = gigl.Tables()
        for t in (0, 0, 1, 0, 0, 1):
            tables.nodes.add_row(t, flags=gigl.NODE_IS_SAMPLE if t == 0 else 0)
        for child, parent in ((0, 2), (1, 2), (3, 5), (4, 5)):
            tables.add_iedge_row(child, parent, child_right=10)
        tables.sort()
        return tables

    def test_roots_agree(self):
        tables = self.two_root_tables()
        tables.mutations.add_row(2, 4, "A")
        tables.mutations.add_row(5, 4, "A")
        tables.mutations.add_row(0, 4, "C", time=0.5)
        tables.sort()
        ts = tables.graph().to_tree_sequence()
        assert ts.num_sites == 1
        assert ts.site(0).ancestral_state == "A"
        assert [(m.node, m.derived_state) for m in ts.mutations()] == [(0, "C")]

    def test_extra_root_mutation_becomes_mutation(self):
        tables = self.two_root_tables()
        tables.mutations.add_row(2, 4, "A")
        tables.mutations.add_row(2, 4, "G", time=2)
        tables.mutations.add_row(5, 4, "A")
        tables.sort()
        ts = tables.graph().to_tree_sequence()
        assert ts.site(0).ancestral_state == "A"
        assert [(m.node, m.derived_state) for m in ts.mutations()] == [(2, "G")]

    def test_roots_disagree(self):
        tables = self.two_root_tables()
        tables.mutations.add_row(2, 4, "A")
        tables.mutations.add_row(5, 4, "T")
        tables.sort()
        with pytest.raises(ValueError, match="different ancestral states"):
            tables.graph().to_tree_sequence()

    def test_missing_ancestral_state(self):
        tables = self.two_root_tables()
        tables.mutations.add_row(2, 4, "A")
        tables.mutations.add_row(4, 4, "C")  # no mutation above root 5
        tables.sort()
        with pytest.raises(ValueError, match="not all roots"):
            tables.graph().to_tree_sequence()

    def test_mixed_times(self):
        tables = self.two_root_tables()
        tables.mutations.add_row(2, 4, "A")
        tables.mutations.add_row(5, 4, "A")
        tables.mutations.add_row(0, 4, "C", time=0.5)
        tables.mutations.add_row(1, 4, "G")
        tables.sort()
        with pytest.raises(ValueError, match="known and unknown times"):
            tables.graph().to_tree_sequence()

    def test_multiple_chromosomes(self):
        tables = self.two_root_tables()
        tables.mutations.add_row(2, 4, "A")
        tables.mutations.add_row(0, 4, "A", chromosome=1)
        tables.sort()
        with pytest.raises(ValueError, match="all be on one chromosome"):
            tables.graph().to_tree_sequence()

    def test_mutations_on_different_chromosome_to_iedges(self):
        # The only mutation is on chromosome 1, but the iedges are on chromosome 0
        tables = self.two_root_tables()
        tables.mutations.add_row(2, 4, "A", chromosome=1)
        tables.sort()
        with pytest.raises(ValueError, match="all be on one chromosome"):
            tables.graph().to_tree_sequence()

    def test_position_beyond_sequence_length(self):
        tables = self.two_root_tables()
        tables.mutations.add_row(2, 12, "A")
        tables.sort()
        with pytest.raises(ValueError, match="less than the sequence length"):
            tables.graph().to_tree_sequence()
