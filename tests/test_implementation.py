"""
Tests of implementation details, rather than of the public API.

These tests check how the current (numpy-based) implementation stores tables and
maintains its internal indexes: for example preallocated storage sizes, the contents
of internal index structures, and internal attributes such as ``_data``. They are
expected to be rewritten or removed if the storage is reimplemented (e.g. in Rust).

All other test files should only use the public API, so that they can be used to
check any new implementation. See the "Internal storage" design note
(docs/design/storage.md, or https://hyanwong.github.io/giglib/design/storage.html).
"""

import numpy as np
import pytest

import giglib as gigl
from giglib.constants import ValidFlags
from tests.gigutil import valid_tables_without_flags


class TestPreallocation:
    @pytest.mark.parametrize("cls", ["NodeTable", "IEdgeTable", "IndividualTable", "MutationTable"])
    def test_initial_sizes(self, cls):
        table = getattr(gigl.tables, cls)(initial_size=10)
        assert table._num_preallocated_rows == 10

    def test_from_tree_sequence_initial_sizes(self, simple_ts):
        tables = gigl.Tables.from_tree_sequence(simple_ts)
        assert tables.nodes._num_preallocated_rows == simple_ts.num_nodes
        assert tables.iedges._num_preallocated_rows == simple_ts.num_edges
        assert tables.individuals._num_preallocated_rows == simple_ts.num_individuals

    def test_initial_sizes_copy(self, simple_ts):
        tables = gigl.Tables.from_tree_sequence(simple_ts)
        copy = tables.copy()
        assert copy.nodes._num_preallocated_rows == simple_ts.num_nodes
        assert copy.iedges._num_preallocated_rows == simple_ts.num_edges
        assert copy.individuals._num_preallocated_rows == simple_ts.num_individuals


class TestFreezeData:
    @pytest.mark.parametrize(
        ("TableClass", "params"),
        [
            ("NodeTable", {"time": 0}),
            ("IndividualTable", {}),
            (
                "IEdgeTable",
                dict(child_left=0, child_right=1, parent_left=0, parent_right=1, child=0, parent=1),
            ),
        ],
    )
    def test_replace_data(self, TableClass, params):
        table = getattr(gigl.tables, TableClass)()
        table.add_row(**params)
        table.freeze()
        with pytest.raises((AttributeError, ValueError)):
            table._data = []
        unfrozen = table.copy()
        unfrozen._data = []
        assert len(unfrozen) == 0
        unfrozen.add_row(**params)
        assert len(unfrozen) == 1


class TestFrozenStorage:
    def test_data_not_writeable(self, simple_ts):
        # _data is a view created before freezing, so needs its own flag unsetting
        gig = gigl.Graph.from_tree_sequence(simple_ts)
        for name in gig.tables.table_classes._fields:
            table = getattr(gig.tables, name)
            assert not table._datastore.flags.writeable
            assert not table._data.flags.writeable


class TestIEdgeIndex:
    def test_index_used_in_equality(self, trivial_gig):
        tables = trivial_gig.tables
        sample = trivial_gig.tables.sample_nodes()[0]
        chrom = 0
        tables_copy = tables.copy()
        assert tables_copy.iedges == tables.iedges
        # Warning: this is a hack to test the index, and will invalidate it
        tables_copy.iedges._id_range_for_child[sample][chrom][0] += 1
        assert tables_copy.iedges != tables.iedges
        tables_copy.iedges._id_range_for_child[sample][chrom][0] -= 1  # Restore
        assert tables_copy.iedges == tables.iedges

    def test_no_entry_for_childless_node(self):
        tables = gigl.Tables()
        tables.nodes.add_row(time=0)
        assert len(tables.iedges.ids_for_child(0)) == 0
        assert 0 not in tables.iedges._id_range_for_child

    def test_cleared(self, trivial_gig):
        tables = trivial_gig.tables.copy()
        tables.iedges.add_row(0, 4, child_left=10, child_right=20)
        assert len(tables.iedges._id_range_for_child) > 0
        tables.iedges.clear()
        assert len(tables.iedges._id_range_for_child) == 0


class TestMutationIndex:
    def test_range_for_node(self):
        # The (start, end) ranges match ids_for_node(), without creating an array
        mutations = gigl.Tables().mutations
        for node, chrom, pos in [(3, 0, 1), (3, 0, 1), (3, 0, 4), (3, 2, 0), (1, 0, 7)]:
            mutations.add_row(node, pos, "A", chromosome=chrom)
        assert mutations._range_for_node(3) == (0, 4)
        assert mutations._range_for_node(3, chromosome=2) == (3, 4)
        assert mutations._range_for_node(0) == (0, 0)
        assert mutations._range_for_node(3, chromosome=1) == (0, 0)
        for node, chrom in [(3, None), (3, 2), (1, None), (0, None), (3, 1)]:
            start, end = mutations._range_for_node(node, chromosome=chrom)
            kwargs = {} if chrom is None else {"chromosome": chrom}
            assert list(range(start, end)) == list(mutations.ids_for_node(node, **kwargs))


class TestGraphParentIndex:
    def test_no_edges(self):
        tables = gigl.Tables()
        tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)
        tables.nodes.add_row(1, flags=0)
        gig = tables.graph()
        # Cached values should be defined but empty
        assert len(gig._id_range_for_parent) == 0
        assert gig._iedge_map_sorted_by_parent.shape == (0,)

    @pytest.mark.parametrize(
        "name",
        ["parent", "child", "parent_left", "parent_right", "child_left", "child_right"],
    )
    def test_parent_order_matches_tskit(self, simple_ts, name):
        # Iedges sorted by parent are in the same order as tskit edges
        gig = gigl.Graph.from_tree_sequence(simple_ts)
        suffix = name.split("_")[-1]
        ts_order = gig._iedge_map_sorted_by_parent
        assert np.all(getattr(gig.tables.iedges, name)[ts_order] == getattr(simple_ts, "edges_" + suffix))
        ie = gig.iedges[ts_order[0]]
        assert getattr(ie, name) == getattr(simple_ts.edge(0), suffix)


class TestIncrementalValidationChecks:
    """
    Which rows and flags Tables.validate() checks, as recorded by wrapping the private
    method that checks them: incremental validation only checks what it needs to
    """

    @staticmethod
    def record_checks(tables, monkeypatch):
        # Record the (flags, start) of each check of some iedges
        checks = []
        check = tables._check_iedges

        def recording_check(flags, start=0, stop=None):
            if flags and start < (len(tables.iedges) if stop is None else stop):
                checks.append((flags, start))
            check(flags, start, stop)

        monkeypatch.setattr(tables, "_check_iedges", recording_check)
        return checks

    def test_only_new_rows_are_checked(self, monkeypatch):
        tables = valid_tables_without_flags()
        tables.validate()
        checks = self.record_checks(tables, monkeypatch)
        tables.validate()
        assert checks == []  # nothing new to check
        n = len(tables.iedges)
        tables.nodes.add_row(0)  # adding nodes doesn't affect the existing iedges
        tables.iedges.add_row(4, 1, child_right=10)
        tables.validate()
        assert checks == [(ValidFlags.GIG, n)]
        assert tables.iedges.flags == ValidFlags.GIG

    def test_newly_required_flags_checked_for_all_rows(self, monkeypatch):
        tables = valid_tables_without_flags()
        tables.validate(ValidFlags.IEDGES_INTERVALS)
        checks = self.record_checks(tables, monkeypatch)
        tables.iedges.add_row(3, 1, child_left=10, child_right=12, child_chromosome=1)
        tables.validate(ValidFlags.IEDGES_INTERVALS | ValidFlags.IEDGES_INTEGERS)
        assert checks == [(ValidFlags.IEDGES_INTEGERS, 0), (ValidFlags.IEDGES_INTERVALS, len(tables.iedges) - 1)]

    def test_changed_nodes_rechecked_for_all_rows(self, monkeypatch):
        tables = valid_tables_without_flags()
        tables.validate()
        checks = self.record_checks(tables, monkeypatch)
        tables.change_times(1)
        tables.validate()
        assert checks == [(ValidFlags.IEDGES_COMBO_NODE_TABLE, 0)]

    def test_copied_tables_keep_node_time_flags(self, monkeypatch):
        tables = valid_tables_without_flags()
        tables.validate()
        copy = tables.copy()
        checks = self.record_checks(copy, monkeypatch)
        copy.validate()
        assert checks == []  # nothing needs checking again


class TestClosestHomologuesStopsEarly:
    def test_stops_once_all_positions_have_homology(self, monkeypatch):
        # u and v (nodes 0 and 1) are children of A (node 2), which has a chain of 50
        # older ancestors above it. All the positions of u and v find their closest
        # homology at A, so tracing should stop there, rather than continue up the chain
        tables = gigl.Tables()
        tables.nodes.add_row(0)
        tables.nodes.add_row(0)
        for time in range(1, 52):
            tables.nodes.add_row(time)
        for child in range(52):
            parent = 2 if child < 2 else child + 1
            tables.iedges.add_row(child, parent, child_right=10)
        tables.sort()
        tables.validate()

        # Record the nodes whose iedges are looked up as tracing goes up through them
        looked_up = []
        ids_for_child = tables.iedges.ids_for_child

        def recording_ids_for_child(node, *args, **kwargs):
            looked_up.append(node)
            return ids_for_child(node, *args, **kwargs)

        monkeypatch.setattr(tables.iedges, "ids_for_child", recording_ids_for_child)
        tables.find_homology(0, 1, mode="closest_homologues")
        assert set(looked_up) <= {0, 1, 2}
        looked_up.clear()
        tables.find_homology(0, 1, mode="all")
        assert len(set(looked_up)) > 50  # "all" carries on to the root
