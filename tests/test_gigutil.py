"""
Tests of the helpers in gigutil, which build the examples used by other tests
"""

from tests.gigutil import random_tables


class TestRandomTables:
    def test_without_recombination_each_child_has_one_parent(self):
        tables = random_tables(1)
        for u in range(1, len(tables.nodes)):
            assert len({tables.iedges[i].parent for i in tables.iedges.ids_for_child(u)}) == 1

    def test_unequal_crossovers(self):
        # Some children have two parents, and differ in length from both of them
        tables = random_tables(1, recombination=0.5)
        gig = tables.graph()
        unequal = []
        for u in range(1, len(tables.nodes)):
            parents = {tables.iedges[i].parent for i in tables.iedges.ids_for_child(u)}
            if len(parents) == 2 and all(gig.sequence_length(u) != gig.sequence_length(p) for p in parents):
                unequal.append(u)
        assert len(unequal) > 0
