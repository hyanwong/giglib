import GeneticInheritanceGraph as gig
import msprime
import pytest


@pytest.fixture(scope="session")
def simple_ts():
    return msprime.sim_ancestry(
        2, recombination_rate=1, sequence_length=10, random_seed=1
    )


@pytest.fixture(scope="session")
def simple_gig():
    interval_data = [
        (6, 0, 0, 300, 0, 300),
        (6, 1, 0, 300, 0, 300),
        (9, 6, 0, 200, 0, 200),
        (9, 6, 200, 300, 100, 200),
        (11, 9, 0, 200, 0, 200),
        (7, 2, 0, 100, 0, 100),
        (7, 3, 0, 100, 0, 100),
        (10, 7, 0, 50, 0, 50),
        (10, 7, 50, 100, 150, 200),
        (11, 10, 0, 200, 0, 200),
        (8, 4, 0, 200, 0, 200),
        (8, 5, 0, 200, 0, 200),
        (12, 8, 0, 80, 0, 80),
        (12, 8, 80, 160, 160, 80),
        (12, 8, 160, 200, 160, 200),
        (12, 11, 0, 200, 0, 200),
    ]

    node_data = [
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (1, 0),
        (2, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (6, 0),
    ]

    table_group = gig.TableGroup()
    for row in interval_data:
        table_group.intervals.add_row(
            parent=row[0],
            child=row[1],
            child_left=row[2],
            child_right=row[3],
            parent_left=row[4],
            parent_right=row[5],
        )
    for row in node_data:
        table_group.nodes.add_row(time=row[0], flags=row[1], individual=gig.NULL)
    return table_group
