import GeneticInheritanceGraph as gig
import msprime
import pytest
import tests.util_tests as util_tests


@pytest.fixture(scope="session")
def simple_ts():
    return msprime.sim_ancestry(
        2, recombination_rate=1, sequence_length=10, random_seed=1
    )


@pytest.fixture(scope="session")
def all_mutation_types_gig():
    # p | c | c_left | c_right | p_left | p_right
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
    # time | flags
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
    table_group.intervals = util_tests.make_intervals_table(interval_data, table_group)
    table_group.nodes = util_tests.make_nodes_table(node_data, table_group)
    return table_group


@pytest.fixture(scope="session")
def trivial_gig():
    # p | c | c_left | c_right | p_left | p_right
    interval_data = [
        (4, 3, 0, 5, 0, 5),
        (3, 0, 3, 0, 0, 3),
        (3, 0, 3, 5, 3, 5),
        (4, 1, 0, 5, 0, 5),
        (4, 2, 0, 5, 0, 5),
    ]
    # time | flags
    node_data = [
        (0, 1),
        (0, 1),
        (0, 1),
        (1, 0),
        (2, 0),
    ]
    table_group = gig.TableGroup()
    table_group.intervals = util_tests.make_intervals_table(interval_data, table_group)
    table_group.nodes = util_tests.make_nodes_table(node_data, table_group)
    return table_group
