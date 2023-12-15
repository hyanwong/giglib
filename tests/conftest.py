import GeneticInheritanceGraph as gigl
import msprime
import pytest
import tests.gigutil as gigutil


@pytest.fixture(scope="session")
def simple_ts():
    return msprime.sim_ancestry(
        2, recombination_rate=1, sequence_length=10, random_seed=1
    )


@pytest.fixture(scope="session")
def simple_ts_with_mutations():
    ts = msprime.sim_ancestry(
        2, recombination_rate=1, sequence_length=10, random_seed=1
    )
    mutated_ts = msprime.sim_mutations(ts, rate=0.1, random_seed=1)
    return mutated_ts


@pytest.fixture(scope="session")
def ts_with_multiple_pops():
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=500)
    demography.add_population(name="B", initial_size=500)
    demography.add_population(name="C", initial_size=200)
    demography.add_population_split(time=100, derived=["A", "B"], ancestral="C")
    ts = msprime.sim_ancestry(
        samples={"A": 1, "B": 1}, demography=demography, random_seed=1
    )
    return ts


@pytest.fixture(scope="session")
def all_sv_types_gig():
    """
    Contains a single deletion, a single duplication, and a single inversion
    """
    # p | c | c_left | c_right | p_left | p_right
    iedge_data = [
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

    tables = gigl.Tables()
    tables.iedges = gigutil.make_iedges_table(iedge_data, tables)
    tables.nodes = gigutil.make_nodes_table(node_data, tables)
    tables.sort()
    return gigl.Graph(tables)


@pytest.fixture(scope="session")
def trivial_gig():
    # p | c | c_left | c_right | p_left | p_right
    iedge_data = [
        (4, 3, 0, 5, 0, 5),
        (3, 0, 0, 3, 3, 0),
        (3, 0, 3, 5, 3, 5),
        (4, 1, 0, 5, 0, 5),
        (4, 2, 0, 5, 0, 5),
    ]
    # time | flags
    node_data = [
        (0, gigl.NODE_IS_SAMPLE),
        (0, gigl.NODE_IS_SAMPLE),
        (0, gigl.NODE_IS_SAMPLE),
        (1, 0),
        (2, 0),
    ]
    tables = gigl.Tables()
    tables.iedges = gigutil.make_iedges_table(iedge_data, tables)
    tables.nodes = gigutil.make_nodes_table(node_data, tables)
    tables.sort()
    return gigl.Graph(tables)


@pytest.fixture(scope="session")
def gig_from_degree2_ts():
    ts = msprime.sim_ancestry(
        5, sequence_length=10, recombination_rate=0.02, random_seed=1
    )
    assert ts.num_trees == 2
    return gigl.from_tree_sequence(ts)
