import msprime
import numpy as np
import pytest
import tskit

import GeneticInheritanceGraphLibrary as gigl
from GeneticInheritanceGraphLibrary.constants import Const
from tests.gigutil import add_iedge, make_nodes_table


def no_population_ts():
    ts = msprime.sim_ancestry(2, recombination_rate=1, sequence_length=10, random_seed=1)
    # remove population data: GIGs do not have defined populations
    tables = ts.dump_tables()
    tables.populations.clear()
    tables.populations.metadata_schema = tskit.MetadataSchema(None)
    tables.nodes.population = np.full_like(tables.nodes.population, tskit.NULL)
    return tables.tree_sequence()


@pytest.fixture(scope="session")
def simple_ts():
    return no_population_ts()


@pytest.fixture(scope="session")
def simple_ts_with_mutations():
    ts = no_population_ts()
    mutated_ts = msprime.sim_mutations(ts, rate=0.1, random_seed=1)
    return mutated_ts


@pytest.fixture(scope="session")
def ts_with_multiple_pops():
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=500)
    demography.add_population(name="B", initial_size=500)
    demography.add_population(name="C", initial_size=200)
    demography.add_population_split(time=100, derived=["A", "B"], ancestral="C")
    ts = msprime.sim_ancestry(samples={"A": 1, "B": 1}, demography=demography, random_seed=1)
    return ts


@pytest.fixture(scope="session")
def trivial_gig():
    # time | flags
    node_data = [
        (2, 0),
        (1, 0),
        (0, gigl.NODE_IS_SAMPLE),
        (0, gigl.NODE_IS_SAMPLE),
        (0, gigl.NODE_IS_SAMPLE),
    ]
    tables = gigl.Tables()
    tables.nodes = make_nodes_table(node_data, tables)
    add_iedge(tables, 0, 5, 0, 5, c=1, p=0)
    add_iedge(tables, 0, 5, 0, 5, c=2, p=0)
    add_iedge(tables, 0, 5, 0, 5, c=3, p=0)
    add_iedge(tables, 0, 3, 3, 0, c=4, p=1)  # inversion
    add_iedge(tables, 3, 5, 3, 5, c=4, p=1)
    return gigl.Graph(tables)


@pytest.fixture(scope="session")
def extended_inversion_gig():
    """
    Contains a single inversion that covers a larger region that passed up from sample A.
    This is useful to test the case where the inversion edges are outside the sample
    resolved region
    """
    node_data = [
        (0, gigl.NODE_IS_SAMPLE),
        (0, gigl.NODE_IS_SAMPLE),
        (1, 0),
        (2, 0),
    ]
    tables = gigl.Tables()
    tables.nodes = make_nodes_table(node_data, tables)
    add_iedge(tables, 20, 155, 20, 155, c=0, p=2)
    add_iedge(tables, 10, 160, 160, 10, c=2, p=3)
    add_iedge(tables, 0, 100, 0, 100, c=1, p=3)
    tables.sort()
    return gigl.Graph(tables)


@pytest.fixture(scope="session")
def double_inversion_gig():
    """
    Contains two stacked inversions on one branch that should
    be invisible once simplified
    """
    node_data = [
        (0, gigl.NODE_IS_SAMPLE),
        (0, gigl.NODE_IS_SAMPLE),
        (1, 0),
        (2, 0),
    ]
    tables = gigl.Tables()
    tables.nodes = make_nodes_table(node_data, tables)
    add_iedge(tables, 0, 100, 300, 200, c=0, p=2)
    add_iedge(tables, 200, 300, 100, 0, c=2, p=3)
    add_iedge(tables, 0, 100, 0, 100, c=1, p=3)
    tables.sort()
    return gigl.Graph(tables)


@pytest.fixture(scope="session")
def multi_chromosome_gig():
    """
    Switch into a separate numbered chromosome for one of the samples
    and have 2 chrs in one of the ancestors
    """
    # time | flags
    node_data = [
        (2, 0),
        (1, 0),  # will have material on chr 1 and 3
        (0, gigl.NODE_IS_SAMPLE),  # will have material on chr 5
        (0, gigl.NODE_IS_SAMPLE),
        (0, gigl.NODE_IS_SAMPLE),
        (0, gigl.NODE_IS_SAMPLE),  # will have no related material
    ]
    tables = gigl.Tables()
    tables.nodes = make_nodes_table(node_data, tables)
    add_iedge(tables, 0, 10, 0, 10, c=1, p=0, child_chromosome=1, parent_chromosome=0)
    # actually only 0..5 are passed to node 1 from the samples on chr 1
    add_iedge(tables, 0, 3, 5, 8, c=1, p=0, child_chromosome=3, parent_chromosome=0)
    add_iedge(tables, 3, 5, 10, 8, c=1, p=0, child_chromosome=3, parent_chromosome=0)
    add_iedge(tables, 0, 10, 0, 10, c=2, p=0, child_chromosome=5, parent_chromosome=0)
    add_iedge(tables, 0, 5, 0, 5, c=3, p=1, child_chromosome=0, parent_chromosome=1)
    add_iedge(tables, 5, 10, 0, 5, c=3, p=1, child_chromosome=0, parent_chromosome=3)
    add_iedge(tables, 0, 5, 0, 5, c=4, p=1, child_chromosome=0, parent_chromosome=1)
    add_iedge(tables, 5, 10, 0, 5, c=4, p=1, child_chromosome=0, parent_chromosome=3)
    # last one attaches to chr 1 of root, which no others do
    add_iedge(tables, 0, 10, 0, 10, c=5, p=0, child_chromosome=0, parent_chromosome=1)
    return gigl.Graph(tables)


@pytest.fixture(scope="session")
def all_sv_types_no_re_gig():
    """
    Contains a single deletion, a single duplication, and a single inversion.
    See https://github.com/hyanwong/GeneticInheritanceGraphLibrary/issues/2
    """
    # time | flags
    node_data = [
        (6, 0),
        (5, 0),
        (4, 0),
        (4, 0),
        (2, 0),
        (2, 0),
        (2, 0),
        (0, gigl.NODE_IS_SAMPLE),
        (0, gigl.NODE_IS_SAMPLE),
        (0, gigl.NODE_IS_SAMPLE),
        (0, gigl.NODE_IS_SAMPLE),
        (0, gigl.NODE_IS_SAMPLE),
        (0, gigl.NODE_IS_SAMPLE),
    ]
    tables = gigl.Tables()
    tables.nodes = make_nodes_table(node_data, tables)
    add_iedge(tables, 0, 200, 0, 200, c=1, p=0)
    add_iedge(tables, 0, 200, 0, 200, c=2, p=1)
    add_iedge(tables, 0, 200, 0, 200, c=3, p=1)
    add_iedge(tables, 0, 50, 0, 50, c=4, p=2)
    add_iedge(tables, 50, 100, 150, 200, c=4, p=2)
    add_iedge(tables, 0, 200, 0, 200, c=5, p=3)
    add_iedge(tables, 200, 300, 100, 200, c=5, p=3)
    add_iedge(tables, 0, 20, 0, 20, c=6, p=0)
    add_iedge(tables, 20, 120, 120, 20, c=6, p=0)
    add_iedge(tables, 120, 200, 120, 200, c=6, p=0)
    add_iedge(tables, 0, 100, 0, 100, c=7, p=4)
    add_iedge(tables, 0, 100, 0, 100, c=8, p=4)
    add_iedge(tables, 0, 300, 0, 300, c=9, p=5)
    add_iedge(tables, 0, 300, 0, 300, c=10, p=5)
    add_iedge(tables, 0, 200, 0, 200, c=11, p=6)
    add_iedge(tables, 0, 200, 0, 200, c=12, p=6)
    return gigl.Graph(tables)


@pytest.fixture(scope="session")
def all_sv_types_1re_gig():
    """
    A gig with 3 SV types and 1 recombination event: useful for viz
    """
    # time | flags
    node_data = [
        (6, 0),
        (5, 0),
        (4, 0),
        (4, 0),
        (3, 0),
        (3, 0),
        (2, Const.NODE_IS_RE),  # new "RE" node
        (0, gigl.NODE_IS_SAMPLE),
        (0, gigl.NODE_IS_SAMPLE),
        (0, gigl.NODE_IS_SAMPLE),
        (0, gigl.NODE_IS_SAMPLE),
    ]
    tables = gigl.Tables()
    tables.nodes = make_nodes_table(node_data, tables)
    add_iedge(tables, 0, 200, 0, 200, c=1, p=0)
    add_iedge(tables, 0, 200, 0, 200, c=2, p=1)
    add_iedge(tables, 0, 200, 0, 200, c=3, p=1)
    # deletion
    add_iedge(tables, 0, 50, 0, 50, c=4, p=2)
    add_iedge(tables, 50, 100, 150, 200, c=4, p=2)
    # duplication
    add_iedge(tables, 0, 200, 0, 200, c=5, p=3)
    add_iedge(tables, 200, 300, 100, 200, c=5, p=3)
    # recombination combines two SVs
    add_iedge(tables, 0, 70, 0, 70, c=6, p=4)
    add_iedge(tables, 70, 200, 170, 300, c=6, p=5)
    #
    add_iedge(tables, 0, 100, 0, 100, c=7, p=4)  # unrecombined
    add_iedge(tables, 0, 200, 0, 200, c=8, p=6)
    add_iedge(tables, 0, 300, 0, 300, c=9, p=5)  # unrecombined
    # inversion
    add_iedge(tables, 0, 20, 0, 20, c=10, p=0)
    add_iedge(tables, 20, 120, 120, 20, c=10, p=0)
    add_iedge(tables, 120, 200, 120, 200, c=10, p=0)
    return gigl.Graph(tables)


@pytest.fixture(scope="session")
def all_sv_types_2re_gig():
    """
    A gig with 3 SV types and 2 recombination events
    """
    # time | flags
    node_data = [
        (6, 0),
        (5, 0),
        (4, 0),
        (4, 0),
        (3, 0),
        (3, 0),
        (3, 0),
        (2, 0),
        (1, Const.NODE_IS_RE),  # new "RE" node
        (1, Const.NODE_IS_RE),  # new "RE" node
        (0, gigl.NODE_IS_SAMPLE),
        (0, gigl.NODE_IS_SAMPLE),
        (0, gigl.NODE_IS_SAMPLE),
        (0, gigl.NODE_IS_SAMPLE),
        (0, gigl.NODE_IS_SAMPLE),
    ]
    tables = gigl.Tables()
    tables.nodes = make_nodes_table(node_data, tables)
    add_iedge(tables, 0, 200, 0, 200, c=1, p=0)
    add_iedge(tables, 0, 200, 0, 200, c=2, p=1)
    add_iedge(tables, 0, 200, 0, 200, c=3, p=1)
    # deletion
    add_iedge(tables, 0, 50, 0, 50, c=4, p=2)
    add_iedge(tables, 50, 100, 150, 200, c=4, p=2)
    # duplication
    add_iedge(tables, 0, 200, 0, 200, c=5, p=3)
    add_iedge(tables, 200, 300, 100, 200, c=5, p=3)
    # inversion
    add_iedge(tables, 0, 20, 0, 20, c=6, p=0)
    add_iedge(tables, 20, 120, 120, 20, c=6, p=0)
    add_iedge(tables, 120, 200, 120, 200, c=6, p=0)
    #
    # Extra coalescent node for duplication
    add_iedge(tables, 0, 300, 0, 300, c=7, p=5)
    #
    # recombination combines two SVs
    add_iedge(tables, 0, 70, 0, 70, c=8, p=4)
    add_iedge(tables, 70, 200, 170, 300, c=8, p=7)
    #
    # recombination reinstates original coords
    add_iedge(tables, 0, 150, 0, 150, c=9, p=7)
    add_iedge(tables, 150, 200, 150, 200, c=9, p=6)
    #
    add_iedge(tables, 0, 100, 0, 100, c=10, p=4)  # unrecombined
    add_iedge(tables, 0, 200, 0, 200, c=11, p=8)
    add_iedge(tables, 0, 300, 0, 300, c=12, p=5)  # unrecombined
    add_iedge(tables, 0, 200, 0, 200, c=13, p=9)
    add_iedge(tables, 0, 200, 0, 200, c=14, p=6)  # unrecombined
    tables.sort()
    return gigl.Graph(tables)


@pytest.fixture(scope="session")
def gig_from_degree2_ts():
    ts = msprime.sim_ancestry(5, sequence_length=10, recombination_rate=0.02, random_seed=1)
    assert ts.num_trees == 2
    return gigl.Graph.from_tree_sequence(ts)


@pytest.fixture(scope="session")
def degree2_2_tip_ts():
    ts = msprime.sim_ancestry(1, sequence_length=2, recombination_rate=1, random_seed=1)
    assert ts.num_trees == 2
    return ts


@pytest.fixture(scope="session")
def inverted_duplicate_gig():
    """
    A simple GIG with 2 samples (A and B) and a single inverted duplication in A
    """
    A = 0
    B = 1
    # time | flags
    node_data = [
        (0, gigl.NODE_IS_SAMPLE),  # node A
        (0, gigl.NODE_IS_SAMPLE),  # node B
        (1, 0),  # 2
        (2, 0),  # 3
    ]
    tables = gigl.Tables()
    tables.nodes = make_nodes_table(node_data, tables)
    add_iedge(tables, 0, 5, 100, 105, c=A, p=2)  # inherited region at 100, 105 in prnt 2
    add_iedge(tables, 5, 15, 110, 100, c=A, p=2)  # duplicated inversion
    add_iedge(tables, 0, 10, 10, 20, c=B, p=3)
    add_iedge(tables, 90, 190, 0, 100, c=2, p=3)  # an iedge which is not sample-resolved
    tables.sort()
    return gigl.Graph(tables)


@pytest.fixture(scope="session")
def inverted_duplicate_with_missing_gig():
    """
    A simple GIG with 2 samples (A and B) and a single inverted duplication in A
    with a missing (deleted) segment in the middle (equivalent to missing data)
    """
    A = 0
    B = 1
    # time | flags
    node_data = [
        (0, gigl.NODE_IS_SAMPLE),  # node A
        (0, gigl.NODE_IS_SAMPLE),  # node B
        (1, 0),  # 2
        (2, 0),  # 3
    ]
    tables = gigl.Tables()
    tables.nodes = make_nodes_table(node_data, tables)
    add_iedge(tables, 0, 5, 100, 105, c=A, p=2)  # inherited region at 100, 105 in prnt 2
    add_iedge(tables, 25, 35, 110, 100, c=A, p=2)  # duplicated inversion, with missing
    add_iedge(tables, 0, 10, 10, 20, c=B, p=3)
    add_iedge(tables, 90, 190, 0, 100, c=2, p=3)  # an iedge which is not sample-resolved
    tables.sort()
    return gigl.Graph(tables)
