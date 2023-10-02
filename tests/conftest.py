import msprime
import pytest


@pytest.fixture(scope="session")
def simple_ts():
    return msprime.sim_ancestry(
        2, recombination_rate=1, sequence_length=10, random_seed=1
    )
