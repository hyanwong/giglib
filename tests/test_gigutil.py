import numpy as np

from .gigutil import DTWF_no_recombination_sim
from .gigutil import DTWF_one_recombination_no_SV_slow_sim

# Tests for functions in tests/gigutil.py


def test_no_recomb_sim():
    gens = 10
    simulator = DTWF_no_recombination_sim()
    gig = simulator.run(num_diploids=10, seq_len=100, generations=gens, random_seed=1)
    assert len(np.unique(gig.tables.nodes.time)) == gens + 1
    assert gig.num_iedges > 0


def test_one_recomb_sim():
    gens = 10
    simulator = DTWF_one_recombination_no_SV_slow_sim()
    gig = simulator.run(num_diploids=10, seq_len=100, generations=gens, random_seed=1)
    assert len(np.unique(gig.tables.nodes.time)) == gens + 1
    assert gig.num_iedges > 0
    print(gig.tables)
    ts = gig.to_tree_sequence()
    assert ts.num_samples == len(gig.samples)
