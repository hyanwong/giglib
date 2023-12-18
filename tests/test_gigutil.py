import numpy as np

from .gigutil import DTWF_no_recombination_sim

# Tests for functions in tests/gigutil.py


def test_no_recomb_sim():
    gens = 10
    gig = DTWF_no_recombination_sim(
        num_diploids=10, seq_len=100, generations=gens, random_seed=1
    )
    assert len(np.unique(gig.tables.nodes.time)) == gens + 1
    assert gig.num_iedges > 0
