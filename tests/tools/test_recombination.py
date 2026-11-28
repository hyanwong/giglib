"""
Tests for giglib.tools.recombination: the rules for placing crossovers using homology
"""

import numpy as np

from giglib.tools import recombination
from tests.gigutil import two_samples


def allowed(tables, **kwargs):
    params = {"min_chain_length": 10, "max_gap": 5, **kwargs}
    return recombination.allowed_crossovers(tables, 1, 2, **params)


class TestAllowedCrossovers:
    def test_identical_genomes(self):
        # u and v both inherit the root's 0-20, so a crossover can go anywhere
        tables = two_samples([(0, 20, 0, 20)], [(0, 20, 0, 20)])
        assert allowed(tables) == [(0, 20, 0)]

    def test_offset(self):
        # v has an extra 5 positions at its start, so its positions are u's plus 5
        tables = two_samples([(0, 20, 0, 20)], [(5, 25, 0, 20)])
        assert allowed(tables) == [(0, 20, 5)]

    def test_chain_too_short(self):
        tables = two_samples([(0, 20, 0, 20)], [(0, 20, 0, 20)])
        assert allowed(tables, min_chain_length=21) == []

    def test_not_in_reversed_blocks(self):
        # v has the root's 10-20 inverted: the chain is long enough, but a crossover can
        # only go where u and v are in the same orientation
        tables = two_samples([(0, 30, 0, 30)], [(0, 10, 0, 10), (10, 20, 20, 10), (20, 30, 20, 30)])
        assert allowed(tables, max_gap=10) == [(0, 10, 0), (20, 30, 0)]

    def test_excluded_regions(self):
        # Regions excluded in u, and in v (in v's coordinates, 5 more than u's)
        tables = two_samples([(0, 20, 0, 20)], [(5, 25, 0, 20)])
        assert allowed(tables, u_excluded=[(2, 4)], v_excluded=[(15, 20)]) == [(0, 2, 5), (4, 10, 5), (15, 20, 5)]


class TestRandomCrossover:
    def test_breaks_inside_both_genomes(self):
        rng = np.random.default_rng(1)
        breaks = {recombination.random_crossover([(0, 20, 5)], rng) for _ in range(500)}
        # Each break is between two positions of the stretch, so after its first position
        assert breaks == {(x, x + 5) for x in range(1, 20)}

    def test_uniform_over_all_breaks(self):
        # A stretch of 2 positions has 1 break, and one of 10 positions has 9
        rng = np.random.default_rng(1)
        lefts = [recombination.random_crossover([(0, 2, 0), (100, 110, 0)], rng)[0] for _ in range(1000)]
        assert 0.05 < np.mean([x == 1 for x in lefts]) < 0.15

    def test_nowhere_allowed(self):
        rng = np.random.default_rng(1)
        assert recombination.random_crossover([], rng) is None
        assert recombination.random_crossover([(3, 4, 0)], rng) is None
