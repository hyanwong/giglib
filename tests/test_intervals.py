"""
Tests for giglib.intervals, the operations on sets of intervals used when tracing
ancestry, checked against the portion library
"""

import math

import numpy as np
import portion as P
import pytest

from giglib import intervals


def random_intervals(rng, num=4, length=30):
    """A list of up to ``num`` random intervals within 0-``length`` (they may overlap)"""
    result = []
    for _ in range(rng.integers(num + 1)):
        left = int(rng.integers(length))
        result.append((left, left + int(rng.integers(1, 10))))
    return result


def as_portion(interval_list):
    result = P.empty()
    for left, right in interval_list:
        result |= P.closedopen(left, right)
    return result


def as_list(portion_interval):
    return [(i.lower, i.upper) for i in portion_interval if not i.empty]


class TestNormalise:
    def test_merges_overlapping_and_touching(self):
        assert intervals.normalise([(5, 8), (0, 3), (3, 4), (7, 10), (12, 12)]) == [(0, 4), (5, 10)]

    def test_infinite(self):
        assert intervals.normalise([(5, math.inf), (0, 6)]) == [(0, math.inf)]


class TestAgainstPortion:
    @pytest.mark.parametrize("seed", range(100))
    def test_operations(self, seed):
        rng = np.random.default_rng(seed)
        a = intervals.normalise(random_intervals(rng))
        b = intervals.normalise(random_intervals(rng))
        assert a == as_list(as_portion(a))  # normalised as portion would
        assert intervals.union(a, b) == as_list(as_portion(a) | as_portion(b))
        assert intervals.intersect(a, b) == as_list(as_portion(a) & as_portion(b))
        assert intervals.subtract(a, b) == as_list(as_portion(a) - as_portion(b))
        for x in range(-1, 42):
            assert intervals.contains(a, x) == (x in as_portion(a))


class TestPieces:
    def test_example(self):
        u_sets = {"a": [(0, 10)], "b": [(5, 20)]}
        v_sets = {"x": [(0, 15)]}
        assert intervals.pieces(u_sets, v_sets) == [
            (0, 5, {"a"}, {"x"}),
            (5, 10, {"a", "b"}, {"x"}),
            (10, 15, {"b"}, {"x"}),
        ]

    def test_adjacent_pieces_with_the_same_keys_are_joined(self):
        # Two intervals of the same key that touch would be one interval; but two keys
        # that hand over to each other give separate pieces
        assert intervals.pieces({"a": [(0, 10)]}, {"x": [(0, 5)], "y": [(5, 10)]}) == [
            (0, 5, {"a"}, {"x"}),
            (5, 10, {"a"}, {"y"}),
        ]

    @pytest.mark.parametrize("seed", range(100))
    def test_against_each_position(self, seed):
        # Each position covered by both sides is in exactly one piece, with the keys that
        # cover it
        rng = np.random.default_rng(seed)
        u_sets = {key: intervals.normalise(random_intervals(rng)) for key in "ab"}
        v_sets = {key: intervals.normalise(random_intervals(rng)) for key in "xyz"}
        expected = {}
        for x in range(40):
            u_keys = {k for k, ivls in u_sets.items() if any(left <= x < right for left, right in ivls)}
            v_keys = {k for k, ivls in v_sets.items() if any(left <= x < right for left, right in ivls)}
            if u_keys and v_keys:
                expected[x] = (u_keys, v_keys)
        found = {}
        for left, right, u_keys, v_keys in intervals.pieces(u_sets, v_sets):
            for x in range(left, right):
                assert x not in found
                found[x] = (u_keys, v_keys)
        assert found == expected
