import numpy as np
import pytest

import giglib as gigl
import giglib.util as util

# Tests for functions in util.py


def test_truncate_rows():
    assert list(util.truncate_rows(5, limit=None)) == [0, 1, 2, 3, 4]
    assert list(util.truncate_rows(5, limit=10)) == [0, 1, 2, 3, 4]
    assert list(util.truncate_rows(10, limit=4)) == [0, 1, -1, 8, 9]


def test_set_print_options():
    assert gigl._print_options == {"max_lines": 40}
    util.set_print_options(max_lines=None)
    assert gigl._print_options == {"max_lines": None}
    util.set_print_options(max_lines=50)
    assert gigl._print_options == {"max_lines": 50}
    with pytest.raises(TypeError):
        util.set_print_options(40)


class TestAsIds:
    def test_as_id(self, trivial_gig):
        assert util.as_id(3) == 3
        assert util.as_id(np.int32(3)) == 3
        assert util.as_id(trivial_gig.nodes[1]) == 1
        for bad in ("1", 1.0, True, None):
            with pytest.raises(TypeError):
                util.as_id(bad)

    def test_as_ids(self, trivial_gig):
        samples = trivial_gig.sample_nodes()
        assert util.as_ids(samples) is samples.ids
        arr = np.array([2, 0])
        assert util.as_ids(arr) is arr
        assert list(util.as_ids([2, np.int64(0)])) == [2, 0]
        assert list(util.as_ids(list(samples))) == list(samples.ids)
        assert util.as_ids([]).dtype == np.int64
        with pytest.raises(TypeError):
            util.as_ids([0, "a"])


def test_warn_tskit_name():
    msg = r"gig\.foo does not exist in giglib: for the equivalent of tskit's ts\.foo, try bar"
    with pytest.warns(gigl.TskitNameWarning, match=msg):
        util.warn_tskit_name("foo", "bar")
    msg = r"tables\.foo does not exist in giglib: for the equivalent of tskit's tables\.foo, try bar"
    with pytest.warns(gigl.TskitNameWarning, match=msg):
        util.warn_tskit_name("foo", "bar", obj="tables", tskit_obj="tables")
    assert issubclass(gigl.TskitNameWarning, UserWarning)
