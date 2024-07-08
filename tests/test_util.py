import pytest

import GeneticInheritanceGraphLibrary as gigl
import GeneticInheritanceGraphLibrary.util as util

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
