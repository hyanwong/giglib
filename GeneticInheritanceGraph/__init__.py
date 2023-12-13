import sys

import tskit

if sys.version_info[0] < 3:  # pragma: no cover
    raise Exception("Python 3 only")

from .tables import Tables  # noqa: F401
from .util import set_print_options  # noqa: F401

NODE_IS_SAMPLE = tskit.NODE_IS_SAMPLE  # noqa: F401
NULL = tskit.NULL  # noqa: F401

_print_options = {"max_lines": 40}
