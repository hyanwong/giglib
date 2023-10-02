import sys

if sys.version_info[0] < 3:
    raise Exception("Python 3 only")

from .tables import NULL  # noqa: F401
from .tables import TableGroup  # noqa: F401
from .util import set_print_options  # noqa: F401

_print_options = {"max_lines": 40}
