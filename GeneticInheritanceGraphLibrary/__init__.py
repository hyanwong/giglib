import sys

if sys.version_info[0] < 3:  # pragma: no cover
    raise Exception("Python 3 only")

from .tables import Tables  # noqa: F401
from .graph import from_tree_sequence, Graph  # noqa: F401
from .util import set_print_options  # noqa: F401
from .constants import NODE_IS_SAMPLE, NULL, ROOTWARDS, LEAFWARDS  # noqa: F401

_print_options = {"max_lines": 40}
