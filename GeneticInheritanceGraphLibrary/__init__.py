import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

if sys.version_info[0] < 3:  # pragma: no cover
    raise Exception("Python 3 only")

try:
    __version__ = version("GeneticInheritanceGraphLibrary")  # noqa: F401
except PackageNotFoundError:
    __version__ = "unknown version"  # noqa: F401

from .tables import Tables  # noqa: F401
from .graph import from_tree_sequence, Graph  # noqa: F401
from .util import set_print_options  # noqa: F401
from .constants import Const

NULL = Const.NULL
NODE_IS_SAMPLE = Const.NODE_IS_SAMPLE

_print_options = {"max_lines": 40}
