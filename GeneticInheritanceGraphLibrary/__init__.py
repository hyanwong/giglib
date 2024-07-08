import sys
from importlib.metadata import PackageNotFoundError, version

if sys.version_info[0] < 3:  # pragma: no cover
    raise Exception("Python 3 only")

try:
    __version__ = version("GeneticInheritanceGraphLibrary")
except PackageNotFoundError:
    __version__ = "unknown version"

from .constants import (
    Const,
    ValidFlags,  # noqa: F401
)
from .graph import Graph  # noqa: F401
from .tables import Tables  # noqa: F401
from .util import set_print_options  # noqa: F401

NULL = Const.NULL
NODE_IS_SAMPLE = Const.NODE_IS_SAMPLE

_print_options = {"max_lines": 40}
