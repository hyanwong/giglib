import sys

if sys.version_info[0] < 3:
    raise Exception("Python 3 only")

from .tables import NULL
from .tables import TableCollection
