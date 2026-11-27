import itertools
import types
import warnings

import numpy as np
import tskit


class TskitNameWarning(UserWarning):
    """
    Warning issued when a name that exists in tskit (but not in giglib) is used
    on a giglib object, e.g. ``gig.num_nodes``. The tskit-like behaviour is
    provided, but the warning suggests the giglib equivalent.
    """


def warn_tskit_name(name, suggestion, *, obj="gig", tskit_obj="ts"):
    """
    Warn that ``<obj>.<name>`` does not exist in giglib, and suggest the giglib
    equivalent of the tskit ``<tskit_obj>.<name>``. By default, ``obj`` is a
    :class:`Graph` (``gig``), the equivalent of a tskit ``TreeSequence`` (``ts``).
    """
    warnings.warn(
        f"{obj}.{name} does not exist in giglib: for the equivalent of tskit's {tskit_obj}.{name}, try {suggestion}",
        TskitNameWarning,
        stacklevel=3,
    )


def is_unknown_time(time):
    """
    Test whether a time (or an array of times) is :data:`~giglib.UNKNOWN_TIME`. This is
    needed because ``UNKNOWN_TIME`` is a NaN value, so ``time == UNKNOWN_TIME`` is
    always False.

    :param time: A single time, or an array of times
    :return: A boolean (or an array of booleans)
    """
    return tskit.is_unknown_time(time)


def _iupac_complements():
    pairs = {"A": "T", "C": "G", "R": "Y", "K": "M", "S": "S", "W": "W", "B": "V", "D": "H", "N": "N"}
    complements = {}
    for a, b in pairs.items():
        for x, y in ((a, b), (a.lower(), b.lower())):
            complements[x] = y
            complements[y] = x
    complements["-"] = "-"
    return complements


#: The default map of allele complements, containing the IUPAC nucleotide codes
#: (in upper and lower case) and ``"-"`` (which maps to itself)
DEFAULT_ALLELE_COMPLEMENTS = types.MappingProxyType(_iupac_complements())


def complement_allele(allele, complements):
    """
    Return the allele that would be read on the opposite strand, using a map of
    complements (see :ref:`sec_design_mutations_strands`):

    1. If ``allele`` is a key in ``complements``, return the corresponding value.
    2. Otherwise, if ``allele`` has more than one character, and every character is
       in ``complements``, return its reverse complement.
    3. Otherwise, return ``allele`` unchanged (it is treated as strand-symmetric).

    :param str allele: The allele to complement
    :param dict complements: The map of complements
    :rtype: str
    """
    if allele in complements:
        return complements[allele]
    if len(allele) > 1 and all(c in complements for c in allele):
        return "".join(complements[c] for c in reversed(allele))
    return allele


def as_id(x):
    """
    Convert a single item to an integer ID. ``x`` can either be an integer,
    or an object with an integer ``.id`` attribute (e.g. a :class:`~giglib.graph.Node`).
    """
    if isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)):
        return x
    try:
        x_id = x.id
    except AttributeError:
        raise TypeError(f"Expected an integer ID or an object with an .id attribute, not {x!r}") from None
    return as_id(x_id)


def as_ids(x):
    """
    Convert a collection of items to a numpy array of integer IDs. ``x`` can be
    an array of IDs, a view of objects returned by a :class:`~giglib.Graph` method
    (which has an ``.ids`` attribute), or any iterable of integers or of objects
    with an ``.id`` attribute.
    """
    if hasattr(x, "ids") and not hasattr(x, "id"):
        return x.ids
    if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.integer):
        return x
    return np.array([as_id(i) for i in x], dtype=np.int64)


def truncate_rows(num_rows, limit=None):
    """
    Return a list of indexes into a set of rows, but if a ``limit`` is set, truncate the
    number of rows and place a single ``-1`` entry, instead of the intermediate indexes
    """
    if limit is None or num_rows <= limit:
        return range(num_rows)
    return itertools.chain(
        range(limit // 2),
        [-1],
        range(num_rows - (limit - (limit // 2)), num_rows),
    )


def set_print_options(*, max_lines=40):
    """
    Set the options for printing to strings and HTML

    :param integer max_lines: The maximum number of lines to print from a table, beyond
    this number the middle of the table will be skipped.
    """
    # avoid circular import complaints
    from . import _print_options  # pylint: disable=import-outside-toplevel

    _print_options["max_lines"] = max_lines
