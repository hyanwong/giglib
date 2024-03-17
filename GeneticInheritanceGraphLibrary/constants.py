from enum import auto
from enum import IntFlag

import tskit


class Const(IntFlag):
    NODE_IS_SAMPLE = tskit.NODE_IS_SAMPLE
    NODE_IS_RE = 1 << 19  # mainly for testing
    NULL = tskit.NULL
    ROOTWARDS = 0
    LEAFWARDS = 1


class ValidFlags(IntFlag):
    """
    Flags involving validity of the GIG. There are quite a lot of these, so they are
    kept separate from the other constants.
    """

    # GIG DEFINITIONAL REQUIREMENTS

    IEDGES_INTEGERS = auto()

    # Most important: valid finite nonzero intervals with parent span == child span
    IEDGES_INTERVALS = auto()

    # Second most important: each chrom position in a child has only one interval above
    IEDGES_FOR_CHILD_NONOVERLAPPING = auto()

    # Ensure the graph is acyclic (requires knowledge of the times in the nodes table)
    # Also useful for guaranteeing that there are nodes with the parent and child IDs
    IEDGES_PARENT_OLDER_THAN_CHILD = auto()

    # Iedges with the same edge_ID must have the same combination of parentID / childID
    # (note the inverse may not be true: iedges with the same parentID / childID can have
    # different edge_IDs)
    IEDGES_SAME_PARENT_CHILD_FOR_EDGE = auto()

    # IEDGE EFFICIENCY REQUIREMENTS (e.g. to define a canonical iedge order)

    # keep child_left < child_right (but inversions will have parent_left > parent_right)
    IEDGES_CHILD_INTERVAL_POSITIVE = auto()

    # all iedges with the same child ID are adjacent in the table
    IEDGES_FOR_CHILD_ADJACENT = auto()

    # within a set of iedges for the same child, iedges are ordered first
    # by chromosome number
    IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC = auto()

    # within a set of iedges for the same child & chromosome, iedges are ordered by left
    # position (if IEDGES_FOR_CHILD_NONOVERLAPPING and IEDGES_CHILD_INTERVAL_POSITIVE are
    # set, then this means that iedges must also be ordered by right position)
    IEDGES_FOR_CHILD_SECONDARY_ORDER_LEFT_ASC = auto()

    # the adjacent rows for a child are ordered in descending order of child node time
    # (requires knowledge of the times in the nodes table)
    IEDGES_PRIMARY_ORDER_CHILD_TIME_DESC = auto()

    # ties (child nodes at the same time) are broken by putting the lowest child ID first
    # (note: probably not very important, but creates a canonical ordering; it also
    # implies IEDGES_FOR_CHILD_ADJACENT must be set)
    IEDGES_SECONDARY_ORDER_CHILD_ID_ASC = auto()

    IEDGES_ALL = (
        IEDGES_INTEGERS
        | IEDGES_INTERVALS
        | IEDGES_FOR_CHILD_NONOVERLAPPING
        | IEDGES_PARENT_OLDER_THAN_CHILD
        | IEDGES_CHILD_INTERVAL_POSITIVE
        | IEDGES_SAME_PARENT_CHILD_FOR_EDGE
        | IEDGES_FOR_CHILD_ADJACENT
        | IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC
        | IEDGES_FOR_CHILD_SECONDARY_ORDER_LEFT_ASC
        | IEDGES_PRIMARY_ORDER_CHILD_TIME_DESC
        | IEDGES_SECONDARY_ORDER_CHILD_ID_ASC
    )

    GIG = IEDGES_ALL

    # COMBINATIONS OF FLAGS

    # These are the requirements that do not simply involve the iedge table
    # (e.g. they also involve knowing node table information). We need to include the
    # secondary order sorting by child ID here because we can't check whether this is
    # satisfied without knowing whether a row is at the same timepoint as a previous one
    IEDGES_COMBO_NODE_TABLE = (
        IEDGES_PARENT_OLDER_THAN_CHILD
        | IEDGES_PRIMARY_ORDER_CHILD_TIME_DESC
        | IEDGES_SECONDARY_ORDER_CHILD_ID_ASC
    )

    # These are the requirement that simply involve the iedge table and no linked tables
    IEDGES_COMBO_STANDALONE = IEDGES_ALL & ~IEDGES_COMBO_NODE_TABLE

    # Some algorithms (e.g. iterating over edges for a child / chromosome
    # only require sorting by child ID and within the child
    IEDGES_WITHIN_CHILD_SORTED = (
        IEDGES_FOR_CHILD_ADJACENT
        | IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC
        | IEDGES_FOR_CHILD_SECONDARY_ORDER_LEFT_ASC
    )
    # The stricter canonical order of iedge sorting
    IEDGES_SORTED = (
        IEDGES_WITHIN_CHILD_SORTED
        | IEDGES_PRIMARY_ORDER_CHILD_TIME_DESC
        | IEDGES_SECONDARY_ORDER_CHILD_ID_ASC
    )

    NONE = 0

    @classmethod
    def iedges_combo_standalone_iter(cls):
        # could tidy this in python 3.11 by using the builtin iterators
        yield cls.IEDGES_INTEGERS
        yield cls.IEDGES_INTERVALS
        yield cls.IEDGES_FOR_CHILD_NONOVERLAPPING
        yield cls.IEDGES_CHILD_INTERVAL_POSITIVE
        yield cls.IEDGES_SAME_PARENT_CHILD_FOR_EDGE
        yield cls.IEDGES_FOR_CHILD_ADJACENT
        yield cls.IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC
        yield cls.IEDGES_FOR_CHILD_SECONDARY_ORDER_LEFT_ASC
