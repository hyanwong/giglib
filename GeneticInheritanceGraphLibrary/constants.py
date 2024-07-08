from enum import IntFlag, auto

import tskit


class Const:
    NODE_IS_SAMPLE = tskit.NODE_IS_SAMPLE
    NODE_IS_RE = 1 << 19  # mainly for testing

    #: The default NULL integer value, identical to :data:`tskit:tskit.NULL`
    NULL = tskit.NULL
    ROOTWARDS = 0
    LEAFWARDS = 1


# There are quite a lot of validity flags, so these
# are kept separate from the other constants.
class ValidFlags(IntFlag):
    r"""
    Flags involving validity of the GIG. Those starting with ``IEDGES\_`` are
    specific to the iedges table.
    """

    # GIG DEFINITIONAL REQUIREMENTS

    #: If set, iedge data is guaranteed to be integers
    IEDGES_INTEGERS = auto()

    #: If set, intervals are guaranteed finite and have parent span == child span
    IEDGES_INTERVALS = auto()

    #: IF set, each genomic position in a child's chromosome is guaranteed to have
    #: only one interval above
    IEDGES_FOR_CHILD_NONOVERLAPPING = auto()

    #: If set, all parents are older than their children, guaranteeing the GIG is acyclic
    #: (note this requires knowledge of the times in the nodes table). This flag also
    #: guarantees that nodes with those parent and child IDs exist in the nodes table.
    IEDGES_PARENT_OLDER_THAN_CHILD = auto()

    #: If set, iedges with the same edge_ID are guaranteed to have the same combination
    #: of parentID / childID. Note the inverse may not be true: iedges with the same
    #: parentID / childID can have different edge_IDs.
    IEDGES_SAME_PARENT_CHILD_FOR_EDGE = auto()

    # IEDGE EFFICIENCY REQUIREMENTS (e.g. to define a canonical iedge order)

    #: Set if ``child_left`` < ``child_right`` (inversions can have
    #: ``parent_left`` > ``parent_right``)
    IEDGES_CHILD_INTERVAL_POSITIVE = auto()

    #: Set if all iedges with the same child ID are adjacent in the table
    IEDGES_FOR_CHILD_ADJACENT = auto()

    #: Set if within a set of iedges for the same child, iedges are ordered first
    #: by chromosome number
    IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC = auto()

    #: Set if, within a set of iedges for the same child & chromosome, iedges are ordered
    #: by left position
    IEDGES_FOR_CHILD_SECONDARY_ORDER_LEFT_ASC = auto()

    #: Set if the adjacent rows for a child are ordered in descending order of child
    #: node time. Note that this requires knowledge of the times in the nodes table.
    IEDGES_PRIMARY_ORDER_CHILD_TIME_DESC = auto()

    #: Set if iedges are ordered such that where iedge children are tied at the same
    #: time, iedges with the lowest child ID come first. Note that although this is often
    #: not algorithmically necessary, it helps to creates a canonical iedge ordering.
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
        IEDGES_PARENT_OLDER_THAN_CHILD | IEDGES_PRIMARY_ORDER_CHILD_TIME_DESC | IEDGES_SECONDARY_ORDER_CHILD_ID_ASC
    )

    # These are the requirement that simply involve the iedge table and no linked tables
    IEDGES_COMBO_STANDALONE = IEDGES_ALL & ~IEDGES_COMBO_NODE_TABLE

    # Some algorithms (e.g. iterating over edges for a child / chromosome
    # only require sorting by child ID and within the child
    IEDGES_WITHIN_CHILD_SORTED = (
        IEDGES_FOR_CHILD_ADJACENT | IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC | IEDGES_FOR_CHILD_SECONDARY_ORDER_LEFT_ASC
    )
    # The stricter canonical order of iedge sorting
    IEDGES_SORTED = (
        IEDGES_WITHIN_CHILD_SORTED | IEDGES_PRIMARY_ORDER_CHILD_TIME_DESC | IEDGES_SECONDARY_ORDER_CHILD_ID_ASC
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
