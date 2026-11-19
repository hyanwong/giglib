from enum import IntFlag

import numpy as np
import tskit


class Const:
    # Node flags are uint32, like the node flags column, so that masks such as
    # ``flags & ~NODE_IS_SAMPLE`` work (NumPy 2 refuses a negative Python int here)
    NODE_IS_SAMPLE = np.uint32(tskit.NODE_IS_SAMPLE)
    NODE_IS_RE = np.uint32(1 << 19)  # marks recombination nodes, mainly for testing

    #: The default NULL integer value, identical to :data:`tskit:tskit.NULL`
    NULL = tskit.NULL
    #: The value used for an unknown mutation time, identical to
    #: :data:`tskit:tskit.UNKNOWN_TIME`. As this is a NaN, test for it using
    #: :func:`giglib.is_unknown_time`, not ``==``.
    UNKNOWN_TIME = tskit.UNKNOWN_TIME
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
    IEDGES_INTEGERS = 1

    #: If set, intervals are guaranteed finite and have parent span == child span
    IEDGES_INTERVALS = 1 << 1

    #: IF set, each genomic position in a child's chromosome is guaranteed to have
    #: only one interval above
    IEDGES_FOR_CHILD_NONOVERLAPPING = 1 << 2

    #: If set, all parents are older than their children, guaranteeing the GIG is acyclic
    #: (note this requires knowledge of the times in the nodes table). This flag also
    #: guarantees that nodes with those parent and child IDs exist in the nodes table.
    IEDGES_PARENT_OLDER_THAN_CHILD = 1 << 3

    #: If set, iedges with the same edge_ID are guaranteed to have the same combination
    #: of parentID / childID. Note the inverse may not be true: iedges with the same
    #: parentID / childID can have different edge_IDs.
    IEDGES_SAME_PARENT_CHILD_FOR_EDGE = 1 << 4

    # IEDGE EFFICIENCY REQUIREMENTS (e.g. to define a canonical iedge order)

    #: Set if ``child_left`` < ``child_right`` (inversions can have
    #: ``parent_left`` > ``parent_right``)
    IEDGES_CHILD_INTERVAL_POSITIVE = 1 << 5

    #: Set if all iedges with the same child ID are adjacent in the table
    IEDGES_FOR_CHILD_ADJACENT = 1 << 6

    #: Set if within a set of iedges for the same child, iedges are ordered first
    #: by chromosome number
    IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC = 1 << 7

    #: Set if, within a set of iedges for the same child & chromosome, iedges are ordered
    #: by left position
    IEDGES_FOR_CHILD_SECONDARY_ORDER_LEFT_ASC = 1 << 8

    #: Set if the adjacent rows for a child are ordered in descending order of child
    #: node time. Note that this requires knowledge of the times in the nodes table.
    IEDGES_PRIMARY_ORDER_CHILD_TIME_DESC = 1 << 9

    #: Set if iedges are ordered such that where iedge children are tied at the same
    #: time, iedges with the lowest child ID come first. Note that although this is often
    #: not algorithmically necessary, it helps to creates a canonical iedge ordering.
    IEDGES_SECONDARY_ORDER_CHILD_ID_ASC = 1 << 10

    #: All the iedge requirements
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

    #: All the requirements of a valid GIG, as needed to create a :class:`~giglib.Graph`
    #: (currently the same as :data:`IEDGES_ALL`). See :ref:`sec_validation`.
    GIG = IEDGES_ALL

    # COMBINATIONS OF FLAGS

    #: The requirements that do not simply involve the iedge table, but also need the
    #: times in the node table. This includes the secondary ordering by child ID, as
    #: checking it requires knowing whether a row is at the same time as the previous one.
    IEDGES_COMBO_NODE_TABLE = (
        IEDGES_PARENT_OLDER_THAN_CHILD | IEDGES_PRIMARY_ORDER_CHILD_TIME_DESC | IEDGES_SECONDARY_ORDER_CHILD_ID_ASC
    )

    #: The combination of requirements that simply involve the iedge table and no linked tables
    IEDGES_COMBO_STANDALONE = IEDGES_ALL & ~IEDGES_COMBO_NODE_TABLE

    #: The combination of requirements that identify if edges for a give child are sorted.
    #: Some algorithms (e.g. iterating over edges for a child / chromosome
    #: only require sorting by child ID and within the child
    IEDGES_WITHIN_CHILD_SORTED = (
        IEDGES_FOR_CHILD_ADJACENT | IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC | IEDGES_FOR_CHILD_SECONDARY_ORDER_LEFT_ASC
    )
    #: A stricter canonical order of iedge sorting
    IEDGES_SORTED = (
        IEDGES_WITHIN_CHILD_SORTED | IEDGES_PRIMARY_ORDER_CHILD_TIME_DESC | IEDGES_SECONDARY_ORDER_CHILD_ID_ASC
    )

    #: The requirements guaranteed by the edges of a valid tskit tree sequence, when each
    #: edge becomes an iedge with the same child and parent coordinates. This does not
    #: include :data:`IEDGES_INTEGERS`, as tskit edges can have non-integer coordinates,
    #: or the sorting requirements, as tskit sorts edges differently.
    IEDGES_FROM_TSKIT = (
        IEDGES_INTERVALS
        | IEDGES_FOR_CHILD_NONOVERLAPPING
        | IEDGES_PARENT_OLDER_THAN_CHILD
        | IEDGES_SAME_PARENT_CHILD_FOR_EDGE
        | IEDGES_CHILD_INTERVAL_POSITIVE
    )

    #: The requirements that don't depend on the order of the iedges in the table: they
    #: still hold if iedges are removed or reordered, or trimmed to part of their
    #: interval (at integer positions, trimming the child and parent intervals to
    #: corresponding positions), as long as node times are unchanged. Table algorithms
    #: that build new iedges from old ones in this way (e.g.
    #: :meth:`~giglib.Tables.sample_resolve`) keep these flags.
    IEDGES_ORDER_INDEPENDENT = IEDGES_ALL & ~IEDGES_SORTED

    #: The requirements of algorithms that trace intervals up through the iedges on a
    #: set of tables, such as :meth:`~giglib.Tables.find_mrca_regions` and
    #: :meth:`~giglib.Tables.sample_resolve`: parents older than children, valid
    #: intervals, and the iedges for each child adjacent and ordered by chromosome (so
    #: that the iedges above a node can be found).
    IEDGES_FOR_TRACING = (
        IEDGES_PARENT_OLDER_THAN_CHILD
        | IEDGES_INTERVALS
        | IEDGES_FOR_CHILD_ADJACENT
        | IEDGES_FOR_CHILD_PRIMARY_ORDER_CHR_ASC
    )

    NONE = 0
