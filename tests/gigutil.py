import giglib as gigl

# Utilities for creating and editing gigs


def make_iedges_table(arr, tables):
    """
    Make an intervals table from a list of tuples.
    """
    for row in arr:
        tables.iedges.add_row(
            row[4], row[5], child_left=row[0], child_right=row[1], parent_left=row[2], parent_right=row[3]
        )
    return tables.iedges


def add_iedge(tables, cl, cr, pl, pr, *, c, p, **kwargs):
    """
    A helper function to make an IEdgeTableRow using 'c' and 'p' abbreviations.
    """
    return tables.iedges.add_row(c, p, child_left=cl, child_right=cr, parent_left=pl, parent_right=pr, **kwargs)


def make_nodes_table(arr, tables):
    """
    Make a nodes table from a list of tuples.
    """
    for row in arr:
        tables.nodes.add_row(time=row[0], flags=row[1], individual=gigl.NULL)
    return tables.nodes


def homologous_positions(homology):
    """
    The homology as a set of ``(position in u, position in v, mrca, position in the
    mrca)``, one for each homologous position of u (positions as ``(position,
    chromosome)``). Comparing these, rather than the records, means that tests don't
    depend on how the homology is cut into records. ``homology`` can be a list of
    :class:`giglib.Homology` records, or of plain ``(u, v, mrca, mrca_segment)`` tuples
    of the same form (with each segment as ``(left, right, chromosome)``).
    """
    result = set()
    for u, v, mrca, mrca_segment in homology:
        to_v = gigl.PositionMap.from_intervals(u[:2], v[:2])
        to_mrca = gigl.PositionMap.from_intervals(u[:2], mrca_segment[:2])
        for x in range(u[0], u[1]):
            result.add(((x, u[2]), (to_v.position(x), v[2]), mrca, (to_mrca.position(x), mrca_segment[2])))
    return result


def valid_tables_without_flags():
    """
    Tables that are valid as a GIG, but were added without validation, so no iedge flags
    are set
    """
    tables = gigl.Tables()
    tables.nodes.add_row(2)  # 0
    tables.nodes.add_row(1)  # 1
    tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)  # 2
    tables.nodes.add_row(0, flags=gigl.NODE_IS_SAMPLE)  # 3
    tables.iedges.add_row(1, 0, child_right=10, edge=0)
    tables.iedges.add_row(2, 1, child_right=4, edge=1)
    tables.iedges.add_row(2, 0, child_left=4, child_right=10, parent_left=10, parent_right=4)
    tables.iedges.add_row(3, 1, child_right=10, child_chromosome=1, parent_chromosome=0)
    tables.iedges.unset_bitflag(gigl.ValidFlags.IEDGES_ALL)
    return tables
