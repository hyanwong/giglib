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
