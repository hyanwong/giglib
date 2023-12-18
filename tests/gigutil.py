import GeneticInheritanceGraph as gigl

# Utilities for creating and editing gigs


def make_iedges_table(arr, tables):
    """
    Make an intervals table from a list of tuples.
    """
    for row in arr:
        tables.iedges.add_row(
            parent=row[0],
            child=row[1],
            child_left=row[2],
            child_right=row[3],
            parent_left=row[4],
            parent_right=row[5],
        )
    return tables.iedges


def make_nodes_table(arr, tables):
    """
    Make a nodes table from a list of tuples.
    """
    for row in arr:
        tables.nodes.add_row(time=row[0], flags=row[1], individual=gigl.NULL)
    return tables.nodes
