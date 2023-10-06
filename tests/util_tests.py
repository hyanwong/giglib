import GeneticInheritanceGraph as gig


def make_intervals_table(arr, table_group):
    """
    Make an intervals table from a list of tuples.
    """
    for row in arr:
        table_group.intervals.add_row(
            parent=row[0],
            child=row[1],
            child_left=row[2],
            child_right=row[3],
            parent_left=row[4],
            parent_right=row[5],
        )
    return table_group.intervals


def make_nodes_table(arr, table_group):
    """
    Make a nodes table from a list of tuples.
    """
    for row in arr:
        table_group.nodes.add_row(time=row[0], flags=row[1], individual=gig.NULL)
    return table_group.nodes
