import itertools

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
