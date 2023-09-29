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

def set_print_options(*, max_lines=40):
    """
    Set the options for printing to strings and HTML

    :param integer max_lines: The maximum number of lines to print from a table, beyond
    this number the middle of the table will be skipped.
    """
    # avoid circular import complaints
    from . import _print_options  # pylint: disable=import-outside-toplevel
    _print_options["max_lines"] = max_lines
