import itertools

import matplotlib.patches as patches


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


def add_rectangle(ax, X, y, color, height=0.8):
    """
    Plotting utility function for adding rectangles to MRCA plots
    """
    width = X[1] - X[0]
    rect = patches.Rectangle((X[0], y), width, height, facecolor=color)
    ax.add_patch(rect)


def add_triangle(ax, X, y, direction, color, height=0.8):
    """
    Plotting utility function for adding triangles to MRCA plots
    """
    if direction == "right":
        vertices = [(X[0], y), (X[0], y + height), (X[1], y + height / 2)]
    else:
        vertices = [(X[0], y + height / 2), (X[1], y), (X[1], y + height)]

    triangle = patches.Polygon(vertices, closed=True, facecolor=color)
    ax.add_patch(triangle)


def add_row_label(ax, x_offset, y_pos, text, fontsize, color):
    """
    Plotting utility function for adding row labels to MRCA plots
    """
    ax.text(x_offset, y_pos, text, va="center", ha="right", fontsize=fontsize, color=color)
