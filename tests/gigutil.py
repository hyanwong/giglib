import numpy as np

import giglib as gigl
from giglib.tools.homology import random_homologous_positions

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
    depend on how the homology is cut into records. A homologous position that is
    reported more than once (e.g. by overlapping records) fails an assertion, as it
    would otherwise be hidden by the set. ``homology`` can be a list of
    :class:`giglib.Homology` records, or of plain ``(u, v, mrca, mrca_segment)`` tuples
    of the same form (with each segment as ``(left, right, chromosome)``).
    """
    result = set()
    for u, v, mrca, mrca_segment in homology:
        to_v = gigl.PositionMap.from_intervals(u[:2], v[:2])
        to_mrca = gigl.PositionMap.from_intervals(u[:2], mrca_segment[:2])
        for x in range(u[0], u[1]):
            position = ((x, u[2]), (to_v.position(x), v[2]), mrca, (to_mrca.position(x), mrca_segment[2]))
            assert position not in result, f"Homologous position reported twice: {position}"
            result.add(position)
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


def same_homology(found, expected):
    """
    Whether ``found`` and ``expected`` describe the same homology, i.e. the same
    homologous positions, each with the same MRCA and position in it (see
    :func:`homologous_positions`), however it is cut into records
    """
    return homologous_positions(found) == homologous_positions(expected)


def random_tables(seed, *, generations=12, size=5, recombination=0):
    """
    A small random population, from one genome of 20 positions with a random base at each.
    Each child has an inversion, a tandem duplication or a deletion, of up to 5 positions,
    or none, and gains a mutation. These nest, so copies change separately, and inversions
    overlap. With probability ``recombination``, a child instead has two parents, with a
    crossover at a random pair of homologous positions of theirs: if these are in different
    copies of a duplication, the crossover is unequal, and the child's genome has a
    different length from both parents'.
    """
    rng = np.random.default_rng(seed)
    tables = gigl.Tables()
    root = tables.nodes.add_row(generations)
    lengths = {root: 20}
    for x in range(20):
        tables.mutations.add_row(root, x, str(rng.choice(list("ACGT"))))
    population = [root]
    for time in range(generations - 1, -1, -1):
        children = []
        for _ in range(size):
            parent = population[rng.integers(len(population))]
            n = lengths[parent]
            child = tables.nodes.add_row(time, flags=gigl.NODE_IS_SAMPLE if time == 0 else 0)
            # Each piece is (parent, child_left, child_right, parent_left, parent_right)
            pieces = [(parent, 0, n, 0, n)]
            if len(population) > 1 and rng.random() < recombination:
                others = [u for u in population if u != parent]
                other = others[rng.integers(len(others))]
                homology = tables.find_homology(parent, other)
                pair = random_homologous_positions(homology, rng) if homology else None
                if pair is not None and not pair.opposite_orientations:
                    # The start of parent (before x) and the end of other (from y)
                    x, y = pair.u.position, pair.v.position
                    pieces = [(parent, 0, x, 0, x), (other, x, x + lengths[other] - y, y, lengths[other])]
            else:
                length = int(rng.integers(1, min(5, n) + 1))
                start = int(rng.integers(0, n - length + 1))
                end = start + length
                event = rng.choice(["inversion", "duplication", "deletion", "none"])
                if event == "inversion":
                    pieces = [(parent, 0, start, 0, start), (parent, start, end, end, start), (parent, end, n, end, n)]
                elif event == "duplication":
                    pieces = [
                        (parent, 0, end, 0, end),
                        (parent, end, end + length, start, end),
                        (parent, end + length, n + length, end, n),
                    ]
                elif event == "deletion" and length < n:
                    pieces = [(parent, 0, start, 0, start), (parent, start, n - length, end, n)]
            for piece_parent, child_left, child_right, parent_left, parent_right in pieces:
                if child_right > child_left:
                    tables.iedges.add_row(
                        child,
                        piece_parent,
                        child_left=child_left,
                        child_right=child_right,
                        parent_left=parent_left,
                        parent_right=parent_right,
                    )
            lengths[child] = pieces[-1][2]
            tables.mutations.add_row(child, int(rng.integers(lengths[child])), str(rng.choice(list("ACGT"))))
            children.append(child)
        population = children
        tables.validate()  # so that the next generation can find homology between parents
    tables.sort()
    tables.validate()
    return tables
