"""
Operations on sets of half-open intervals of positions, as used when tracing ancestry
through the iedges. A set of intervals is a list of ``(left, right)`` tuples, sorted,
with ``left < right``, and neither overlapping nor touching (so each set has a single
form). The right end can be ``math.inf``.

These are deliberately simple: giglib only needs half-open intervals of integers, and
general interval libraries (such as portion) spend most of their time on generality
that isn't needed here (open or closed ends, any type of value, immutable results).
"""

import bisect


def normalise(intervals):
    """
    Return the set of intervals covering the same positions as ``intervals`` (a list of
    ``(left, right)`` tuples in any order, which may overlap, touch, or be empty)
    """
    result = []
    for left, right in sorted(intervals):
        if left >= right:
            continue
        if result and left <= result[-1][1]:
            # Overlaps or touches the previous interval: merge them
            result[-1] = (result[-1][0], max(result[-1][1], right))
        else:
            result.append((left, right))
    return result


def contains(intervals, x):
    """Whether position ``x`` is in the set of intervals ``intervals``"""
    # The last interval starting at or before x
    i = bisect.bisect_right(intervals, (x, float("inf"))) - 1
    return i >= 0 and x < intervals[i][1]


def union(a, b):
    """The positions in either of the sets of intervals ``a`` and ``b``"""
    return normalise(a + b)


def intersect(a, b):
    """The positions in both of the sets of intervals ``a`` and ``b``"""
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        left = max(a[i][0], b[j][0])
        right = min(a[i][1], b[j][1])
        if left < right:
            result.append((left, right))
        # Move on from whichever interval ends first
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return result


def subtract(a, b):
    """The positions in the set of intervals ``a`` but not in ``b``"""
    result = []
    j = 0
    for left, right in a:
        # Skip the intervals of b that end before this interval starts
        while j < len(b) and b[j][1] <= left:
            j += 1
        k = j
        while k < len(b) and b[k][0] < right:
            if b[k][0] > left:
                result.append((left, b[k][0]))
            left = max(left, b[k][1])
            k += 1
        if left < right:
            result.append((left, right))
    return result


def coverage(sets):
    """
    Cut the positions covered by any of the sets of intervals in ``sets`` (a dict mapping
    keys to sets of intervals) into pieces, each covered by the same keys. Adjacent pieces
    covered by the same keys are joined.

    :return: A list of ``(left, right, keys)`` tuples, sorted by position, where ``keys``
        is the (frozen) set of keys whose intervals cover the piece
    """
    # Each interval starts and ends at an "event": sweep through them in order of
    # position, keeping track of which keys cover the current position
    events = []
    for key, intervals in sets.items():
        for left, right in intervals:
            events.append((left, 1, key))
            events.append((right, -1, key))
    events.sort(key=lambda event: event[0])
    covering = set()
    result = []
    i = 0
    while i < len(events):
        position = events[i][0]
        # Apply all the events at this position
        while i < len(events) and events[i][0] == position:
            _, change, key = events[i]
            if change > 0:
                covering.add(key)
            else:
                covering.discard(key)
            i += 1
        if i == len(events) or not covering:
            continue
        next_position = events[i][0]
        keys = frozenset(covering)
        if result and result[-1][1] == position and result[-1][2] == keys:
            result[-1] = (result[-1][0], next_position, keys)
        else:
            result.append((position, next_position, keys))
    return result


def pieces(u_sets, v_sets):
    """
    Cut the positions covered by both some set of intervals in ``u_sets`` and some set in
    ``v_sets`` (dicts mapping keys to sets of intervals) into pieces, each covered by the
    same keys. Adjacent pieces covered by the same keys are joined.

    :return: A list of ``(left, right, u_keys, v_keys)`` tuples, sorted by position,
        where ``u_keys`` and ``v_keys`` are the sets of keys whose intervals cover the
        piece
    """
    # Tag each key with its side (0 for u, 1 for v), so the keys of each side stay apart
    tagged = {(0, key): intervals for key, intervals in u_sets.items()}
    tagged.update({(1, key): intervals for key, intervals in v_sets.items()})
    result = []
    for left, right, keys in coverage(tagged):
        u_keys = frozenset(key for side, key in keys if side == 0)
        v_keys = frozenset(key for side, key in keys if side == 1)
        if u_keys and v_keys:
            result.append((left, right, u_keys, v_keys))
    return result
