"""Utility helpers used by the training driver."""

import ast


def read_strategy_file(filepath, line_number):
    """
    Parse one line of a scheduler strategy file.

    Each line is expected to be a Python literal of shape ``list[list[int]]``,
    e.g. ``[[0, 1], [2, 3], [96]]``, optionally followed by a ``#`` comment.
    An entry ``[a, b]`` denotes a paired slot (two snapshot groups starting at
    indices ``a`` and ``b``). An entry ``[a]`` denotes a singleton slot.

    Args:
        filepath:    path to the strategy file.
        line_number: 1-indexed line number; different models typically live
                     on different lines of the same file.
    Returns:
        list[list[int]]: the parsed schedule.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    raw = lines[line_number - 1].strip().split('#')[0]
    data = ast.literal_eval(raw)
    return [[int(x) for x in slot] for slot in data]
