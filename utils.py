#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils.py

"""
Utility functions.
"""

import numpy as np


def contains_row(array, row):
    """Return whether the array contains the row."""
    return (array == row).all(axis=1)


# TODO test
def unique_rows(array, upto=[], counts=False):
    """Return the unique rows of the last dimension of an array.

    Args:
        array (np.ndarray): The array to consider.

    Keyword Args:
        n (int): Return only the ``n`` most common rows.
        upto (tuple(int)): Consider uniqueness only up to these row elements.
        counts (bool): Return the unique rows with their counts (sorted).
        indirect (bool): Return the indices of the rows.
    """
    # Get the array in 2D form.
    array = array.reshape(-1, array.shape[-1])
    # Lexicographically sort, considering only elements of a subset of columns,
    # if provided.
    pruned = array[:, upto] if upto else array
    sorted_idx = np.lexsort(pruned.T)
    sorted_array = array[sorted_idx, :]
    # Get the indices where a new row appears.
    sorted_pruned = sorted_array[:, upto] if upto else sorted_array
    diff_idx = np.where(np.any(np.diff(sorted_pruned, axis=0), 1))[0]
    # Get the unique rows.
    unique = sorted_array[np.append(diff_idx, -1), :]
    # Return immediately if counts aren't needed.
    if not counts:
        return unique
    # Get the number of occurences of each unique state (the -1 is needed at
    # the beginning, rather than 0, because of fencepost concerns).
    counts = np.diff(
        np.append(np.insert(diff_idx, 0, -1), sorted_array.shape[0] - 1))
    # Get (row, count) pairs sorted by count.
    sorted_by_count = list(sorted(zip(unique, counts), key=lambda x: x[1],
                                  reverse=True))
    # TODO Return (unique, counts) rather than pairs?
    return sorted_by_count


def signchange(a):
    """Detects sign changes in an array. Doesn't count zero as a separate
    sign.
    >>> signchange(np.array([-2, -1, 0, 1, 2, 3]))
    array([1, 0, 1, 0, 0, 0])
    >>> signchange(np.array([0, -1]))
    array([1, 1])
    >>> signchange(np.array([-1, 0]))
    array([1, 1])
    >>> signchange(np.array([-1, 0, -1]))
    array([0, 1, 1])
    """
    # See http://stackoverflow.com/a/2652425/1085344
    asign = np.sign(a)
    sz = asign == 0
    if sz.any():
        asign[sz] = 1
    return ((np.roll(asign, 1) - asign) != 0).astype(int)


def rolling_window(a, size):
    # See http://stackoverflow.com/a/7100681/1085344
    shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.strides + (a. strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


import datetime

TIME_MINUTE = 60
TIME_HOUR = 3600
TIME_DAY = 86400
TIME_WEEK = 604800


# From natural.date (https://github.com/tehmaze/natural)
def compress(t, sign=False, pad=u''):
    """Convert the input to compressed format.

    Works with a :class:`datetime.timedelta` object or a number that represents
    the number of seconds you want to compress.  If you supply a timestamp or a
    :class:`datetime.datetime` object, it will give the delta relative to the
    current time.
    You can enable showing a sign in front of the compressed format with the
    ``sign`` parameter, the default is not to show signs. Optionally, you can
    chose to pad the output. If you wish your values to be separated by spaces,
    set ``pad`` to ``' '``.

    :param t: seconds or :class:`datetime.timedelta` object
    :param sign: default ``False``
    :param pad: default ``''``
    >>> compress(1)
    '1s'
    >>> compress(12)
    '12s'
    >>> compress(123)
    '2m3s'
    >>> compress(1234)
    '20m34s'
    >>> compress(12345)
    '3h25m45s'
    >>> compress(123456)
    '1d10h17m36s'
    """

    if isinstance(t, datetime.timedelta):
        seconds = t.seconds + (t.days * TIME_DAY)
    elif isinstance(t, (float, int)):
        return compress(datetime.timedelta(seconds=t), sign, pad)

    parts = []
    if sign:
        parts.append('-' if t.days < 0 else '+')

    weeks, seconds = divmod(seconds, TIME_WEEK)
    days, seconds = divmod(seconds, TIME_DAY)
    hours, seconds = divmod(seconds, TIME_HOUR)
    minutes, seconds = divmod(seconds, TIME_MINUTE)

    if weeks:
        parts.append('{}w'.format(weeks))
    if days:
        parts.append('{}d'.format(days))
    if hours:
        parts.append('{}h'.format(hours))
    if minutes:
        parts.append('{}m'.format(minutes))
    if seconds:
        parts.append('{}s'.format(seconds))

    if not parts:
        parts = ['{:.2f}s'.format(t.microseconds / 1000000)]

    return pad.join(parts)
