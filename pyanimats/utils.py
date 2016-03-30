#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils.py

"""
Utility functions.
"""

import datetime
import os
import subprocess
import sys

import numpy as np

from .__about__ import __version__
from .constants import DAYS, HOURS, MINUTES, WEEKS


def get_version():
    """Return repo description if available, otherwise version number."""
    git_describe = subprocess.run(['git', 'describe'], stdout=subprocess.PIPE)
    return (git_describe.stdout.decode(sys.stdout.encoding).strip()
            if git_describe.returncode == 0 else __version__)


def ensure_exists(path):
    """Makes a path if it doesn't exist and returns it."""
    os.makedirs(path, exist_ok=True)
    return path


def rowset(array, **kwargs):
    """Return the unique rows of an array as a set of tuples."""
    return set(map(tuple, unique_rows(array, **kwargs)))


def contains_row(array, row):
    """Return whether the array contains the row."""
    return (array == row).all(axis=1)


def unique_rows(array, upto=[], indices=False, counts=False, sort=False):
    """Return the unique rows of the last dimension of an array.

    Args:
        array (np.ndarray): The array to consider.

    Keyword Args:
        upto (tuple(int)): Consider uniqueness only up to these row elements.
        indices (bool): Also return the indices of each input row in the array
            of unique rows.
        counts (bool): Also return the row counts (sorted).
        sort (bool): Return the unique rows in descending order by frequency.
    """
    # Cast to np.array if necessary
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    # Return immediately for empty arrays.
    if array.size == 0:
        return array
    # Get the array in 2D form if necessary.
    if array.ndim != 2:
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
    # Return immediately if only the unsorted unique rows are desired.
    if not (counts or indices or sort):
        return unique
    if counts or sort:
        # Get the number of occurences of each unique state (the -1 is needed at
        # the beginning, rather than 0, because of fencepost concerns).
        unq_counts = np.diff(
            np.append(np.insert(diff_idx, 0, -1), sorted_array.shape[0] - 1))
        # Get sorted order.
        sorted_order = np.argsort(unq_counts)
        unique = unique[sorted_order][::-1]
        unq_counts = unq_counts[sorted_order][::-1]
    if not (counts or indices):
        return unique
    secondary_results = []
    if indices:
        unq_idx = np.insert((diff_idx + 1), 0, 0)
        # For each row, get the index of the unique row it maps to.
        unq_idx = np.array([
            np.where(np.append(unq_idx > s, True))[0][0] - 1
            for s in np.argsort(sorted_idx)
        ])
        secondary_results.append(unq_idx)
    if counts:
        secondary_results.append(unq_counts)
    return (unique,) + tuple(secondary_results)


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
        seconds = t.seconds + (t.days * DAYS)
    elif isinstance(t, (float, int)):
        return compress(datetime.timedelta(seconds=t), sign, pad)

    parts = []
    if sign:
        parts.append('-' if t.days < 0 else '+')

    weeks, seconds = divmod(seconds, WEEKS)
    days, seconds = divmod(seconds, DAYS)
    hours, seconds = divmod(seconds, HOURS)
    minutes, seconds = divmod(seconds, MINUTES)

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
