#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils.py

"""
Utility functions.
"""

import numpy as np


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
    '''
    Convert the input to compressed format, works with a
    :class:`datetime.timedelta` object or a number that represents the number
    of seconds you want to compress.  If you supply a timestamp or a
    :class:`datetime.datetime` object, it will give the delta relative to the
    current time.
    You can enable showing a sign in front of the compressed format with the
    ``sign`` parameter, the default is not to show signs.
    Optionally, you can chose to pad the output. If you wish your values to be
    separated by spaces, set ``pad`` to ``' '``.
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
    '''

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
