#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# serialize.py

"""
Recursively convert NumPy types to native types and call a serialization method
on objects if available.
"""

from collections import Iterable
import math
import sys

import numpy as np


def serializable(obj, method='serializable', nan_to_num=True, **kwargs):
    """Return a serializable representation of an object, recursively calling
    ``method`` if available and converting NumPy types to native types.

    ``inf``, ``-inf``, and ``NaN`` values are converted by default to a very
    large number, a very small number, and zero, respectively. This can be
    disabled with the ``nan_to_num`` keyword argument.

    If ``obj`` has ``method``, then further keyword arguments are passed to it
    (but they are not passed recursively.)
    """
    # Call the serializable method if available.
    if method is not None and hasattr(obj, method):
        return serializable(getattr(obj, method)(**kwargs))
    # If we have a NumPy array, convert it to a list.
    if isinstance(obj, np.ndarray):
        return np.nan_to_num(obj).tolist()
    # If we have a NumPy integer, convert it to a native integer.
    if isinstance(obj, np.integer):
        return int(obj)
    if nan_to_num:
        # If we have native infinite or NaN values, convert them to numbers.
        if isinstance(obj, float):
            if math.isnan(obj):
                return 0.0
            if obj == float('inf'):
                return sys.float_info.max
            if obj == float('-inf'):
                return sys.float_info.min
    # If we have other NumPy types, convert them to their native equivalents.
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    # Recurse over dictionaries.
    if isinstance(obj, dict):
        return _serializable_dict(obj)
    # Recurse over object dictionaries.
    if hasattr(obj, '__dict__'):
        return _serializable_dict(obj.__dict__)
    # Don't recurse over strings.
    if isinstance(obj, str):
        return obj
    # Recurse over iterables.
    if isinstance(obj, Iterable):
        return [serializable(item) for item in obj]
    # Otherwise, give up and hope it's serializable.
    return obj


def _serializable_dict(dictionary):
    return {key: serializable(value) for key, value in dictionary.items()}
