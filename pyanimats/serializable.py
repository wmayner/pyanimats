#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# serializable.py

"""
Recursively convert NumPy types to native types and call a serialization method
on objects if available.
"""

from collections import Iterable

import numpy as np


def serializable(obj, method='serializable'):
    """Return a serializable representation of an object, recursively calling
    ``method`` if available and converting NumPy types to native types."""
    # Call the serializable method if available.
    if method is not None and hasattr(obj, method):
        return serializable(getattr(obj, method)())
    # If we have a NumPy array, convert it to a list.
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # If we have NumPy datatypes, convert them to native types.
    if isinstance(obj, np.integer):
        return int(obj)
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
