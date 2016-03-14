#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_fitness_functions.py

import pytest
import numpy as np

from conftest import p
from pyanimats.utils import unique_rows


@pytest.fixture()
def a():
    return np.array([
        [0, 0, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
    ])


def test_unique_rows_empty():
    result = unique_rows(np.array([[]]))
    answer = np.array([[]])
    p(result, answer)
    assert np.array_equal(result, answer)


def test_unique_rows_one_row():
    result = unique_rows(np.array([[0]]))
    answer = np.array([[0]])
    p(result, answer)
    assert np.array_equal(result, answer)


def test_unique_rows_no_secondary(a):
    result = unique_rows(a)
    answer = np.array([
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 0, 0, 0, 1],
        [0, 0, 1, 1, 1],
    ])
    p(result, answer)
    assert np.array_equal(result, answer)


def test_unique_rows_subset_columns(a):
    result = unique_rows(a, upto=[0, 2, 3])
    answer = np.array([
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
    ])
    p(result, answer)
    assert np.array_equal(result, answer)


def test_unique_rows_one_column(a):
    result = unique_rows(a, upto=[0])
    answer = np.array([
        [0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0],
    ])
    p(result, answer)
    assert np.array_equal(result, answer)


def test_unique_rows_with_counts(a):
    result = unique_rows(a, counts=True)
    answer = (np.array([[1, 1, 0, 0, 0],
                        [0, 0, 1, 1, 1],
                        [1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0]]),
              np.array([3, 2, 2, 1]))
    p(result, answer)
    for r, a in zip(result, answer):
        assert np.array_equal(r, a)


def test_unique_rows_with_indices(a):
    result = unique_rows(a, indices=True)
    answer = (np.array([[1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [1, 0, 0, 0, 1],
                        [0, 0, 1, 1, 1]]),
              np.array([3, 0, 2, 1, 1, 2, 1, 3]))
    p(result, answer)
    for r, a in zip(result, answer):
        assert np.array_equal(r, a)


def test_unique_rows_with_counts_and_indices(a):
    result = unique_rows(a, indices=True, counts=True)
    answer = (np.array([[1, 1, 0, 0, 0],
                        [0, 0, 1, 1, 1],
                        [1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0]]),
              np.array([3, 0, 2, 1, 1, 2, 1, 3]),
              np.array([3, 2, 2, 1]))
    p(result, answer)
    for r, a in zip(result, answer):
        assert np.array_equal(r, a)


def test_unique_rows_with_sort(a):
    result = unique_rows(a, sort=True)
    answer = np.array([[1, 1, 0, 0, 0],
                       [0, 0, 1, 1, 1],
                       [1, 0, 0, 0, 1],
                       [1, 0, 0, 0, 0]])
    p(result, answer)
    assert np.array_equal(result, answer)
