#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_fitness_functions.py

import pytest
import numpy as np

import fitness_functions as ff
from conftest import p


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


def test_unique_rows(a):
    result = ff.unique_rows(a)
    answer = np.array([
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 0, 0, 0, 1],
        [0, 0, 1, 1, 1],
    ])
    p(result, answer)
    assert np.array_equal(result, answer)


# TODO implement unique_rows tests
def test_unique_rows_subset_columns(a):
    result = ff.unique_rows(a, upto=[0, 2, 3])
    answer = np.array([
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
    ])
    p(result, answer)
    assert np.array_equal(result, answer)


def test_unique_rows_one_column(a):
    result = ff.unique_rows(a, upto=[0])
    answer = np.array([
        [0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0],
    ])
    p(result, answer)
    assert np.array_equal(result, answer)


# TODO implement unique_rows tests
def test_unique_rows_with_counts(a):
    result = ff.unique_rows(a, counts=True)
    answer = [
        (np.array([1, 1, 0, 0, 0]), 3),
        (np.array([1, 0, 0, 0, 1]), 2),
        (np.array([0, 0, 1, 1, 1]), 2),
        (np.array([1, 0, 0, 0, 0]), 1),
    ]
    p(result, answer)
    for r, a in zip(result, answer):
        assert np.array_equal(r[0], a[0])
        assert r[1] == a[1]
