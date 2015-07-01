#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_fitness_functions.py

import pytest
import numpy as np

import fitness_functions as ff


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
    assert np.array_equal(result, answer)


# TODO implement unique_rows tests
def test_unique_rows_specific_columns(a):
    result = ff.unique_rows(a, upto=[0, 2, 3])
    answer = np.array([
        [1, 0, 0, 0, 1],
        [0, 0, 1, 1, 1],
    ])
    assert np.array_equal(result, answer)


# TODO implement unique_rows tests
def test_unique_rows_top_n():
    pass
