#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fitness_transforms.py

"""
Classes handling transformation and combination of fitness functions.
"""

from math import sqrt

import numpy as np

from . import utils, fitness_functions


# Theoretical minima and theoretical (or in unbounded cases, practical) maxima
# fitness function values.
DEFAULT_RANGES = {
    'nat':        (64, 128),
    'mi':         (0, 2),
    'mi_wvn':     (0, 1),
    'ex':         (0, 4),
    'ex_wvn':     (0, 1),
    'sp':         (0, 4),
    'sp_wvn':     (0, 1),
    'bp':         (0, 4),
    'bp_wvn':     (0, 4),
    'sd_wvn':     (0, 8),
    'mat':        (0, 1),
}
# We want the exponential fitness function to look roughly like 1.02^X where X
# ranges from 64–128, so after normalizing to 0–1 we need to transform the
# range.
DEFAULT_EXPONENTIAL_TRANSFORM = {
    'base': 1.02,
    'scale': 64,
    'add': 64
}


class ExponentialMultiFitness:

    """
    Combines a primary fitness function with ``N`` others according to the
    function

        F = b^(s*f + a) * (product from i = 1 to N of [(k_i / k_max) + 1)],

    where ``b`` is an arbitrary base, ``f`` is the primary fitness value, ``s``
    and ``a`` are arbitrary constants, and ``k_i`` and ``k_max`` are the actual
    value and theoretical maximum value of the ``i``th fitness function.

    Args:
        function_names (tuple(str)): The names of the functions from
            ``fitness_functions`` to combine. The first function will be used
            as the primary (exponential) value.

    Keyword Args:
        transform (dict): A dictionary containing keys ``base``, ``scale``, and
            ``add``, which are the constants ``B``, ``S``, and ``A`` in the
            formula. Defaults to ``DEFAULT_EXPONENTIAL_TRANSFORM``.
        ranges (list(tuple(float))): A list of pairs giving the theoretical
            minimum and maximum values of each fitness function in
            ``function_names``. Defaults to the those in ``DEFAULT_RANGES``.
    """

    def __init__(self, function_names, transform=None, ranges=None):
        self.transform = transform or DEFAULT_EXPONENTIAL_TRANSFORM
        self.ranges = ranges or [DEFAULT_RANGES[f] for f in function_names]
        self.functions = [fitness_functions.__dict__[f]
                          for f in function_names]
        self.multivalued = [fname in fitness_functions.MULTIVALUED
                            for fname in function_names]
        # Get normalizing functions for each fitness function
        self.norms = [utils.normalizer(self.ranges[i])
                      for i in range(len(function_names))]

    def normalize(self, fitnesses):
        # Get the first value if the fitness is multivalued
        fitnesses = [
            fitnesses[i][0] if self.multivalued[i] else fitnesses[i]
            for i in range(len(fitnesses))
        ]
        # Normalize the values
        return tuple(self.norms[i](f) for i, f in enumerate(fitnesses))

    def __call__(self, ind, **kwargs):
        # TODO: code smell: order matters in fitness eval, since animat.correct
        # is updated each time the game is played, and some fitness functions
        # use the scambled game
        fitnesses = tuple(f(ind, **kwargs) for f in self.functions)
        normalized = np.array(self.normalize(fitnesses))
        exponential = self.transform['base']**(
            normalized[0] * self.transform['scale'] + self.transform['add'])
        combined = exponential * np.product(normalized[1:] + 1)
        return (combined, fitnesses)

    def __repr__(self):
        return 'ExponentialMultiFitness({}, transform={})'.format(
            [f.__name__ for f in self.functions], self.transform)

    def __str__(self):
        return repr(self)


def product(f1, f2, iterations=(1, 1)):
    """Returns the product of a pair of fitness functions.

    The raw values of the fitness functions are transformed according to
    ``constants.FITNESS_TRANSFORMS`` before being multiplied.
    """
    norm1 = utils.normalizer(DEFAULT_RANGES[f1.__name__])
    norm2 = utils.normalizer(DEFAULT_RANGES[f2.__name__])

    def product_func(ind):
        fitness1 = sum([f1(ind) for i in range(iterations[0])]) / iterations[0]
        fitness2 = sum([f2(ind) for i in range(iterations[1])]) / iterations[1]
        normalized1, normalized2 = norm1(fitness1), norm2(fitness2)
        return (sqrt(max(0.0, normalized1 * normalized2)), fitness1, fitness2)

    product_func.__name__ = f1.__name__ + '_' + f2.__name__
    return product_func
