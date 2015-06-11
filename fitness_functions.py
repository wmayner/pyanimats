#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fitness_functions.py

"""
Fitness functions for driving animat evolution.
"""

import math
import numpy as np
from sklearn.metrics import mutual_info_score

from parameters import params


# A registry of available fitness functions
functions = {}


def register(f):
    """Register a fitness function to the directory."""
    functions[f.__name__] = f.__doc__
    return f


def print_functions():
    """Display a list of available fitness functions with their
    descriptions."""
    print('')
    for name, doc in functions.items():
        print(name + '\n    ' + doc)
        print('')


@register
def natural(ind):
    """Animats are evaluated based on the number of game trials they
    successfully complete. Each additional correct trial is weighted
    exponentially higher than the last in order to keep selection pressure more
    even.

    This is accomplished by using the ``NATURAL_FITNESS_BASE`` parameter as the
    base and the number of correct trials as the exponent."""
    ind.play_game()
    return (params.NATURAL_FITNESS_BASE**ind.animat.correct,)


@register
def mi(ind):
    # TODO implement mutual information
    """Animats are evaluated based on the mutual information between the
    sensors and motors."""
    # Play the game and get the state transitions for each trial.
    game = ind.play_game()
    # The contingency matrix has a row for every sensors state and a column for
    # every motor state.
    c = np.zeros([2**params.NUM_SENSORS, 2**params.NUM_MOTORS])
    # Count!
    for trial in game:
        for state in trial:
            # Get contingency matrix row and column indices from sensor and
            # motor states and increment the count.
            i, j = 0, 0
            for node in state[:params.NUM_SENSORS]:
                i = (i << 1) | node
            for node in state[-params.NUM_MOTORS:]:
                j = (j << 1) | node
            c[i, j] += 1
    # Calculate mutual information in nats.
    mi_nats = mutual_info_score(None, None, contingency=c)
    # Convert from nats to bits and return as a tuple for DEAP.
    return (mi_nats / math.log(2),)
