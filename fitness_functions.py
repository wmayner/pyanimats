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


# Natural fitness
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@register
def nat(ind):
    """Natural: Animats are evaluated based on the number of game trials they
    successfully complete. Each additional correct trial is weighted
    exponentially higher than the last in order to keep selection pressure more
    even.

    This is accomplished by using the ``NATURAL_FITNESS_BASE`` parameter as the
    base and the number of correct trials as the exponent."""
    ind.play_game()
    return (params.NATURAL_FITNESS_BASE**ind.animat.correct,)


# Mutual information
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def bitlist(i, padlength):
    """Return a list of the bits of an integer, padded up to ``padlength``."""
    return list(map(int, bin(i)[2:].zfill(padlength)))


NUM_SENSOR_STATES = 2**params.NUM_SENSORS
NUM_MOTOR_STATES = 2**params.NUM_MOTORS
SENSOR_MOTOR_STATES = [
    ((i, j), bitlist(i, params.NUM_SENSORS) + bitlist(j, params.NUM_MOTORS))
    for i in range(NUM_SENSOR_STATES) for j in range(NUM_MOTOR_STATES)
]


@register
def mi(ind):
    # TODO implement mutual information
    """Mutual information: Animats are evaluated based on the mutual
    information between their sensors and motors."""
    # Play the game and get the state transitions for each trial.
    game = np.array(ind.play_game())
    # The contingency matrix has a row for every sensors state and a column for
    # every motor state.
    contingency = np.zeros([NUM_SENSOR_STATES, NUM_MOTOR_STATES])
    # Get only the sensor and motor states.
    sensor_motor = np.concatenate([game[:, :, :params.NUM_SENSORS],
                                   game[:, :, -params.NUM_MOTORS:]], axis=2)
    # Count!
    for idx, state in SENSOR_MOTOR_STATES:
        contingency[idx] = (sensor_motor == state).all(axis=2).sum()
    # Calculate mutual information in nats.
    mi_nats = mutual_info_score(None, None, contingency=contingency)
    # Convert from nats to bits and return as a tuple for DEAP.
    return (mi_nats / math.log(2),)
