#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fitness_functions.py

"""
Fitness functions for driving animat evolution.
"""

import math
import numpy as np
from sklearn.metrics import mutual_info_score
import pyphi

import textwrap
wrapper = textwrap.TextWrapper(width=80)

from parameters import params


# A registry of available fitness functions
functions = {}
# Mapping from parameter values to descriptive names
LaTeX_NAMES = {
    'mi': 'Mutual\ Information',
    'nat': 'Correct\ Trials'
}


def register(f):
    """Register a fitness function to the directory."""
    functions[f.__name__] = f.__doc__
    return f


def print_functions():
    """Display a list of available fitness functions with their
    descriptions."""
    for name, doc in functions.items():
        print('\n' + name + '\n    ' + doc)
    print('\n' + wrapper.fill(
        'NB: In order to make selection pressure more even, the fitness '
        'function used in the selection algorithm is transformed so that it '
        'is exponential. This is accomplished by using the ``FITNESS_BASE`` '
        'parameter as the base and the fitnesses descibed above as the '
        'exponent.'))
    print('')


# Natural fitness
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@register
def nat(ind):
    """Natural: Animats are evaluated based on the number of game trials they
    successfully complete."""
    ind.play_game()
    return ind.animat.correct


# Mutual information
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _bitlist(i, padlength):
    """Return a list of the bits of an integer, padded up to ``padlength``."""
    return list(map(int, bin(i)[2:].zfill(padlength)))


NUM_SENSOR_STATES = 2**params.NUM_SENSORS
NUM_MOTOR_STATES = 2**params.NUM_MOTORS
SENSOR_MOTOR_STATES = [
    ((i, j), _bitlist(i, params.NUM_SENSORS) + _bitlist(j, params.NUM_MOTORS))
    for i in range(NUM_SENSOR_STATES) for j in range(NUM_MOTOR_STATES)
]
BIT_CONVERSION_FACTOR = math.log(2)


@register
def mi(ind):
    """Mutual information: Animats are evaluated based on the mutual
    information between their sensors and motors."""
    # Play the game and get the state transitions for each trial.
    game = ind.play_game()
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
    # Convert from nats to bits and return.
    return mi_nats / BIT_CONVERSION_FACTOR


# Extrinsic cause information
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _state_counts(game):
    """Return a dictionary pairing animat states with the number of times they
    occured in the game."""
    game = game.reshape(game.shape[0] * game.shape[1], game.shape[2])
    counts = {}
    for state in game:
        key = tuple(state)
        if key not in counts:
            counts[key] = 0
        counts[key] += 1
    return counts


def _sorted_state_counts(game):
    """Return the ``n`` most frequent states given a game history."""
    return sorted(_state_counts(game).items(), key=lambda x: x[1],
                  reverse=True)


def _most_frequent_states(game, n=False):
    counts = _sorted_state_counts(game)
    if not n:
        n = len(counts)
        return list(zip(*counts[:n])[0])


@register
def ex(ind):
    """Extrinsic cause information: Animats are evaluated based on the sum of φ
    for concepts that are “about” the sensors."""
    game = ind.play_game()
    states = _state_counts(game).keys()
    sums = np.empty(len(states))
    for i, state in enumerate(states):
        subsystem = ind.brain_and_sensors(state)

        hidden = subsystem.indices2nodes(params.HIDDEN_INDICES)
        sensors = subsystem.indices2nodes(params.SENSOR_INDICES)

        mechanisms = tuple(pyphi.utils.powerset(hidden))
        purviews = tuple(pyphi.utils.powerset(sensors))

        mice = [subsystem.core_cause(mechanism, purviews=purviews)
                for mechanism in mechanisms]
        sums[i] = sum(m.phi for m in mice)
    return sums.mean()
