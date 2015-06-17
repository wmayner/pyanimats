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

def _most_common_states(game, n=False):
    # Get the array in 2D form.
    game = game.reshape(-1, game.shape[-1])
    # Lexicographically sort.
    sorted_game = game[np.lexsort(game.T), :]
    # Get the indices where a new state appears.
    diff_idx = np.where(np.any(np.diff(sorted_game, axis=0), 1))[0]
    # Get the unique states.
    unique_states = [sorted_game[i] for i in diff_idx] + [sorted_game[-1]]
    # Get the number of occurences of each unique state (the -1 is needed at
    # the beginning, rather than 0, because of fencepost concerns).
    counts = np.diff(np.insert(diff_idx, 0, -1))
    # Return all by default.
    if n is False or n > counts.size:
        n = counts.size
    # Return the (row, count) pairs sorted by count.
    return list(sorted(zip(unique_states, counts), key=lambda x: x[1],
                       reverse=True))[:n]


@register
def ex(ind):
    """Extrinsic cause information: Animats are evaluated based on the sum of φ
    for concepts that are “about” the sensors."""
    game = ind.play_game()
    unique_states_and_counts = _most_common_states(game)
    sums = np.empty(len(unique_states_and_counts))
    for i, (state, count) in enumerate(unique_states_and_counts):
        subsystem = ind.brain_and_sensors(state)

        hidden = subsystem.indices2nodes(params.HIDDEN_INDICES)
        sensors = subsystem.indices2nodes(params.SENSOR_INDICES)

        mechanisms = tuple(pyphi.utils.powerset(hidden))
        purviews = tuple(pyphi.utils.powerset(sensors))

        mice = [subsystem.core_cause(mechanism, purviews=purviews)
                for mechanism in mechanisms]
        sums[i] = sum(m.phi for m in mice)
    return sums.mean()
