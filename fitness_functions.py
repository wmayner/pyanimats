#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fitness_functions.py

"""
Fitness functions for driving animat evolution.
"""

import textwrap
wrapper = textwrap.TextWrapper(width=80)

from functools import wraps
import math
import numpy as np
from sklearn.metrics import mutual_info_score
import pyphi

from parameters import params


# A registry of available fitness functions
functions = {}
# Mapping from parameter values to descriptive names
LaTeX_NAMES = {
    'mi': 'Mutual\ Information',
    'nat': 'Correct\ Trials',
    'ex': 'Extrinsic\ cause\ information',
    'sp': '\sum\\varphi',
    'bp': '\Phi',
}


def _register(f):
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
        'is exponential, according to the formula F(R) = B^(S*R + A), where '
        'R is one of the “raw” fitness values described above, and where B, '
        'S, A are controlled with the FITNESS_BASE, FITNESS_EXPONENT_SCALE, '
        'and FITNESS_EXPONENT_ADD parameters, respectively.'))
    print('')


# Helper functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _most_common_states(game, n=0):
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
    if not 0 < n <= counts.size:
        n = counts.size
    # Return the (row, count) pairs sorted by count.
    return list(sorted(zip(unique_states, counts), key=lambda x: x[1],
                       reverse=True))[:n]


def _average_over_visited_states(n=0):
    """A decorator that takes an animat and applies a function for every unique
    state the animat visits during a game and returns the average.

    The wrapped function must take an animat, state, and optionally count, and
    return a number.

    The optional parameter ``n`` can be set to consider only the ``n`` most
    common states. Nonpositive ``n`` means all states."""
    def decorator(func):
        @wraps(func)
        def wrapper(ind, **kwargs):
            game = ind.play_game()
            unique_states_and_counts = _most_common_states(game, n=n)
            return np.array([
                func(ind, state, count=count, **kwargs)
                for (state, count) in unique_states_and_counts
            ]).mean()
        return wrapper
    return decorator


POSSIBLE_STATES = [pyphi.convert.loli_index2state(i, params.NUM_NODES)
                   for i in range(2**params.NUM_NODES)]


def _average_over_fixed_states(states=POSSIBLE_STATES):
    """A decorator that takes an animat and applies a function for a fixed set
    of states (defaulting to all possible states) and returns the average.

    The wrapped function must take an animat and a state, and return a
    number."""
    def decorator(func):
        @wraps(func)
        def wrapper(ind, **kwargs):
            return np.array([
                func(ind, state, **kwargs) for state in states
            ]).mean()
        return wrapper
    return decorator


# Natural fitness
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@_register
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
NAT_TO_BIT_CONVERSION_FACTOR = 1 / math.log(2)


@_register
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
    return mi_nats * NAT_TO_BIT_CONVERSION_FACTOR


# Extrinsic cause information
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@_register
@_average_over_visited_states
def ex(ind, state, count=1):
    """Extrinsic cause information: Animats are evaluated based on the sum of φ
    for concepts that are “about” the sensors. This sum is averaged over every
    unique state the animat visits during a game."""
    subsystem = ind.brain_and_sensors(state)

    hidden = subsystem.indices2nodes(params.HIDDEN_INDICES)
    sensors = subsystem.indices2nodes(params.SENSOR_INDICES)

    mechanisms = tuple(pyphi.utils.powerset(hidden))
    purviews = tuple(pyphi.utils.powerset(sensors))

    mice = [subsystem.core_cause(mechanism, purviews=purviews)
            for mechanism in mechanisms]
    return sum(m.phi for m in mice)


# Sum of small-phi
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SENSORS_AND_HIDDEN_POWERSET = tuple(
    pyphi.utils.powerset(params.SENSOR_INDICES + params.HIDDEN_INDICES))
HIDDEN_AND_MOTOR_POWERSET = tuple(
    pyphi.utils.powerset(params.HIDDEN_INDICES + params.MOTOR_INDICES))
SP_STATES = [[1] * i + [0] * (params.NUM_NODES - i)
             for i in range(params.NUM_NODES + 1)]


@_register
@_average_over_fixed_states(states=SP_STATES)
def sp(ind, state, count=1):
    """Sum of φ: Animats are evaluated based on the sum of φ for all the
    concepts of the animat's hidden units, or “brain”. This sum is averaged
    over all possible states of the animat."""
    subsystem = ind.brain(state)
    constellation = pyphi.compute.constellation(
        subsystem,
        past_purviews=SENSORS_AND_HIDDEN_POWERSET,
        future_purviews=HIDDEN_AND_MOTOR_POWERSET)
    return sum(concept.phi for concept in constellation)


# Big phi
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@_register
@_average_over_visited_states
def bp(ind, state, count=1):
    """ϕ: Animats are evaluated based on the ϕ-value of their brains, averaged
    over every unique state the animat visits during a game."""
    subsystem = ind.brain(state)
    return pyphi.compute.big_phi(subsystem)
