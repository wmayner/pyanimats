#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fitness_functions.py

"""
Fitness functions for driving animat evolution.
"""

import textwrap
wrapper = textwrap.TextWrapper(width=80)

from collections import OrderedDict
from functools import wraps
import math
import numpy as np
from sklearn.metrics import mutual_info_score
import pyphi
from pyphi.convert import nodes2idices as n2i

import config
import constants as _


# A registry of available fitness functions
functions = OrderedDict()
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

# TODO test
def unique_rows(array, n=0, upto=[], counts=False):
    """Return the unique rows of the last dimension of an array.

    Args:
        array (np.ndarray): The array to consider.

    Keyword Args:
        n (int): Return only the ``n`` most common rows.
        upto (tuple(int)): Consider uniqueness only up to these row elements.
        counts (bool): Return the unique rows with their counts (sorted).
        indirect (bool): Return the indices of the rows.
    """
    # Get the array in 2D form.
    array = array.reshape(-1, array.shape[-1])
    # Lexicographically sort.
    sorted_array = array[np.lexsort(array.T), :]
    # Consider only elements of a subset of columns, if provided.
    sorted_pruned = sorted_array[:, upto] if upto else sorted_array
    # Get the indices where a new state appears.
    diff_idx = np.where(np.any(np.diff(sorted_pruned, axis=0), 1))[0]
    # Get the unique rows.
    unique = sorted_array[np.append(diff_idx, -1), :]
    # Return immediately if counts aren't needed.
    if not counts:
        if not n:
            return unique
        else:
            return unique[:n]
    # Get the number of occurences of each unique state (the -1 is needed at
    # the beginning, rather than 0, because of fencepost concerns).
    counts = np.diff(
        np.append(np.insert(diff_idx, 0, -1), sorted_array.shape[0] - 1))
    # Get (row, count) pairs sorted by count.
    sorted_by_count = list(sorted(zip(unique, counts), key=lambda x: x[1],
                                  reverse=True))
    # Return all by default.
    if not 0 < n <= counts.size:
        return sorted_by_count
    # TODO Return (unique, counts) rather than pairs?
    return sorted_by_count[:n]


def _average_over_visited_states(n=0, upto=False):
    """A decorator that takes an animat and applies a function for every unique
    (up to sensors and hidden units only) state the animat visits during a game
    and returns the average.

    The wrapped function must take an animat, state, and optionally count, and
    return a number.

    The optional parameter ``n`` can be set to consider only the ``n`` most
    common states. Nonpositive ``n`` means all states."""
    def decorator(func):
        @wraps(func)
        def wrapper(ind, **kwargs):
            # TODO don't reshape in Individual.play_game (use default parameter)
            # TODO return weighted average? (update docs)
            # TODO don't pass count to func
            game = ind.play_game()
            unique_states = unique_rows(game, n=n, upto=upto)
            return np.array([
                func(ind, state, **kwargs) for state in unique_states
            ]).mean()
        return wrapper
    return decorator


def _average_over_fixed_states(states):
    """A decorator that takes an animat and applies a function for a fixed set
    of states and returns the average.

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


# TODO test
def _most_similar_row(row, array):
    """Return the row in the 2D array most similar to the given row (Hamming
    distance). Interprets everything as binary arrays."""
    return array[np.argmin(np.logical_xor(array, row).sum(axis=1))]


# TODO test
def _get_similar_possible(ind, states):
    """Returns a set of possible states as similar as possible to the given
    set (Hamming distance)."""
    valid_states = []
    invalid_states = []
    for state in states:
        try:
            pyphi.validate.state_reachable(state, ind.network,
                                           constrained_nodes=_.HIDDEN_)
            valid_states.append(state)
        except pyphi.validate.StateUnreachableError:
            invalid_states.append(state)
    if invalid_states:
        possible = unique_rows(ind.tpm)
        for state in invalid_states:
            valid_states.append(_most_similar_row(state, possible))
    return unique_rows(valid_states)


def _average_over_subset_of_possible_states(semifixed_states):
    """A decorator that takes an animat and applies a function for a given set
    of states and returns the average. If any of the given states is not
    possible given the animat's TPM, the most similar possible state is used
    instead (Hamming distance).

    The wrapped function must take an animat and a state, and return a
    number."""
    def decorator(func):
        @wraps(func)
        def wrapper(ind, **kwargs):
            states = _get_similar_possible(ind, semifixed_states)
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
    successfully complete. For each task given in the ``TASKS`` parameter,
    there is one trial per direction (left or right) of block descent, per
    initial animat position (given by ``config.WORLD_WIDTH``)."""
    ind.play_game()
    return ind.animat.correct


# Mutual information
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NAT_TO_BIT_CONVERSION_FACTOR = 1 / math.log(2)


@_register
def mi(ind):
    """Mutual information: Animats are evaluated based on the mutual
    information between their sensors and motor over the course of a game."""
    # Play the game and get the state transitions for each trial.
    game = ind.play_game()
    # The contingency matrix has a row for every sensors state and a column for
    # every motor state.
    contingency = np.zeros([_.NUM_SENSOR_STATES, _.NUM_MOTOR_STATES])
    # Get only the sensor and motor states.
    sensor_motor = np.concatenate([game[:, :, :config.NUM_SENSORS],
                                   game[:, :, -config.NUM_MOTORS:]], axis=2)
    # Count!
    for idx, state in _.SENSOR_MOTOR_STATES:
        contingency[idx] = (sensor_motor == state).all(axis=2).sum()
    # Calculate mutual information in nats.
    mi_nats = mutual_info_score(None, None, contingency=contingency)
    # Convert from nats to bits and return.
    return mi_nats * NAT_TO_BIT_CONVERSION_FACTOR


# Extrinsic cause information
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@_register
@_average_over_visited_states()
def ex(ind, state):
    """Extrinsic cause information: Animats are evaluated based on the sum of φ
    for concepts that are “about” the sensors. This sum is averaged over every
    unique state the animat visits during a game."""
    # TODO generate powerset once (change PyPhi to use indices in find_mice
    # purview restriction)?
    subsystem = ind.brain_and_sensors(state)

    hidden = subsystem.indices2nodes(_.HIDDEN_INDICES)
    sensors = subsystem.indices2nodes(_.SENSOR_INDICES)

    mechanisms = tuple(pyphi.utils.powerset(hidden))
    purviews = tuple(pyphi.utils.powerset(sensors))

    mice = [subsystem.core_cause(mechanism, purviews=purviews)
            for mechanism in mechanisms]
    return sum(m.phi for m in mice)


# Sum of small-phi
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _sp_one_state(ind, state):
    subsystem = ind.as_subsystem(state)
    constellation = pyphi.compute.constellation(
        subsystem,
        mechanisms=_.HIDDEN_POWERSET,
        past_purviews=_.SENSORS_AND_HIDDEN_POWERSET,
        future_purviews=_.HIDDEN_AND_MOTOR_POWERSET)
    return sum(concept.phi for concept in constellation)


@_register
def sp(ind):
    """Sum of φ: Animats are evaluated based on the sum of φ for all the
    concepts of the animat's hidden units, or “brain”, averaged
    over the unique states the animat visits during a game (where uniqueness is
    considered up to the state of the hidden units)."""
    game = ind.play_game()
    unique_states = unique_rows(game, upto=_.HIDDEN_INDICES)
    # Short-circuit for zero connectivity
    if ind.cm.sum() == 0:
        return 0
    return np.array([
        _sp_one_state(ind, state) for state in unique_states
    ]).mean()


# Big-Phi
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TODO cache by TPM?
@_register
def bp(ind):
    """ϕ: Animats are evaluated based on the ϕ-value of their brains, averaged
    over the unique states the animat visits during a game (where uniqueness is
    considered up to the state of the sensors and hidden units)."""
    game = ind.play_game()
    unique_states = unique_rows(game, n=5, upto=_.SENSOR_HIDDEN_INDICES)
    return np.array([
        pyphi.compute.big_phi(ind.brain(state)) for state in unique_states
    ]).mean()


# Matching
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def same_concept(concept, other):
    """Return whether two concepts are equivalent up to φ, mechanism nodes,
    mechanism states, purviews, purview states, and repertoires."""
    return (pyphi.utils.phi_eq(concept.phi, other.phi)
            and n2i(concept.mechanism) == n2i(other.mechanism)
            and ([n.state for n in concept.mechanism] ==
                 [n.state for n in concept.mechanism])
            and n2i(concept.cause.purview) == n2i(other.cause.purview)
            and n2i(concept.effect.purview) == n2i(other.effect.purview)
            and ([n.state for n in concept.cause.purview] ==
                 [n.state for n in other.cause.purview])
            and ([n.state for n in concept.effect.purview] ==
                 [n.state for n in other.effect.purview])
            and concept.eq_repertoires(other))


@_register
def mat(ind, state):
    """Matching: Animats are evaluated based on how well they “match” their
    environment. Roughly speaking, this captures the degree to which their
    conceptual structure “resonates” with statistical regularities in the
    world. This quantity is given by:

        ϕ * (Σφ'(W) - Σφ'(N)),

    where ϕ is just the animat's ϕ-value (averaged over a fixed set of states),
    Σφ'(W) is the sum of φ for each unique concept that the animat obtains when
    presented with a stimulus set from the world, and Σφ'(N) is the same but
    for a stimulus set that has been scrambled first in space and then in
    time."""
