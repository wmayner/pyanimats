#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fitness_functions.py

"""
Fitness functions for driving animat evolution.
"""

import textwrap
wrapper = textwrap.TextWrapper(width=80)

from collections import OrderedDict, Counter
from functools import wraps
import math
import numpy as np
from sklearn.metrics import mutual_info_score
import pyphi

import config
import constants as _
from utils import unique_rows


# A registry of available fitness functions
functions = OrderedDict()
# Mapping from parameter values to descriptive names
LaTeX_NAMES = {
    'mi': 'Mutual\ Information',
    'nat': 'Correct\ Trials',
    'ex': 'Extrinsic\ cause\ information',
    'sp': '\sum\\varphi',
    'bp': '\Phi',
    'mat': 'Matching'
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

def _average_over_visited_states(shortcircuit=True, upto=False):
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
            # Short-circuit if the animat has no connections.
            if shortcircuit and ind.cm.sum() == 0:
                return 0.0
            game = ind.play_game()
            unique_states = unique_rows(game.animat_states, upto=upto)
            values = [func(ind, state, **kwargs) for state in unique_states]
            return sum(values) / len(values)
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
            values = [func(ind, state, **kwargs) for state in states]
            return sum(values) / len(values)
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
            values = [func(ind, state, **kwargs) for state in states]
            return sum(values) / len(values)
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
    return ind.correct


# Mutual information
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NAT_TO_BIT_CONVERSION_FACTOR = 1 / math.log(2)


@_register
def mi(ind):
    """Mutual information: Animats are evaluated based on the mutual
    information between their sensors and motor over the course of a game."""
    # Play the game and get the state transitions for each trial.
    game = ind.play_game()
    states = game.animat_states
    # The contingency matrix has a row for every sensors state and a column for
    # every motor state.
    contingency = np.zeros([_.NUM_SENSOR_STATES, _.NUM_MOTOR_STATES])
    # Get only the sensor and motor states.
    sensor_motor = np.concatenate([states[:, :, :config.NUM_SENSORS],
                                   states[:, :, -config.NUM_MOTORS:]], axis=2)
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
    # Short-circuit if the animat has no connections.
    if ind.cm.sum() == 0:
        return 0
    # TODO generate powerset once (change PyPhi to use indices in find_mice
    # purview restriction)?
    subsystem = ind.subsystem(state)
    hidden_and_motors = subsystem.indices2nodes(_.HIDDEN_MOTOR_INDICES)
    sensors = subsystem.indices2nodes(_.SENSOR_INDICES)
    mechanisms = tuple(pyphi.utils.powerset(hidden_and_motors))
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
    # Short-circuit if the animat has no connections.
    if ind.cm.sum() == 0:
        return 0
    game = ind.play_game()
    unique_states = unique_rows(game.animat_states, upto=_.HIDDEN_INDICES)
    values = [_sp_one_state(ind, state) for state in unique_states]
    return sum(values) / len(values)


# Big-Phi
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TODO cache by TPM?
@_register
def bp(ind):
    """ϕ: Animats are evaluated based on the ϕ-value of their brains, averaged
    over the 5 most-common unique states the animat visits during a game (where
    uniqueness is considered up to the state of the sensors and hidden
    units)."""
    # Short-circuit if the animat has no connections.
    if ind.cm.sum() == 0:
        return 0
    game = ind.play_game()
    unique_states = unique_rows(game.animat_states,
                                upto=_.SENSOR_HIDDEN_INDICES,
                                sort=True)
    most_frequent = unique_states[:5]
    values = [pyphi.compute.main_complex(ind.network, state).phi
              for state in most_frequent]
    return sum(values) / len(values)


# Matching
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def matching(W, N, constellations):
    # Collect the constellations specified in the world.
    world_constellations = [constellations[state] for state in W]
    # Collect those specified in noise.
    noise_constellations = [constellations[state] for state in N]
    # Join the constellations for every state visited in the world and uniquify
    # the resulting set of concepts. Concepts should be considered the same
    # when they have the same φ, same mechanism, same mechanism state, and the
    # same cause and effect purviews and repertoires.
    world_concepts = set.union(*(C for C in world_constellations))
    # Do the same for noise.
    noise_concepts = set.union(*(C for C in noise_constellations))
    # Calculate and return the final value for matching: the difference in the
    # sum of small phi for the unique concepts specified when presented with
    # the world and that when presented with a scrambled world, weighted by
    # existence in the world.
    return (sum(c.phi for c in world_concepts) -
            sum(c.phi for c in noise_concepts))


def matching_weighted(W, N, constellations, complexes):
    world = np.array([
        sum(complexes[state].phi * c.phi for c in constellations[state])
        for state in W
    ])
    noise = np.array([
        sum(complexes[state].phi * c.phi for c in constellations[state])
        for state in N
    ])
    return world.sum() - noise.sum()


def matching_average_weighted(W, N, constellations, complexes):
    # Collect the constellations specified in the world.
    world_constellations = [constellations[state] for state in W]
    # Collect those specified in noise.
    noise_constellations = [constellations[state] for state in N]
    # Join the constellations for every state visited in the world and uniquify
    # the resulting set of concepts. Concepts should be considered the same
    # when they have the same φ, same mechanism, same mechanism state, and the
    # same cause and effect purviews and repertoires.
    world_concepts = set.union(*(C for C in world_constellations))
    # Do the same for noise.
    noise_concepts = set.union(*(C for C in noise_constellations))
    # Map concepts to the ϕ values.
    big_phis_w = {}
    for state in W:
        for c in constellations[state]:
            if c not in big_phis_w:
                big_phis_w[c] = []
            big_phis_w[c].append(complexes[state].phi)
    big_phis_n = {}
    for state in N:
        for c in constellations[state]:
            if c not in big_phis_n:
                big_phis_n[c] = []
            big_phis_n[c].append(complexes[state].phi)
    # Average the ϕ values.
    big_phis_w = {concept: np.mean(values)
                  for concept, values in big_phis_w.items()}
    big_phis_n = {concept: np.mean(values)
                  for concept, values in big_phis_n.items()}
    return (sum(c.phi * big_phis_w[c] for c in world_concepts) -
            sum(c.phi * big_phis_n[c] for c in noise_concepts))


@_register
def mat(ind):
    """Matching: Animats are evaluated based on how well they “match” their
    environment. Roughly speaking, this captures the degree to which their
    conceptual structure “resonates” with statistical regularities in the
    world. This quantity is given by:

        ϕ * (Σφ'(W) - Σφ'(N)),

    where ϕ is just the animat's ϕ-value (averaged over the 5 most-common
    unique states that it visits during a game), Σφ'(W) is the sum of φ for
    each *unique* concept that the animat obtains when presented with a
    stimulus set from the world, and Σφ'(N) is the same but for a stimulus set
    that has been scrambled first in space and then in time."""
    # Short-circuit if the animat has no connections.
    if ind.cm.sum() == 0:
        return (0, 0, 0)
    # Play the game and a scrambled version of it.
    world = ind.play_game().animat_states
    noise = ind.play_game(scrambled=True).animat_states
    # Since the motor states can't influence φ or ϕ, we set them to zero to
    # make uniqifying the states simpler.
    world[_.MOTOR_INDICES] = 0
    noise[_.MOTOR_INDICES] = 0
    # Get a flat list of all the the states.
    combined = np.concatenate([world, noise])
    combined = combined.reshape(-1, combined.shape[-1])
    # Get unique world and noise states and their counts, up to sensor and
    # hidden states (we care about the sensors since sensor states can
    # influence φ and ϕ as background conditions). The motor states are ignored
    # since now they're all zero.
    all_states = Counter(tuple(state) for state in combined)
    # Get the main complexes for each unique state.
    complexes = {
        state: pyphi.compute.main_complex(ind.network, state)
        for state in all_states
    }
    # TODO weight by frequency?
    # Existence is the mean of the ϕ values.
    big_phis, counts = zip(*[(complexes[state].phi, count)
                             for state, count in all_states.items()])
    existence = np.average(big_phis, weights=counts)
    # Get the unique concepts in each constellation.
    constellations = {
        state: set(bm.unpartitioned_constellation)
        for state, bm in complexes.items()
    }
    # Get the set of unique states in each trial for world and noise.
    world = [set(tuple(state) for state in trial) for trial in world]
    noise = [set(tuple(state) for state in trial) for trial in noise]
    # Now we calculate the matching terms for many stimulus sets (each trial)
    # which are later averaged to obtain the matching value for a “typical”
    # stimulus set.
    # TODO weight each concept by average big phi of its states?
    raw_matching = np.mean([
        matching(W, N, constellations) for W, N in zip(world, noise)
    ])
    raw_matching_weighted = np.mean([
        matching_weighted(W, N, constellations, complexes)
        for W, N in zip(world, noise)
    ])
    raw_matching_average_weighted = np.mean([
        matching_average_weighted(W, N, constellations, complexes)
        for W, N in zip(world, noise)
    ])
    return (existence * raw_matching_average_weighted,
            existence * raw_matching_weighted,
            existence * raw_matching)
