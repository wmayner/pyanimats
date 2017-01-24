#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fitness_functions.py

"""
Fitness functions for driving animat evolution.
"""

import textwrap
from collections import Counter, OrderedDict
from functools import wraps

import numpy as np
import pyphi
from sklearn.metrics import mutual_info_score

from . import constants
from .utils import unique_rows
from c_animat import CORRECT_CATCH, WRONG_CATCH

_WRAPPER_WIDTH = 72
_base_wrapper = textwrap.TextWrapper(width=_WRAPPER_WIDTH)
# Metadata associated with the available fitness functions.
metadata = OrderedDict()
# Mapping from short names to descriptive names
LaTeX_NAMES = {
    'nat': 'Correct\ Trials',
    'mi': 'Mutual\ Information',
    'mi_wvn': 'Mutual\ Information\ (world\ vs.\ noise)',
    'ex': 'Extrinsic\ cause\ information',
    'ex_wvn': 'Extrinsic\ cause\ information\ (world\ vs.\ noise)',
    'sp': '\sum\\varphi',
    'sp_wvn': '\sum\\varphi\ (world\ vs.\ noise)',
    'bp': '\Phi',
    'bp_wvn': '\Phi\ (world\ vs.\ noise)',
    'sd_wvn': 'State\ differentiation (world\ vs.\ noise)',
    'mat': 'Matching'
}
MULTIVALUED = ['mat']
CHEAP = ['nat']


def _register(data_function=None):
    """Register a fitness function to the directory.

    Also associates the function to data-gathering data_functions, if any.
    """
    def wrapper(f):
        metadata[f.__name__] = {'doc': f.__doc__,
                                'data_function': data_function}
    return wrapper


def _docstring_dedent(docstring):
    """Dedents like ``textwrap.dedent`` but ignores the first line."""
    lines = docstring.splitlines()
    return lines[0].strip() + '\n' + textwrap.dedent('\n'.join(lines[1:]))


def _wrap_docstring(docstring, width=_WRAPPER_WIDTH, indent='  '):
    """Wraps a docstring with the given indent and width."""
    wrapper = textwrap.TextWrapper(width=width, initial_indent=indent,
                                   subsequent_indent=indent)
    # Dedent and split into paragraphs
    paragraphs = _docstring_dedent(docstring).split('\n\n')
    return '\n\n'.join(map(wrapper.fill, paragraphs))


# TODO update to describe multifitness
def print_functions():
    """Display a list of available fitness functions."""
    print('\n\n'.join(name + '\n' + _wrap_docstring(data['doc'])
                      for name, data in metadata.items()))
    print('\n' + _base_wrapper.fill(textwrap.dedent("""\
        Note: In order to make selection pressure more even, the fitness
        function used in the selection algorithm is transformed so that it is
        exponential, according to the formula F(R) = B^(S*R + A), where R is
        one of the “raw” fitness values described above, and where B, S, A are
        controlled with the ``experiment.fitness_transform``
        parameters.\n""")))


# Helper functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def shortcircuit_if_empty(value=0.0):
    """Immediately return ``value`` if the animat has no connections."""
    def decorator(func):
        @wraps(func)
        def wrapper(ind, **kwargs):
            if ind.cm.sum() == 0:
                return value
            return func(ind, **kwargs)
        return wrapper
    return decorator


# TODO document kwargs
def avg_over_visited_states(upto_attr=False, transform=False, n=None,
                            scrambled=False, noise_level=None):
    """A decorator that takes an animat and applies a function for every unique
    state the animat visits during a game (up to the given units only) and
    returns the average.

    The wrapped function must take an animat and state, and return a number."""
    def decorator(func):
        @wraps(func)
        def wrapper(ind, **kwargs):
            if noise_level is None:
                noise_level = ind.noise_level
            upto = getattr(ind, upto_attr) if upto_attr else False
            game = ind.play_game(scrambled=scrambled, noise_level=noise_level)
            sort = n is not None
            unique_states = unique_rows(game.animat_states, upto=upto,
                                        sort=sort)[:n]
            values = [func(ind, state, **kwargs) for state in unique_states]
            if transform:
                values = list(map(transform, values))
            return sum(values) / len(values)
        return wrapper
    return decorator


def wvn_trial(world_trial, noise_trial, state_data, transform, reduce,
              upto):
    if upto:
        world_values = [state_data[tuple(state[upto])]
                        for state in world_trial]
        noise_values = [state_data[tuple(state[upto])]
                        for state in noise_trial]
    else:
        world_values = [state_data[tuple(state)] for state in world_trial]
        noise_values = [state_data[tuple(state)] for state in noise_trial]
    # Map the values if necessary.
    if transform:
        world_values = transform(world_values)
        noise_values = transform(noise_values)
    # Reduce the values and take the difference.
    return reduce(world_values) - reduce(noise_values)


def wvn(transform=None, reduce=sum, upto_attr=False, shortcircuit=True,
        shortcircuit_value=0.0, noise_level=None):
    """Compute the world vs. noise difference for a given function.

    Args:
        func: The function with which to compare world and noise. Must take an
            individual and the individual's state.

    Keyword Args:
        transform (function): An optional function with which to transform the
            values of ``func`` for a given trial.
        reduce (function): The world values and noise values are reduced to a
            single value with this function, and the difference between the
            reduced world and noise values is returned. Defaults to the sum of
            the values.
        upto (str): The attribute name that holds the indices up to which
            states will be considered different. Defaults to ``False``, which
            means all indices will be considered.
        shortcircuit (bool): If the animat has no connections, then immediately
            return the ``shortcircuit_value``.
        shortcircuit_value (float): The value to immediately return if
            ``shortcircuit`` is enabled.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(ind, **kwargs):
            upto = getattr(ind, upto_attr) if upto_attr else False
            if noise_level is None:
                noise_level = ind.noise_level
            # Play the game and a scrambled version of it.
            world = ind.play_game(noise_level=noise_level).animat_states
            noise = ind.play_game(scrambled=True,
                                  noise_level=noise_level).animat_states
            # Uniqify all states up to the given indices.
            w_and_n = np.concatenate([world, noise])
            w_and_n = w_and_n.reshape(-1, w_and_n.shape[-1])
            unq_w_and_n = unique_rows(w_and_n, upto=upto)
            # Compute the wrapped function for the unique states. Cast unique
            # states to tuples for hashing. Only include `upto` indices.
            if upto:
                state_data = {
                    tuple(state[upto]): func(ind, tuple(state), **kwargs)
                    for state in unq_w_and_n
                }
            else:
                unq_w_and_n = map(tuple, unq_w_and_n)
                state_data = {state: func(ind, state, **kwargs)
                              for state in unq_w_and_n}
            # Return the mean world vs. noise value over all trials.
            num_trials = world.shape[0]
            return sum(
                wvn_trial(world_trial, noise_trial, state_data, transform,
                          reduce, upto, **kwargs)
                for world_trial, noise_trial in zip(world, noise)
            ) / num_trials
        return wrapper
    return decorator


def unq_concepts(constellations):
    """Takes a list of constellations and returns the set of unique concepts in
    them."""
    return set.union(*(set(C) for C in constellations))


def phi_sum(phi_objects):
    """Takes a list of objects that have a ``phi`` attribute and returns the
    sum of those attributes."""
    return sum(o.phi for o in phi_objects)


# Zero fitness
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def zero(ind):
    """Zero: animats are always given a fitness score of zero."""
    return 0.0
_register()(zero)


# Natural fitness
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def nat(ind, scrambled=False, noise_level=None, iterations=10):
    """Natural: Animats are evaluated based on the number of game trials they
    successfully complete. For each task given in the ``experiment.task``
    parameter, there is one trial per direction (left or right) of block
    descent, per initial animat position (given by
    ``experiment.world_width``)."""
    if noise_level is None:
        noise_level = ind.noise_level
    if noise_level == 0:
        return ind.play_game(scrambled=scrambled,
                             noise_level=noise_level).correct
    return sum([
        ind.play_game(scrambled=scrambled, noise_level=noise_level).correct
        for i in range(iterations)
    ]) / iterations
_register()(nat)


# Mutual information
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def mi(ind, scrambled=False, noise_level=None):
    """Mutual information: Animats are evaluated based on the mutual
    information between their sensors and motor over the course of a game.
    """
    if ind.num_motors == 0:
        return 0.0
    if noise_level is None:
        noise_level = ind.noise_level
    states = ind.play_game(scrambled=scrambled,
                           noise_level=noise_level).animat_states
    # The contingency matrix has a row for every sensor state and a column for
    # every motor state.
    contingency = np.zeros([ind.num_sensor_states, ind.num_motor_states])
    # Get only the sensor and motor states.
    sensor_motor = np.concatenate([states[:, :, :ind.num_sensors],
                                   states[:, :, -ind.num_motors:]], axis=2)
    # Count!
    for idx, state in ind.sensor_motor_states:
        contingency[idx] = (sensor_motor == state).all(axis=2).sum()
    # Calculate mutual information in nats.
    mi_nats = mutual_info_score(None, None, contingency=contingency)
    # Convert from nats to bits and return.
    return mi_nats * constants.NAT_TO_BIT_CONVERSION_FACTOR
_register()(mi)


def mi_wvn(ind):
    """Same as ``mi`` but counting the difference between world and noise."""
    return mi(ind, scrambled=False) - mi(ind, scrambled=True)
_register(data_function=mi)(mi_wvn)


# Extrinsic cause information
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def extrinsic_causes(ind, state):
    """Return the core causes of motors and hidden units whose purviews
    are subsets of the sensors."""
    # TODO generate powerset once (change PyPhi to use indices in find_mice
    # purview restriction)?
    subsystem = ind.as_subsystem(state)
    mechanisms = tuple(pyphi.utils.powerset(ind.hidden_motor_indices))
    purviews = tuple(pyphi.utils.powerset(ind.sensor_indices))
    mice = [subsystem.core_cause(mechanism, purviews=purviews)
            for mechanism in mechanisms]
    return list(filter(lambda m: m.phi > 0, mice))


ex = shortcircuit_if_empty()(
    avg_over_visited_states(transform=phi_sum)(extrinsic_causes))
ex.__name__ = 'ex'
ex.__doc__ = """Extrinsic cause information: Animats are evaluated based on the
    sum of φ for core causes that are “about” the sensors (the purview is a
    subset of the sensors). This sum is averaged over every unique state the
    animat visits during a game."""
_register(data_function=extrinsic_causes)(ex)


ex_wvn = wvn(transform=unq_concepts, reduce=phi_sum,
             upto_attr='hidden_motor_indices')(extrinsic_causes)
ex_wvn.__name__ = 'ex_wvn'
ex_wvn.__doc__ = """Same as ``ex`` but counting the difference between the sum of
    φ of unique concepts that appear in the world and a scrambled version of
    it."""
_register(data_function=extrinsic_causes)(ex_wvn)


# Sum of small-phi
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def all_concepts(ind, state):
    """Return the constellation of all concepts."""
    subsystem = ind.as_subsystem(state)
    return pyphi.compute.constellation(
        subsystem,
        mechanisms=ind.hidden_powerset,
        past_purviews=ind.sensors_and_hidden_powerset,
        future_purviews=ind.hidden_and_motor_powerset)


# The states only need to be considered unique up to the hidden units because
# the subsystem is always the entire network (not the main complex), so there
# are no background conditions.
sp = shortcircuit_if_empty()(
    avg_over_visited_states(transform=phi_sum,
                            upto_attr='hidden_indices')(all_concepts))
sp.__name__ = 'sp'
sp.__doc__ = """Sum of φ: Animats are evaluated based on the sum of φ for all
    the concepts of the animat's hidden units, or “brain”, averaged over the
    unique states the animat visits during a game, where uniqueness is
    considered up to the state of the hidden units (since the entire animat is
    the system, no background conditions need to be considered, and since the
    sensors lack incoming connections and the motors lack outgoing, the only
    possible concepts are therefore those whose mechanisms are a subset of the
    hidden units)."""
_register(data_function=all_concepts)(sp)


sp_wvn = wvn(transform=unq_concepts, reduce=phi_sum,
             upto_attr='hidden_indices')(all_concepts)
sp_wvn.__name__ = 'sp_wvn'
sp_wvn.__doc__ = """Same as ``sp`` but counting the difference between the sum of
    φ of unique concepts that appear in the world and a scrambled version of
    it."""
_register(data_function=all_concepts)(sp_wvn)


# Big-Phi
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main_complex(ind, state):
    """Return the main complex of the individual."""
    return pyphi.compute.main_complex(ind.network, state)

# We compute only the N most-frequent states of those visited for performance
# reasons. Ideally we would consider every unique state.
NUM_BIG_PHI_STATES_TO_COMPUTE = 5

bp = shortcircuit_if_empty()(
    avg_over_visited_states(transform=lambda x: x.phi,
                            upto_attr='sensor_hidden_indices',
                            n=NUM_BIG_PHI_STATES_TO_COMPUTE)(main_complex))
bp.__name__ = 'bp'
bp.__doc__ = """Animats are evaluated based on the ϕ-value of their brains,
    averaged over the {}unique states the animat visits during a game (where
    uniqueness is considered up to the state of the sensors and hidden
    units).""".format(str(NUM_BIG_PHI_STATES_TO_COMPUTE) + ' most-common '
                      if NUM_BIG_PHI_STATES_TO_COMPUTE else '')
_register(data_function=main_complex)(bp)


bp_wvn = shortcircuit_if_empty()(wvn(reduce=phi_sum,
                                 upto_attr='hidden_indices')(main_complex))
bp_wvn.__name__ = 'bp_wvn'
bp_wvn.__doc__ = """Same as ``bp`` but counting the difference between world and
    noise."""
_register(data_function=main_complex)(bp_wvn)


# World vs. noise state differentiation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@shortcircuit_if_empty()
def sd_wvn(ind, upto_attr='hidden_indices', upto=None, noise_iterations=10,
           iterations=10, noise_level=None):
    """State differentiation (world vs. noise): Measures the number of
    hidden-unit states that appear only in the world or only in the scrambled
    world.

    Returns a tuple containing the world vs. scrambled differentiation
    difference, the world differentiation, and the scrambled world
    differentiation.
    """
    if upto is None:
        # Use `upto_attr`
        upto = getattr(ind, upto_attr) if upto_attr else False
    elif not upto:
        # Empty set of elements has 0 differentiation
        return (0, 0, 0)

    if noise_level is None:
        noise_level = ind.noise_level
    if noise_level == 0:
        noise_iterations = 1

    results = np.zeros([noise_iterations, iterations, 2])
    for noise_i in range(noise_iterations):
        unscrambled_game = ind.play_game(noise_level=noise_level)
        world = unscrambled_game.animat_states
        num_trials = world.shape[0]
        for i in range(iterations):
            scrambled = ind.play_game(scrambled=True,
                                      noise_level=noise_level).animat_states
            # Get a permutation of the trials.
            shuffled_trials = list(range(num_trials))
            ind.random.shuffle(shuffled_trials)
            # Take the world vs. scrambled difference with randomly paired
            # trials.
            world_values = [
                len(unique_rows(world[[shuffled_trials[i:i + 2]]],
                                upto=upto))
            ]
            scrambled_values = [
                len(unique_rows(scrambled[[shuffled_trials[i:i + 2]]],
                                upto=upto))
            ]
            results[noise_i][i][:] = [
                sum(world_values) / len(world_values),
                sum(scrambled_values) / len(scrambled_values),
            ]
    results = np.mean(results, axis=(0, 1))
    return (results[0] - results[1], results[0], results[1])
_register(data_function=main_complex)(sd_wvn)


# Matching
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def matching(W, N, constellations):
    # First uniqify states, since that's faster than concepts.
    unq_W, unq_N = set(W), set(N)
    # Collect the constellations specified in the world.
    world_constellations = [constellations[state] for state in unq_W]
    # Collect those specified in noise.
    noise_constellations = [constellations[state] for state in unq_N]
    # Join the constellations for every state visited in the world and uniquify
    # the resulting set of concepts. Concepts should be considered the same
    # when they have the same φ, same mechanism, same mechanism state, and the
    # same cause and effect purviews and repertoires.
    world_concepts = unq_concepts(world_constellations)
    # Do the same for noise.
    noise_concepts = unq_concepts(noise_constellations)
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


def matching_average_weighted(W, N, constellations, complexes,
                              conceptwise=False):
    # First uniqify states, since that's faster than concepts.
    unq_W, unq_N = set(W), set(N)
    # Collect the constellations specified in the world.
    world_constellations = [constellations[state] for state in unq_W]
    # Collect those specified in noise.
    noise_constellations = [constellations[state] for state in unq_N]
    # Join the constellations for every state visited in the world and uniquify
    # the resulting set of concepts. Concepts should be considered the same
    # when they have the same φ, same mechanism, same mechanism state, and the
    # same cause and effect purviews and repertoires.
    world_concepts = unq_concepts(world_constellations)
    # Do the same for noise.
    noise_concepts = unq_concepts(noise_constellations)
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
    # Compute the final matching value.
    matching_value = (
        sum(c.phi * big_phis_w[c] for c in world_concepts) -
        sum(c.phi * big_phis_n[c] for c in noise_concepts)
    )
    if not conceptwise:
        return matching_value
    # If desired, compute the concept-wise contributions to matching.
    conceptwise_contributions = {}
    for c in world_concepts | noise_concepts:
        conceptwise_contributions[c] = (
            c.phi * (big_phis_w.get(c, 0) - big_phis_n.get(c, 0))
        )
    return matching_value, conceptwise_contributions


@shortcircuit_if_empty(value=(0, 0, 0, 0, 0))
def mat(ind, iterations=20, precomputed_complexes=None, noise_level=None,
        noise_iterations=10, conceptwise=False):
    """Matching: Animats are evaluated based on how well they “match” their
    environment. Roughly speaking, this captures the degree to which their
    conceptual structure “resonates” with statistical regularities in the
    world. This quantity is given by

        ϕ * (Σφ'(W) - Σφ'(N)),

    where ϕ is the animat's ϕ-value (averaged over the *unique* states that it
    visits during a game), Σφ'(W) is the sum of φ for each *unique* concept
    that the animat obtains when presented with a stimulus set from the world,
    and Σφ'(N) is the same but for a stimulus set that has been scrambled first
    in space and then in time.
    """
    if noise_level is None:
        noise_level = ind.noise_level
    if noise_level == 0:
        noise_iterations = 1
    # Play the game and a scrambled version of it.
    scrambled = np.array([
        ind.play_game(scrambled=True, noise_level=noise_level).animat_states
        for i in range(noise_iterations)
    ])
    world = np.array([
        ind.play_game(noise_level=noise_level).animat_states
        for i in range(noise_iterations)
    ])
    # Since the motor states can't influence φ or ϕ, we set them to zero to
    # make uniqifying the states simpler.
    world[..., ind.motor_indices] = 0
    scrambled[..., ind.motor_indices] = 0
    # Get a flat list of all the the states.
    combined = np.concatenate([world, scrambled])
    combined = combined.reshape(-1, combined.shape[-1])
    # Get unique world and scrambled states and their counts, up to sensor and
    # hidden states (we care about the sensors since sensor states can
    # influence φ and ϕ as background conditions). The motor states are ignored
    # since now they're all zero.
    all_states = Counter(tuple(state) for state in combined)
    # Get the main complexes for each unique state.
    complexes = precomputed_complexes or {}
    for state in all_states:
        if state not in complexes:
            complexes[state] = pyphi.compute.main_complex(ind.network, state)
    # Existence is the mean of the ϕ values.
    big_phis, counts = zip(*[(complexes[state].phi, count)
                             for state, count in all_states.items()])
    existence = np.average(big_phis, weights=counts)
    # Get the unique concepts in each constellation.
    constellations = {
        state: set(bm.unpartitioned_constellation)
        for state, bm in complexes.items()
    }
    conceptwise_contributions = [[None]*iterations]*noise_iterations
    # Preallocate iteration values.
    raw_matching_vals = np.zeros((noise_iterations, iterations))
    matching_weighted_vals = np.zeros((noise_iterations, iterations))
    matching_average_weighted_vals = np.zeros((noise_iterations, iterations))
    for noise_iteration in range(noise_iterations):
        cur_world = world[noise_iteration]
        cur_scrambled = scrambled[noise_iteration]
        shuffled = list(range(len(cur_world)))
        for iteration in range(iterations):
            # Randomly pair trials to form stimulus sets.
            ind.random.shuffle(shuffled)
            world_stimuli = [
                np.vstack((cur_world[shuffled[i]],
                           cur_world[shuffled[i + 1]]))
                for i in range(0, len(cur_world), 2)
            ]
            scrambled_stimuli = [
                np.vstack((cur_scrambled[shuffled[i]],
                           cur_scrambled[shuffled[i + 1]]))
                for i in range(0, len(cur_scrambled), 2)
            ]
            # Get the states in each stimulus set for cur_world and
            # cur_scrambled.
            world_stimuli = [[tuple(state) for state in stimulus]
                             for stimulus in world_stimuli]
            scrambled_stimuli = [[tuple(state) for state in stimulus]
                                 for stimulus in scrambled_stimuli]
            # Now we calculate the matching terms for many stimulus sets (each
            # pair of trials) which are later averaged to obtain the matching
            # value for a “typical” stimulus set.
            raw_matching_vals[noise_iteration][iteration] = \
                np.mean([
                    matching(W, N, constellations)
                    for W, N in zip(world_stimuli, scrambled_stimuli)
                ])
            matching_weighted_vals[noise_iteration][iteration] = \
                np.mean([
                    matching_weighted(W, N, constellations, complexes)
                    for W, N in zip(world_stimuli, scrambled_stimuli)
                ])
            matching_average_weighted_iter = [
                matching_average_weighted(W, N, constellations, complexes,
                                          conceptwise=conceptwise)
                for W, N in zip(world_stimuli, scrambled_stimuli)
            ]
            if conceptwise:
                values, cwise = zip(*matching_average_weighted_iter)
                conceptwise_contributions[noise_iteration][iteration] = cwise
            else:
                matching_average_weighted_vals[noise_iteration][iteration] = \
                    np.mean(matching_average_weighted_iter)
    raw_matching_mean = raw_matching_vals.mean()
    results = (matching_average_weighted_vals.mean(),
               matching_weighted_vals.mean(),
               existence * raw_matching_mean,
               raw_matching_mean,
               existence)
    if not conceptwise:
        return results
    return results, conceptwise_contributions
_register(data_function=main_complex)(mat)


def food_or_die(ind, init_energy=None, baseline_rate=None,
                activity_penalty=None, block_values=None, noise_level=None):
    """Food: Animats are evaluated based on their ability to obtain energy.
    Some blocks are designated as food (with the hit multiplier in the task
    specification), others are poison. Catching food blocks yields energy;
    catching poison blocks costs energy. The animat uses energy at a baseline
    rate, and hidden/motor unit activity depletes energy faster.

    Parameters:
        0: Initial energy
        1: Baseline consumption rate (per timestep)
        2: Activity penalty (per hidden/motor unit firing)
        3: Energy values per block type
    """
    if noise_level is None:
        noise_level = ind.noise_level

    init_energy = init_energy or ind.function_params[0]
    baseline_rate = baseline_rate or ind.function_params[1]
    activity_penalty = activity_penalty or ind.function_params[2]
    block_values = block_values or ind.function_params[3]

    H = ind.world_height
    num_timesteps = ind.num_trials * H

    game = ind.play_game(noise_level=noise_level)
    animat_states, trial_results = game[0], game[3]

    # Initial energy
    energy = np.empty(num_timesteps)
    energy.fill(init_energy)
    # Cumulative food reward
    num_timesteps_per_block = H * len(trial_results) / len(block_values)
    block_values = np.concatenate(
        [np.full(int(num_timesteps_per_block), val, int)
         for val in block_values]
    )
    catches = H * np.where(np.logical_or(trial_results == CORRECT_CATCH,
                                         trial_results == WRONG_CATCH))[0]
    food = np.zeros(num_timesteps)
    food[catches] = 1
    food *= block_values
    food = np.cumsum(food)
    # Cumulative baseline consumption
    baseline = np.arange(0, num_timesteps*baseline_rate, baseline_rate)
    # Cumulative activity penalty
    activity = animat_states.astype(float).reshape(-1, animat_states.shape[-1])
    activity = activity[:, list(ind.hidden_motor_indices)].sum(1)
    activity *= activity_penalty
    activity = np.cumsum(activity)

    energy += (food - baseline - activity)

    if np.any(energy <= 0):
        return 0
    return energy.mean()
_register()(food_or_die)


def food(ind, baseline_penalty=None, activity_penalty=None, block_values=None,
         scrambled=False, noise_level=None, iterations=10):
    """Food: Animats are evaluated based on their ability to obtain energy.
    Some blocks are designated as food (with the hit multiplier in the task
    specification), others are poison. Catching food blocks yields energy;
    catching poison blocks costs energy. The animat uses energy at a baseline
    rate, and hidden/motor unit activity depletes energy faster.

    Parameters:
        0: Baseline penalty
        1: Activity penalty (per hidden/motor unit firing)
        2: Energy values per block type
    """
    if noise_level is None:
        noise_level = ind.noise_level
    if noise_level == 0:
        iterations = 1

    baseline_penalty = baseline_penalty or ind.function_params[0]
    activity_penalty = activity_penalty or ind.function_params[1]
    block_values = block_values or ind.function_params[2]

    num_trials_per_block = int(ind.num_trials / len(block_values))
    block_values = np.concatenate([np.full(num_trials_per_block, val, int)
                                   for val in block_values])

    scores = np.zeros(iterations)
    for i in range(iterations):

        game = ind.play_game(scrambled=scrambled,
                             noise_level=noise_level)

        # Cumulative block consumption
        food = np.zeros(ind.num_trials)
        catches = np.where(np.logical_or(game.trial_results == CORRECT_CATCH,
                                         game.trial_results == WRONG_CATCH))[0]
        food[catches] = 1
        food = np.sum(food * block_values)
        # Cumulative activity penalty
        total_activity_penalty = activity_penalty * np.sum(game.animat_states)

        scores[i] = sum([food, baseline_penalty, total_activity_penalty])

    return sum(scores) / iterations
_register()(food)
