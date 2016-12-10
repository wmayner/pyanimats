#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# analyze.py

from collections import namedtuple
from collections import Counter

import numpy as np
import pyphi

from .fitness_functions import unq_concepts

Game = namedtuple('Game', ['animat_states', 'world_states', 'animat_positions',
                           'trial_results', 'correct', 'incorrect'])


def next_state(tpm, state):
    return np.copy(tpm[tuple(state)]).astype(int)


def simulate(sensors, tpm, stimuli, initial_state=False, given=False):
    """Present the animat with the given stimuli and return its states.

    Args:
        sensors (Iterable): The indices of the sensor units.
        tpm (np.array): The TPM to simulate.
        stimuli (Iterable): An iterable of sensor states to present to the
            animat.

    Keyword Args:
        initial_state (Iterable): The initial state of the animat when
            simulation begins.
        given (Iterable): If supplied, then the animat's non-sensor units are
            set to the given states after each stimulus.
    """
    N = tpm.shape[-1]
    sensors = list(sensors)
    if not initial_state:
        initial_state = [0]*N
    cs = np.array(initial_state)
    cs[sensors] = stimuli[0][sensors]
    animat_states = [cs]
    for stimulus in stimuli[1:]:
        cs = next_state(tpm, cs)
        cs[sensors] = stimulus[sensors]
        animat_states.append(cs)
    return np.array(animat_states)


def gen_world(pattern, h):
    pattern = np.array([int(cell) for cell in pattern])
    return np.array([
        np.roll(pattern, timestep) for timestep in range(h)
    ])


def play_game(experiment, tpm, scrambled=False):
    num_nodes = tpm.shape[-1]
    num_trials = len(experiment.task) * 2 * experiment.world_width
    animat_states = np.zeros([num_trials, experiment.world_height, num_nodes])
    world_states = np.zeros([num_trials, experiment.world_height, experiment.world_width])
    # animat_positions = np.zeros([num_trials, experiment.world_height])
    animat_positions = False
    # trial_results = np.zeros(num_trials)
    trial_results = False
    correct = 0
    incorrect = 0
    cur_trial = 0

    # Trial conditions
    for hit_mult, pattern in experiment.task:
        pattern = pattern.replace('_', '0')
        for direction in [-1, 1]:
            for init_pos in range(experiment.world_width):
                pos = init_pos
                world = gen_world(pattern, experiment.world_height)
                if scrambled:
                    # Space
                    for timestep in range(len(world)):
                        np.random.shuffle(world[timestep])
                    # Time
                    np.random.shuffle(world)
                # TODO extend to allow motors
                animat_states[cur_trial] = simulate(
                    tuple(range(0, experiment.num_sensors)), tpm, world,
                    initial_state=[0]*num_nodes
                )

                animat_mask = np.array(
                    [1, 1, 1] + [0] * (experiment.world_width - 3)
                )
                animat_mask = np.roll(animat_mask, pos)
                hit = np.any(np.logical_and(animat_mask, world[-1]))
                if hit_mult > 0:
                    if hit:
                        correct += 1
                    else:
                        incorrect += 1
                else:
                    if hit:
                        incorrect += 1
                    else:
                        correct += 1

                world_states[cur_trial] = world
                cur_trial += 1
    return Game(animat_states, world_states, animat_positions, trial_results,
                correct, incorrect)


def game_to_json(ind, gen, scrambled=False):
    # Get the full configuration dictionary, including derived constants.
    config = configure.get_dict(full=True)
    # Play the game.
    game = ind.play_game(scrambled=scrambled)
    # Convert world states from the integer encoding to explicit arrays.
    world_states = np.array(
        list(map(lambda i: i2s(i, config['WORLD_WIDTH']),
                 game.world_states.flatten().tolist()))).reshape(
                     game.world_states.shape + (config['WORLD_WIDTH'],))
    phi_data = get_phi_data(ind, game, config)
    # Generate the JSON-encodable dictionary.
    json_dict = {
        'config': config,
        'generation': gen,
        'genome': ind.genome,
        'cm': ind.cm.tolist(),
        'correct': ind.correct,
        'incorrect': ind.incorrect,
        'mechanisms': {i: ind.mechanism(i, separate_on_off=True)
                       for i in range(config['NUM_NODES'])},
        'trials': [
            {
                'num': trialnum,
                # First result bit is whether the block was caught.
                'catch': bool(game.trial_results[trialnum] & 1),
                # Second result bit is whether the animat was correct.
                'correct': bool((game.trial_results[trialnum] >> 1) & 1),
                'timesteps': [
                    {
                        'num': t,
                        'world': world_states[trialnum, t].tolist(),
                        'animat': game.animat_states[trialnum, t].tolist(),
                        'pos': game.animat_positions[trialnum, t].tolist(),
                        'phidata': ((
                            phi_data[tuple(game.animat_states[trialnum, t])] if
                            tuple(game.animat_states[trialnum, t]) in phi_data
                            else False
                        ) if phi_data else False)
                    }
                    for t, world_state in enumerate(world_states[trialnum])
                ],
            } for trialnum in range(game.animat_states.shape[0])
        ],
    }
    assert(ind.correct == sum(trial['correct'] for trial in
                              json_dict['trials']))
    return json_dict

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


def matching_average_weighted(W, N, constellations, complexes):
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
    return (sum(c.phi * big_phis_w[c] for c in world_concepts) -
            sum(c.phi * big_phis_n[c] for c in noise_concepts))


def mat(network, experiment, iterations=10, precomputed_complexes=None):
    """Matching: Animats are evaluated based on how well they “match” their
    environment. Roughly speaking, this captures the degree to which their
    conceptual structure “resonates” with statistical regularities in the
    world. This quantity is given by

        ϕ * (Σφ'(W) - Σφ'(N)),

    where ϕ is the animat's ϕ-value (averaged over the *unique* states that it
    visits during a game), Σφ'(W) is the sum of φ for each *unique* concept
    that the animat obtains when presented with a stimulus set from the world,
    and Σφ'(N) is the same but for a stimulus set that has been scrambled first
    in space and then in time."""
    # Play the game and a scrambled version of it.
    noise = play_game(experiment, network.tpm, scrambled=True).animat_states
    world = play_game(experiment, network.tpm).animat_states
    # Since the motor states can't influence φ or ϕ, we set them to zero to
    # make uniqifying the states simpler.
    # world[..., ind.motor_indices] = 0
    # noise[..., ind.motor_indices] = 0
    # Get a flat list of all the the states.
    combined = np.concatenate([world, noise])
    combined = combined.reshape(-1, combined.shape[-1])
    # Get unique world and noise states and their counts, up to sensor and
    # hidden states (we care about the sensors since sensor states can
    # influence φ and ϕ as background conditions). The motor states are ignored
    # since now they're all zero.
    all_states = Counter(tuple(state) for state in combined)
    # Get the main complexes for each unique state.
    complexes = precomputed_complexes or {}
    for state in all_states:
        if state not in complexes:
            complexes[state] = pyphi.compute.main_complex(network, state)
    # Existence is the mean of the ϕ values.
    big_phis, counts = zip(*[(complexes[state].phi, count)
                             for state, count in all_states.items()])
    existence = np.average(big_phis, weights=counts)
    # Get the unique concepts in each constellation.
    constellations = {
        state: set(bm.unpartitioned_constellation)
        for state, bm in complexes.items()
    }
    # Preallocate iteration values.
    raw_matching = np.zeros(iterations)
    raw_matching_weighted = np.zeros(iterations)
    raw_matching_average_weighted = np.zeros(iterations)
    shuffled = list(range(len(world)))
    for iteration in range(iterations):
        # Randomly pair trials to form stimulus sets.
        np.random.shuffle(shuffled)
        world_stimuli = [
            np.vstack((world[shuffled[i]], world[shuffled[i + 1]]))
            for i in range(0, len(world), 2)
        ]
        noise_stimuli = [
            np.vstack((noise[shuffled[i]], noise[shuffled[i + 1]]))
            for i in range(0, len(noise), 2)
        ]
        # Get the states in each stimulus set for world and noise.
        world_stimuli = [[tuple(state) for state in stimulus]
                         for stimulus in world_stimuli]
        noise_stimuli = [[tuple(state) for state in stimulus]
                         for stimulus in noise_stimuli]
        # Now we calculate the matching terms for many stimulus sets (each pair
        # of trials) which are later averaged to obtain the matching value for
        # a “typical” stimulus set.
        raw_matching[iteration] = np.mean([
            matching(W, N, constellations)
            for W, N in zip(world_stimuli, noise_stimuli)
        ])
        raw_matching_weighted[iteration] = np.mean([
            matching_weighted(W, N, constellations, complexes)
            for W, N in zip(world_stimuli, noise_stimuli)
        ])
        raw_matching_average_weighted[iteration] = np.mean([
            matching_average_weighted(W, N, constellations, complexes)
            for W, N in zip(world_stimuli, noise_stimuli)
        ])
    return (raw_matching_average_weighted.mean(),
            raw_matching_weighted.mean(),
            existence * raw_matching.mean())
