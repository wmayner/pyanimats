#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# analyze.py

"""Animat analysis functions and data management."""

import os
import pickle
import json
import re
from glob import glob
from collections import Counter
from functools import wraps
import numpy as np
import config
import constants
import configure
import scipy.stats
from sklearn.utils.extmath import cartesian
import pyphi
from pyphi.convert import loli_index2state as i2s
from pyphi.convert import state2loli_index as s2i
from pyphi.jsonify import jsonify
from semantic_version import Version

from utils import ensure_exists, unique_rows
from individual import Individual
import fitness_functions


VERSION = Version('0.0.20')
CASE_NAME = os.path.join(
    str(VERSION),
    'mat-from-scratch',
    '3-4-6-5',
    'sensors-3',
    'jumpstart-0',
    'ngen-60000',
)
SEED = 0

SNAPSHOT = False
SNAPSHOT = -1

RESULT_DIR = 'raw_results'
ANALYSIS_DIR = 'compiled_results'
RESULT_PATH = os.path.join(RESULT_DIR, CASE_NAME)
ANALYSIS_PATH = os.path.join(ANALYSIS_DIR, CASE_NAME)

FILENAMES = {
    'config': 'config.json',
    'hof': 'hof.pkl',
    'logbook': 'logbook.pkl',
    'lineages': 'lineages.pkl',
    'metadata': 'metadata.json',
}
if VERSION < Version('0.0.20'):
    FILENAMES['config'] = 'config.pkl'
    FILENAMES['metadata'] = 'metadata.pkl'


# Utilities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _extract_num(path, prefix):
    return int(re.split('-|/', path.split(prefix)[-1])[0])


def get_seed(path):
    return _extract_num(path, 'seed-')


def get_snapshot(path):
    return _extract_num(path, 'snapshot-')


def get_task_name(tasks):
    return '[' + ',\ '.join(str(task[1].count('1')) for task in tasks) + ']'


def _get_desc(config, seed=False, num_seeds=False):
    if not seed and not num_seeds:
        raise Exception('Must provide either a single seed number or the '
                        'number of seeds.')
    return (str(config['NGEN']) + '\ generations,\ ' +
            ('{}\ seeds'.format(num_seeds) if num_seeds
             else 'seed\ {}'.format(seed)) + ',\ task\ ' +
            get_task_name(config['TASKS']) + ',\ population\ size\ '
            + str(config['POPSIZE']))


def _get_correct_trials_axis_label(config):
    return ('$\mathrm{Correct\ trials\ (out\ of\ ' + str(config['NUM_TRIALS'])
            + ')}$')


# Result loading
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load(filetype, input_filepath=RESULT_PATH, seed=SEED, snapshot=SNAPSHOT,
         verbose=False):
    result_path = os.path.join(input_filepath, 'seed-{}'.format(seed))
    if snapshot:
        result_path = os.path.join(result_path,
                                   'snapshot-*'.format(snapshot))

    filename = FILENAMES[filetype]
    ext = os.path.splitext(filename)[-1]
    path = os.path.join(result_path, filename)
    matches = glob(path)
    if not matches:
        raise Exception("Can't load file, path not found: {}".format(path))

    if snapshot < 0:
        matches = sorted(matches, key=get_snapshot)
        path = matches[snapshot]
    elif snapshot > 0:
        matches = {get_snapshot(path): path for path in matches}
        path = matches[str(snapshot)]
    else:
        path = matches[0]

    if verbose:
        print('Loading {} from `{}`...'.format(filetype, path))

    if ext == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
    elif ext == '.pkl':
        with open(path, 'rb') as f:
            data = pickle.load(f)

    if filetype == 'config':
        configure.from_dict(data)
        if verbose:
            print('Updated PyAnimat configuration with the loaded parameters.')

    return data


def load_all_seeds(filetype, input_filepath=RESULT_PATH, snapshot=SNAPSHOT,
                   verbose=False):
    data = {}
    for path in glob(os.path.join(input_filepath, '*')):
        try:
            data[path] = load(filetype, os.path.dirname(path), get_seed(path),
                              snapshot, verbose)
        except:
            continue
    return data


def already_exists_msg(output_filepath):
    return ('Using existing data file `{}`. Use `force=True` to recompute '
            'from raw data and overwrite.'.format(output_filepath))


# CONFIG = load('config', snapshot=SNAPSHOT)


# Correct counts
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_final_correct(case_name=CASE_NAME, force=False):
    if not case_name:
        input_filepath = RESULT_PATH
    else:
        input_filepath = os.path.join(RESULT_DIR, case_name)
    output_filepath = os.path.join(
        ensure_exists(os.path.join(ANALYSIS_DIR, case_name)),
        'final-correct-counts.pkl')
    if os.path.exists(output_filepath) and not force:
        print(already_exists_msg(output_filepath))
        with open(output_filepath, 'rb') as f:
            return pickle.load(f)
    else:
        print('No already-compiled data found; processing raw data...')
    correct_counts = []
    for filename, logbook in load_all_seeds('logbook', input_filepath).items():
        correct_counts.append(logbook.chapters['correct'][-1]['correct'])
    config = load('config', input_filepath)
    data = {'correct_counts': correct_counts, 'config': config}
    with open(output_filepath, 'wb') as f:
        pickle.dump(data, f)
    print('Saved final correct counts to `{}`.'.format(output_filepath))
    return data


# LOD Evolution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_lods(case_name=CASE_NAME, force=False, gen_interval=500, seed=SEED,
             all_seeds=False, chapter='fitness', stat='max'):
    input_filepath = os.path.join(RESULT_DIR, case_name)
    if all_seeds:
        output_filename = 'all-lods-{}-{}'.format(chapter, stat)
    else:
        output_filename = 'lods-seed-{}'.format(seed)
    output_filepath = os.path.join(
        ensure_exists(os.path.join(ANALYSIS_DIR, case_name)),
        output_filename + '-every-{}-gen.pkl'.format(gen_interval))
    if os.path.exists(output_filepath) and not force:
        print(already_exists_msg(output_filepath))
        with open(output_filepath, 'rb') as f:
            return pickle.load(f)
    else:
        print('Compiled-data file {} does not exist yet; processing raw '
              'data...'.format(output_filename))
    if all_seeds:
        logbooks = [l.chapters[chapter] for l in
                    load_all_seeds('logbook', input_filepath).values()]
    else:
        logbooks = [load('logbook', input_filepath, seed).chapters[chapter]]
    lods = np.array([logbook.select(stat)[::gen_interval]
                     for logbook in logbooks])
    config = load('config', input_filepath)
    data = {'lods': lods, 'config': config}
    with open(output_filepath, 'wb') as f:
        pickle.dump(data, f)
    print('Saved LODs to `{}`.'.format(output_filepath))
    return data


# Metadata
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_avg_elapsed(case_name=CASE_NAME):
    metadata = load_all_seeds('metadata', os.path.join(RESULT_DIR, case_name))
    return np.array([d['elapsed'] for d in metadata.values()]).mean()


# Dynamics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# TODO use ff.avg_over_visited_states instead?
def avg_over_noise_states(n=50):
    """Apply a function of animat states ``n`` times and take the average."""
    def decorator(func):
        @wraps(func)
        def wrapper(ind, **kwargs):
            return np.mean([func(ind.play_game(scrambled=True).animat_states)
                            for _ in range(n)])
        return wrapper
    return decorator


def entropy(ind, node_indices, scrambled=False):
    states = ind.play_game(scrambled=scrambled).animat_states
    sensor_states = states[:, :, node_indices]
    sensor_states = sensor_states.reshape(-1, sensor_states.shape[-1])
    counts = Counter(map(tuple, sensor_states.tolist()))
    p = np.zeros(2**len(node_indices))
    for state in counts.keys():
        p[s2i(state)] = counts[state]
    H_nats = scipy.stats.entropy(p)
    return H_nats * constants.NAT_TO_BIT_CONVERSION_FACTOR


def sensor_entropy(scrambled=False):
    ind = Individual([])
    return entropy(ind, constants.SENSOR_INDICES, scrambled=scrambled)


def hidden_entropy(ind, scrambled=False):
    return entropy(ind, constants.HIDDEN_INDICES, scrambled=scrambled)


def motor_entropy(ind, scrambled=False):
    return entropy(ind, constants.MOTOR_INDICES, scrambled=scrambled)


def next_state(ind, state):
    return np.copy(ind.network.tpm[tuple(state)]).astype(int)


def possible_states(num_nodes):
    return cartesian([[0, 1]] * num_nodes)


def state_mapping(ind, from_indices, to_indices, state=False):
    """Return how the animat maps the state of one subset of nodes to the state
    of another subset (given the current state of the animat, which defaults to
    all off)."""
    if state is False:
        state = [0] * config.NUM_NODES
    idx = list(state)
    for i in from_indices:
        idx[i] = slice(None)
    idx.append(list(to_indices))
    N = len(from_indices)
    subtpm = ind.network.tpm[idx].astype(int)
    return {i2s(k, N): tuple(subtpm[i2s(k, N)]) for k in range(2**N)}


def sequence_to_state(ind, length=3, sensors=False):
    """Map sequences of sensor stimuli to animat states."""
    if sensors is False:
        sensors = list(range(config.NUM_SENSORS))
    sensor_states = possible_states(len(sensors))
    sequences = np.array([
        [sensor_states[i] for i in s]
        for s in cartesian([np.arange(sensor_states.shape[0])] * length)
    ])
    terminal_states = np.zeros([sequences.shape[0], config.NUM_NODES])
    zero_state = np.zeros(config.NUM_NODES)
    for i, sequence in enumerate(sequences):
        state = zero_state
        for sensor_state in sequence:
            state[sensors] = sensor_state
            state = next_state(ind, state)
        terminal_states[i] = state
    return sequences, terminal_states.astype(int)


def state_to_sequences(ind, length=3, sensors=False):
    """Map animat states to input sequences that could have lead to that
    state."""
    sequences, terminal_states = sequence_to_state(ind, length, sensors)
    # For each sequence, get the index of the state it leads to in the array of
    # unique states.
    unique, unq_idx = unique_rows(terminal_states, indices=True)
    # Check that the indices recover the original state array.
    assert np.array_equal(unique[unq_idx], terminal_states)
    # Map the terminal states to the input sequences that could have led to
    # them.
    unique = [tuple(u) for u in unique]
    mapping = {u: [] for u in unique}
    for i, u in enumerate(unq_idx):
        mapping[unique[u]].append(sequences[i].tolist())
    return mapping


def limit_cycle(ind, start=None):
    if start is None:
        start = (0, ) * config.NUM_NODES
    prev = tuple(start)
    cur = tuple(next_state(ind, prev))
    seen = []
    while cur not in seen:
        seen.append(cur)
        prev = cur
        cur = tuple(next_state(ind, prev))
    cycle = seen[seen.index(cur):]
    return cycle


def limit_cycles(ind, states=None):
    if states is None:
        states = possible_states(config.NUM_NODES - config.NUM_MOTORS)
        initial_conditions = [
            tuple(state) + (0, ) * config.NUM_MOTORS
            for state in states
        ]
    else:
        initial_conditions = map(tuple, states)
    return {state: limit_cycle(ind, start=state)
            for state in initial_conditions}


def movements(ind, scrambled=False):
    """Return a boolean array of whether the individual moved."""
    states = ind.play_game(scrambled=scrambled).animat_states
    LEFT = constants.MOTOR_INDICES[0]
    RIGHT = constants.MOTOR_INDICES[1]
    return np.logical_xor(states[:, :, [LEFT]], states[:, :, [RIGHT]])


def num_moves(ind, scrambled=False):
    "Return the number of moves the animat made in the game."
    m = movements(ind, scrambled=scrambled)
    return m.sum()


def percentage_moved(ind, scrambled=False):
    """Return the percentage of timesteps in which the individual moved."""
    m = movements(ind, scrambled=scrambled)
    return m.sum() / m.size


def unq_states(ind, scrambled=False):
    """Return the unique states that the individual visits."""
    states = ind.play_game(scrambled=scrambled).animat_states
    return unique_rows(states)


def get_num_unq_states(states):
    return unique_rows(states).size


get_avg_num_unq_noise_states = avg_over_noise_states()(get_num_unq_states)


# Phi-theoretic
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def num_concepts(ind, state):
    mc = pyphi.compute.main_complex(ind.network, state)
    return len(mc.unpartitioned_constellation)

get_avg_num_concepts_world = \
    fitness_functions.avg_over_visited_states(n=5)(num_concepts)
get_avg_num_concepts_noise = \
    fitness_functions.avg_over_visited_states(
        n=5, scrambled=True)(num_concepts)


# Visual interface
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_phi_data(ind, game, config):
    """Calculate the IIT properties of the given animat for every state.

    The data function must take and individual and a state.
    """
    ff = config['FITNESS_FUNCTION']
    # Get the function that returns the data (before condensing it into a
    # simple fitness value).
    data_func = fitness_functions.metadata[ff]['data_function']
    if data_func is None:
        return None
    # Get the data for every state.
    return {state: data_func(ind, state)
            for state in map(tuple, unique_rows(game.animat_states))}


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


def export_game_to_json(case_name=CASE_NAME, seed=SEED, lineage=0,
                        snapshot=SNAPSHOT, scrambled=False, notes=''):
    input_filepath = os.path.join(RESULT_DIR, case_name)
    output_filepath = os.path.join(
        ANALYSIS_DIR, case_name, 'seed-{}'.format(seed))
    if snapshot:
        output_filepath = os.path.join(output_filepath,
                                       'snapshot-{}'.format(snapshot))
    ensure_exists(output_filepath)
    output_file = os.path.join(
        output_filepath, 'game{}.json'.format('-scrambled' if scrambled
                                              else ''))
    # Load config.
    load('config', input_filepath, seed, snapshot)
    # Load logbook.
    logbook = load('logbook', input_filepath, seed, snapshot)
    gen = logbook[-1]['gen']
    # Load individual.
    lineages = load('lineages', input_filepath, seed, snapshot)
    ind = Individual(lineages[lineage][0].genome)
    # Get the JSON.
    json_dict = game_to_json(ind, gen, scrambled=scrambled)
    # Append notes.
    json_dict['notes'] = notes
    # Record fitness.
    json_dict['fitness'] = float(logbook.chapters['fitness'][-1]['max'])
    with open(output_file, 'w') as f:
        json.dump(jsonify(json_dict), f)
    print('Saved game representation to `{}`.'.format(output_file))
    return json_dict


def export_network_to_json(case_name=CASE_NAME, seed=SEED, lineage=0,
                           snapshot=SNAPSHOT):
    input_filepath = os.path.join(RESULT_DIR, case_name)
    output_filepath = os.path.join(
        ANALYSIS_DIR, case_name, 'seed-{}'.format(seed))
    if snapshot:
        output_filepath = os.path.join(output_filepath,
                                       'snapshot-{}'.format(snapshot))
    ensure_exists(output_filepath)
    output_file = os.path.join(output_filepath, 'network.json')
    # Load config.
    load('config', input_filepath, seed, snapshot)
    # Load individual.
    lineages = load('lineages', input_filepath, seed, snapshot)
    ind = Individual(lineages[lineage][0].genome)
    # Make json dictionary.
    json_network = jsonify(ind.network)
    json_dict = {
        'version': '1.0.2',
        'tpm': json_network['tpm'],
        'cm': json_network['connectivity_matrix'],
        'state': [0] * config.NUM_NODES
    }
    with open(output_file, 'w') as f:
            json.dump(json_dict, f)
    print('Saved network representation to `{}`.'.format(output_file))
    return json_dict


def lineage_to_json():
    pass
