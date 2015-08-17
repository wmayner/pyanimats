#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# analyze.py

import os
import pickle
import json
import re
from glob import glob
import numpy as np
import config
import configure
from sklearn.utils.extmath import cartesian
from pyphi.convert import loli_index2state as i2s
from semantic_version import Version

from utils import ensure_exists, unique_rows
from individual import Individual


VERSION = Version('0.0.20')
CASE_NAME = os.path.join(
    str(VERSION),
    'nat',
    '3-4-6-5',
    'sensors-3',
    'jumpstart-0',
    'ngen-60000',
)
SEED = 0
SNAPSHOT = False

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
            _get_task_name(config['TASKS']) + ',\ population\ size\ '
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


CONFIG = load('config', snapshot=SNAPSHOT)


# Correct counts
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_final_correct(case_name=CASE_NAME, force=False):
    input_filepath = os.path.join(RESULT_PATH, case_name)
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

def next_state(ind, state):
    return np.copy(ind.network.tpm[tuple(state)]).astype(int)

def possible_states(num_nodes):
    return cartesian([[0, 1]] * num_nodes)

def sequence_to_state(ind, length=3, sensors=False):
    """Map sequences of sensor stimuli to animat states."""
    if sensors is False:
        sensors = list(range(config.NUM_SENSORS))
    sensor_states = possible_inputs(len(sensors))
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


# Visual interface
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    # Generate the JSON-encodable dictionary.
    json_dict = {
        'config': config,
        'generation': gen,
        'genome': ind.genome,
        'cm': ind.cm.tolist(),
        'correct': ind.correct,
        'incorrect': ind.incorrect,
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
            json.dump(json_dict, f)
    print('Saved game representation to `{}`.'.format(output_file))
    return json_dict


def lineage_to_json():
    pass
