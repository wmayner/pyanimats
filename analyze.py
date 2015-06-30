#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# analyze.py

import os
import pickle
import json
from glob import glob
import numpy as np
import configure

from individual import Individual


CASE_NAME = '0.0.10/sp/3-4-6-5/sensors-3/jumpstart-0/gen-4000'
RESULT_DIR = 'raw_results'
ANALYSIS_DIR = 'compiled_results'
RESULT_PATH = os.path.join(RESULT_DIR, CASE_NAME)
ANALYSIS_PATH = os.path.join(ANALYSIS_DIR, CASE_NAME)
FILENAMES = {
    'config': 'config.pkl',
    'hof': 'hof.pkl',
    'logbook': 'logbook.pkl',
    'lineages': 'lineages.pkl',
    'metadata': 'metadata.pkl',
}


# Utilities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _ensure_exists(path):
    os.makedirs(path, exist_ok=True)
    return path


def _get_task_name(tasks):
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

def load(filetype, input_filepath=RESULT_PATH, seed=0):
    result_path = os.path.join(input_filepath, 'seed-{}'.format(seed))
    print('Loading {} from `{}`...'.format(filetype, result_path))
    with open(os.path.join(result_path, FILENAMES[filetype]), 'rb') as f:
        data = pickle.load(f)
    if filetype == 'config':
        configure.from_dict(data)
        print('Updated PyAnimat configuration with the loaded parameters.')
    return data


def load_all_seeds(filetype, input_filepath=RESULT_PATH):
    data = {}
    for filename in glob(os.path.join(input_filepath, '**',
                                      FILENAMES[filetype])):
        print('Loading {} from `{}`...'.format(filetype, filename))
        with open(filename, 'rb') as f:
            data[filename] = pickle.load(f)
    return data


def already_exists_msg(output_filepath):
    return ('Using existing data file `{}`. Use `force=True` to recompute '
            'from raw data and overwrite.'.format(output_filepath))


# Correct counts
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_final_correct(case_name=CASE_NAME, force=False):
    input_filepath = os.path.join(RESULT_PATH, case_name)
    output_filepath = os.path.join(
        _ensure_exists(os.path.join(ANALYSIS_DIR, case_name)),
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

def get_lods(case_name=CASE_NAME, force=False, gen_interval=500, seed=0,
             all_seeds=False, chapter='fitness', stat='max'):
    input_filepath = os.path.join(RESULT_DIR, case_name)
    if all_seeds:
        output_filename = 'all-lods-{}-{}'.format(chapter, stat)
    else:
        output_filename = 'lods-seed-{}'.format(seed)
    output_filepath = os.path.join(
        _ensure_exists(os.path.join(ANALYSIS_DIR, case_name)),
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


# Visual interface
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_game_states(config):
    trials = []
    # Task
    for task in config['TASKS']:
        # Directions (left/right)
        for direction in (-1, 1):
            # Agent starting position
            for agent_pos in range(config['WORLD_WIDTH']):
                trials.append({
                    'task': {
                        'goalIsCatch': task[0],
                        'block': task[1],
                    },
                    'direction': direction,
                    'initAgentPos': agent_pos,
                })
    # TODO finish


def make_json_record(case_name=CASE_NAME, seed=0, lineage=0, age=0):
    input_filepath = os.path.join(RESULT_DIR, case_name)
    output_file = os.path.join(_ensure_exists(os.path.join(
        ANALYSIS_DIR, case_name, 'seed-{}'.format(seed))), 'game.json')

    config = load('config', input_filepath, seed)

    TASKS = [(task[0], int(task[1][::-1], 2)) for task in config['TASKS']]
    hit_multipliers, patterns = zip(*TASKS)

    lineages = load('lineages', input_filepath, seed)

    def i2s(i):
        return tuple((i >> n) & 1 for n in range(config['NUM_NODES']))

    ind = Individual(lineages[lineage][age].genome)
    transitions = ind.play_game(hit_multipliers, patterns)
    states = [ps[:config['NUM_SENSORS']] + cs[config['NUM_SENSORS']:]
              for ps, cs in zip(map(i2s, transitions[0]),
                                map(i2s, transitions[1]))]

    trial_length = config['WORLD_HEIGHT']

    block_sizes = []
    for pattern in patterns:
        block_sizes += [sum(i2s(pattern))] * int(config['NUM_TRIALS'] /
                                                 len(patterns))

    json_dict = {
        'generation': config['NGEN'] - age,
        'connectivityMatrix': ind.cm.T.tolist(),
        'nodeTypes': {
            'sensors': [0, 1],
            'hidden': [2, 3, 4, 5],
            'motors': [6, 7],
        },
        'blockSize': block_sizes,
        'Trial': [
            {'trialNum': i,
             'lifeTable': states[(i * trial_length):((i + 1) * trial_length)]}
            for i in range(config['NUM_TRIALS'])
        ],
    }
    with open(output_file, 'w') as f:
            json.dump(json_dict, f)
    print('Saved game representation to `{}`.'.format(output_file))

    return json_dict
