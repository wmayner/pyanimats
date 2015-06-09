#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# analyze.py

import os
import pickle
import json
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from individual import Individual


CASE_NAME = os.path.join('0.0.3', '3-4-6-5')
RESULT_DIR = 'raw_results'
ANALYSIS_DIR = 'compiled_results'
FILENAMES = {
    'params': 'params.pkl',
    'hof': 'hof.pkl',
    'logbooks': 'logbooks.pkl',
    'lineages': 'lineages.pkl',
    'metadata': 'metadata.pkl',
}


# Utilities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def close():
    """Close a matplotlib figure window."""
    plt.close()


def ensure_exists(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_task_name(tasks):
    return '[' + ',\ '.join(str(task[1].count('1')) for task in tasks) + ']'


def get_desc(params, seed=False, num_seeds=False):
    if not seed and not num_seeds:
        raise Exception('Must provide either a single seed number or the '
                        'number of seeds.')
    return (str(params['NGEN']) + '\ generations,\ ' +
            ('{}\ seeds'.format(num_seeds) if num_seeds
             else 'seed\ {}'.format(seed)) + ',\ task\ ' +
            get_task_name(params['TASKS']) + ',\ population\ size\ '
            + str(params['POPSIZE']))


def get_correct_trials_axis_label(params):
    return ('$\mathrm{Correct\ trials\ (out\ of\ ' + str(params['NUM_TRIALS'])
            + ')}$')


# Result loading
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load(filetype, input_filepath=RESULT_DIR, seed=0):
    result_path = os.path.join(input_filepath, 'seed-{}'.format(seed))
    print('Loading {} from `{}`...'.format(filetype, result_path))
    with open(os.path.join(result_path, FILENAMES[filetype]), 'rb') as f:
        data = pickle.load(f)
    return data


def load_all_seeds(filetype, input_filepath=RESULT_DIR):
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
    input_filepath = os.path.join(RESULT_DIR, case_name)
    output_filepath = os.path.join(
        ensure_exists(os.path.join(ANALYSIS_DIR, case_name)),
        'final-correct-counts.pkl')
    if os.path.exists(output_filepath) and not force:
        print(already_exists_msg(output_filepath))
        with open(output_filepath, 'rb') as f:
            return pickle.load(f)
    correct_counts = []
    for filename, logbooks in load_all_seeds('logbooks',
                                             input_filepath).items():
        correct_counts.append(logbooks['correct'][-1]['correct'])
    params = load('params', input_filepath)
    data = {'correct_counts': correct_counts, 'params': params}
    with open(output_filepath, 'wb') as f:
        pickle.dump(data, f)
    print('Saved final correct counts to `{}`.'.format(output_filepath))
    return data


def plot_final_correct(case_name=CASE_NAME, force=False,
                       bins=np.arange(64, 128, 2), fontsize=20, title=''):
    data = get_final_correct(case_name, force)
    correct_counts, params = data['correct_counts'], data['params']
    fig = plt.figure(figsize=(14, 12))
    plt.hist(correct_counts, bins, normed=True, facecolor='blue', alpha=0.8)
    plt.xlabel(get_correct_trials_axis_label(params), labelpad=20,
               fontsize=fontsize)
    plt.ylabel('$\mathrm{Normalized\ number\ of\ animats}$', labelpad=20,
               fontsize=fontsize)
    plt.title(title + '$\mathrm{Histogram\ of\ animat\ performance:\ '
              + get_desc(params, num_seeds=len(correct_counts))
              + '}$', fontsize=fontsize)
    plt.grid(True)
    fig.show()
    return fig, data


# LOD Evolution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_lods(case_name=CASE_NAME, force=False, gen_interval=500, seed=0,
             all_seeds=False):
    input_filepath = os.path.join(RESULT_DIR, case_name)
    if all_seeds:
        output_filename = 'all-lods'
    else:
        output_filename = 'lods-seed-{}'.format(seed)
    output_filepath = os.path.join(
        ensure_exists(os.path.join(ANALYSIS_DIR, case_name)),
        output_filename + '-every-{}-gen.pkl'.format(gen_interval))
    if os.path.exists(output_filepath) and not force:
        print(already_exists_msg(output_filepath))
        with open(output_filepath, 'rb') as f:
            return pickle.load(f)
    if all_seeds:
        logbooks = [l['correct'] for l in
                    load_all_seeds('logbooks', input_filepath).values()]
    else:
        logbooks = [load('logbooks', input_filepath, seed)['correct']]
    lods = np.array([logbook.select('correct')[::gen_interval]
                     for logbook in logbooks])
    params = load('params', input_filepath)
    data = {'lods': lods, 'params': params}
    with open(output_filepath, 'wb') as f:
        pickle.dump(data, f)
    print('Saved LODs to `{}`.'.format(output_filepath))
    return data


def plot_lod(case_name=CASE_NAME, force=False, gen_interval=500, seed=0,
             all_seeds=False, avg=False, fontsize=20, title=''):
    data = get_lods(case_name, force, gen_interval, seed, all_seeds)
    lods, params = data['lods'], data['params']
    fig = plt.figure(figsize=(14, 12))
    if avg:
        plt.plot(np.arange(lods.shape[1]) * gen_interval, lods.mean(0))
    else:
        for row in lods:
            plt.plot(np.arrange(lods.shape[1]) * gen_interval, row)
    plt.xlabel('$\mathrm{Generation}$', labelpad=20, fontsize=fontsize)
    plt.ylabel(get_correct_trials_axis_label(params), labelpad=20,
               fontsize=fontsize)
    plt.title(title + '$\mathrm{' + ('Average\ a' if avg else 'A') +
              'nimat\ fitness:\ ' + get_desc(params, num_seeds=len(lods))
              + '}$', fontsize=fontsize)
    plt.ylim([60, 130])
    plt.yticks(np.arange(64, 129, 4))
    plt.grid(True)
    fig.show()
    return fig, data


# Metadata
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_avg_elapsed(case_name=CASE_NAME):
    metadata = load_all_seeds('metadata', os.path.join(RESULT_DIR, case_name))
    return np.array([d['elapsed'] for d in metadata.values()]).mean()


# Visual interface
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_game_states(params):
    trials = []
    # Task
    for task in params['TASKS']:
        # Directions (left/right)
        for direction in (-1, 1):
            # Agent starting position
            for agent_pos in range(params['WORLD_WIDTH']):
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
    output_file = os.path.join(ensure_exists(os.path.join(
        ANALYSIS_DIR, case_name, 'seed-{}'.format(seed))), 'game.json')

    params = load('params', input_filepath, seed)

    TASKS = [(task[0], int(task[1][::-1], 2)) for task in params['TASKS']]
    hit_multipliers, patterns = zip(*TASKS)

    lineages = load('lineages', input_filepath, seed)

    def i2s(i):
        return tuple((i >> n) & 1 for n in range(params['NUM_NODES']))

    ind = Individual(lineages[lineage][age].genome)
    transitions = ind.play_game(hit_multipliers, patterns)
    states = [ps[:params['NUM_SENSORS']] + cs[params['NUM_SENSORS']:]
              for ps, cs in zip(map(i2s, transitions[0]),
                                map(i2s, transitions[1]))]

    trial_length = params['WORLD_HEIGHT']

    block_sizes = []
    for pattern in patterns:
        block_sizes += [sum(i2s(pattern))] * int(params['NUM_TRIALS'] /
                                                 len(patterns))

    json_dict = {
        'generation': params['NGEN'] - age,
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
            for i in range(params['NUM_TRIALS'])
        ],
    }
    with open(output_file, 'w') as f:
            json.dump(json_dict, f)
    print('Saved game representation to `{}`.'.format(output_file))

    return json_dict
