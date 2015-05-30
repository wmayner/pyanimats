#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# analyze.py

import os
import pickle
import json
from glob import glob

from individual import Individual


RESULT_DIR = os.path.join('raw_results', 'test')
ANALYSIS_DIR = 'compiled_results'

PARAM_FILENAME = 'params.pkl'
HOF_FILENAME = 'hof.pkl'
LOGBOOKS_FILENAME = 'logbooks.pkl'
LINEAGES_FILENAME = 'lineages.pkl'
METADATA_FILENAME = 'metadata.pkl'

CORRECT_COUNTS_FILEPATH = os.path.join(ANALYSIS_DIR, 'correct_counts.pkl')


def save_correct_counts(output_filepath=CORRECT_COUNTS_FILEPATH):
    correct_counts = []
    for filename in glob(os.path.join(RESULT_DIR, '**', LOGBOOKS_FILENAME)):
        with open(filename, 'rb') as f:
            print('Processing `{}`'.format(filename))
            logbooks = pickle.load(f)
            correct_counts.append(
                logbooks['correct'][-1]['correct'])
    with open(output_filepath, 'wb') as f:
        pickle.dump(correct_counts, f)
        print('Saved correct counts to `{}`.'.format(output_filepath))
    return correct_counts


def load_correct_counts(input_filepath=CORRECT_COUNTS_FILEPATH):
    with open(input_filepath, 'rb') as f:
        return pickle.load(f)


GAME_JSON_FILEPATH = os.path.join(ANALYSIS_DIR, 'game.json')


def make_json_record(input_filepath=RESULT_DIR, output_file=GAME_JSON_FILEPATH,
                     seed=0, lineage=0, age=0):
    result_path = os.path.join(input_filepath, 'seed-{}'.format(seed))

    with open(os.path.join(result_path, PARAM_FILENAME), 'rb') as f:
        params = pickle.load(f)

    TASKS = [(task[0], int(task[1][::-1], 2)) for task in params['TASKS']]
    hit_multipliers, patterns = zip(*TASKS)

    with open(os.path.join(result_path, LINEAGES_FILENAME), 'rb') as f:
        d = pickle.load(f)

    def i2s(i):
        return tuple((i >> n) & 1 for n in range(params['NUM_NODES']))

    ind = Individual(d[lineage][age].genome)
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
