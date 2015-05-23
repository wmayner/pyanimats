#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# analyze.py

import os
import pickle
import json
from glob import glob

from individual import Individual


RESULT_DIR = 'results/current'


def get_correct_counts(output_filename='correct_counts.pkl'):
    correct_counts = []
    for filename in glob('./results/current/**/logbooks.pkl'):
        with open(filename, 'rb') as f:
            print('Processing `{}`'.format(filename))
            logbooks = pickle.load(f)
            correct_counts.append(
                logbooks['correct'][-1]['correct/incorrect'][0])
    with open(output_filename, 'wb') as f:
        pickle.dump(correct_counts, f)
        print('Saved correct counts to `{}`.'.format(output_filename))
    return correct_counts


def make_json_record(output_file):
    SEED = 1
    LINEAGE = 0
    GENERATION = 0

    filename = os.path.join(
        RESULT_DIR, 'seed-{}_params.pkl'.format(SEED))
    with open(filename, 'rb') as f:
        params = pickle.load(f)

    TASKS = [(task[0], int(task[1][::-1], 2)) for task in params['TASKS']]
    hit_multipliers, patterns = zip(*TASKS)

    filename = os.path.join(
        RESULT_DIR, 'seed-{}_lineages.pkl'.format(SEED))
    with open(filename, 'rb') as f:
        d = pickle.load(f)

    params['NUM_NODES'] = 8
    params['NUM_SENSORS'] = 2
    params['WORLD_WIDTH'] = 16
    params['WORLD_HEIGHT'] = 36
    params['NUM_TRIALS'] = 128

    def i2s(i):
        return tuple((i >> n) & 1 for n in range(params['NUM_NODES']))

    ind = Individual(d[LINEAGE][GENERATION].genome)
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
        'generation': GENERATION,
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

    return json_dict
