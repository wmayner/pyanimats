#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# analysis.py

import os
import pickle
import json

from individual import Individual


# # Get CPP fitnesses.
# cpp_lods = {}
# for f in glob('../animats/results/current/seed-*/seed-*_LOD.csv'):
#     cpp_lods[f] = np.genfromtxt(f, delimiter=',', dtype=int, skip_header=1)

# cpp_correct = [lod[-1][1] for lod in cpp_lods.values()]
# cpp_fit = np.array([FITNESS_BASE**correct for correct in cpp_correct])

# # Get Python fitnesses.
# py_data = {}
# for f in glob('./results/*final*'):
#     with open(f, 'rb') as dill:
#         py_data[f] = pickle.load(dill)

# py_fit = np.array([ind[1].values[0] for d in py_data.values() for ind in d])

RESULT_DIR = 'results/current'
SEED = 1
LINEAGE = 0
GENERATION = 0
GAME_JSON_OUTPUT = 'game.json'


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
states = [ps[:params['NUM_SENSORS']] + cs[params['NUM_SENSORS']:] for ps, cs in
          zip(map(i2s, transitions[0]), map(i2s, transitions[1]))]

trial_length = params['WORLD_HEIGHT']

block_sizes = []
for pattern in patterns:
    block_sizes += [sum(i2s(pattern))] * int(params['NUM_TRIALS'] / len(patterns))

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

with open(GAME_JSON_OUTPUT, 'w') as f:
    json.dump(json_dict, f)
