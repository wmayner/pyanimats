#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# parameters.py

import sys
import animat
from pprint import pprint


params = {
    # Simulation parameters
    'NGEN': 60000,
    'POPSIZE': 100,
    'SEED': 0,
    'TASKS': (
        ( 1, '1000000000000000'),
        (-1, '1110000000000000'),
        ( 1, '1000000000000000'),
        (-1, '1110000000000000'),
    ),
    'SCRAMBLE_WORLD': False,
    # Evolution parameters
    'MUTATION_PROB': 0.005,
    'FITNESS_BASE': 1.02,
    'DUPLICATION_PROB': 0.05,
    'DELETION_PROB': 0.02,
    'MAX_GENOME_LENGTH': 10000,
    'MIN_GENOME_LENGTH': 1000,
    'MIN_DUP_DEL_WIDTH': 15,
    'MAX_DUP_DEL_WIDTH': 511,
    'INIT_GENOME': [127] * 5000,
    # Game parameters
    'WORLD_WIDTH': animat.WORLD_WIDTH,
    'WORLD_HEIGHT': animat.WORLD_HEIGHT,
    'NUM_NODES': animat.NUM_NODES,
    'NUM_SENSORS': animat.NUM_SENSORS,
    'NUM_MOTORS': animat.NUM_MOTORS,
    'DETERMINISTIC': animat.DETERMINISTIC,
}
params['NUM_TRIALS'] = len(params['TASKS']) * 2 * params['WORLD_WIDTH']


# Update module namespace with parameters
sys.modules[__name__].__dict__.update(params)


def printable_params():
    d = params.copy()
    # Don't print initial genome
    del d['INIT_GENOME']
    return d


def print_parameters():
    print(''.center(50, '-'))
    print('PARAMETERS:')
    print(''.center(50, '-'))
    pprint(printable_params())
    print(''.center(50, '-'))
