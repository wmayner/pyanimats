#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# parameters.py

import sys


param_dict = {
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
    'INIT_GENOME': [127] * 5000,
    'MUTATION_PROB': 0.002,
    'DUPLICATION_PROB': 0.05,
    'DELETION_PROB': 0.02,
    'MAX_GENOME_LENGTH': 10000,
    'MIN_GENOME_LENGTH': 1000,
    'MIN_DUP_DEL_WIDTH': 15,
    'MAX_DUP_DEL_WIDTH': 511,
    'FITNESS_BASE': 1.02,
}

sys.modules[__name__].__dict__.update(param_dict)
