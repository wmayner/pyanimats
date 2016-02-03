#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# config.py

"""
Configurable parameters.
"""

# TODO don't use this directly, merge with constants during upate?
# Command-line arguments.
ARGUMENTS = None
# Simulation parameters.
NGEN = 10
FITNESS_FUNCTION = 'nat'
SEED = 0
TASKS = [[1, '1000000000000000'],
         [-1, '1110000000000000'],
         [1, '1000000000000000'],
         [-1, '1110000000000000']]
SCRAMBLE_WORLD = False
POPSIZE = 100
# Evolution parameters.
MUTATION_PROB = 0.005
FITNESS_BASE = 1.02
FITNESS_EXPONENT_SCALE = None
FITNESS_EXPONENT_ADD = None
DUPLICATION_PROB = 0.05
DELETION_PROB = 0.02
MAX_GENOME_LENGTH = 10000
MIN_GENOME_LENGTH = 1000
MIN_DUP_DEL_WIDTH = 15
MAX_DUP_DEL_WIDTH = 511
INIT_START_CODONS = 0
