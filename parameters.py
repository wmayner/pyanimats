#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# parameters.py

"""
Parameters for controlling animat evolution.
"""

from pprint import pformat
import numpy as np
import yaml
import animat


DEFAULTS = {
    # Simulation parameters.
    'NGEN': 10,
    'FITNESS_FUNCTION': 'nat',
    'SEED': 0,
    'TASKS': (
        ( 1, '1000000000000000'),
        (-1, '1110000000000000'),
        ( 1, '1000000000000000'),
        (-1, '1110000000000000'),
    ),
    'SCRAMBLE_WORLD': False,
    'POPSIZE': 100,
    # Evolution parameters.
    'MUTATION_PROB': 0.005,
    'FITNESS_BASE': 1.02,
    'FITNESS_EXPONENT_SCALE': 1,
    'FITNESS_EXPONENT_ADD': 0,
    'DUPLICATION_PROB': 0.05,
    'DELETION_PROB': 0.02,
    'MAX_GENOME_LENGTH': 10000,
    'MIN_GENOME_LENGTH': 1000,
    'MIN_DUP_DEL_WIDTH': 15,
    'MAX_DUP_DEL_WIDTH': 511,
    'INIT_GENOME': [127] * 5000,
    # Game parameters.
    # TODO make these configurable
    'WORLD_WIDTH': animat.WORLD_WIDTH,
    'WORLD_HEIGHT': animat.WORLD_HEIGHT,
    'NUM_NODES': animat.NUM_NODES,
    'NUM_SENSORS': animat.NUM_SENSORS,
    'NUM_MOTORS': animat.NUM_MOTORS,
    'DETERMINISTIC': animat.DETERMINISTIC,
}


# Map command-line argument names to parameter names and types.
param_name_and_types = {
    '--fitness': ('FITNESS_FUNCTION', str),
    '--seed': ('SEED', int),
    '--num-gen': ('NGEN', int),
    '--pop-size': ('POPSIZE', int),
    '--mut-prob': ('MUTATION_PROB', float),
    '--scramble': ('SCRAMBLE_WORLD', bool),
    '--dup-prob': ('DUPLICATION_PROB', float),
    '--del-prob': ('DELETION_PROB', float),
    '--max-length': ('MAX_GENOME_LENGTH', int),
    '--min-length': ('MIN_GENOME_LENGTH', int),
    '--min-dup-del': ('MIN_DUP_DEL_WIDTH', int),
    '--fit-base': ('FITNESS_BASE', float),
    '--fit-exp-add': ('FITNESS_EXPONENT_ADD', float),
    '--fit-exp-scale': ('FITNESS_EXPONENT_SCALE', float),
}


class Parameters(dict):

    """
    Holds evolution parameter values.

    Do not set any parameters directly with this object; use command-line
    arguments or a configuration file instead.
    """

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self._refresh()

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self._refresh()

    def __repr__(self):
        printable = self.copy()
        # Cast initial genome to a NumPy array for compact printing.
        printable['INIT_GENOME'] = np.array(printable['INIT_GENOME'])
        return pformat(printable, indent=1)

    def __str__(self):
        return repr(self)

    def load_from_args(self, arguments):
        """Set parameters from command-line arguments."""
        # Prune optional arguments that weren't used.
        arguments = {key: value for key, value in arguments.items()
                     if not (value is False or value is None)}
        # Load tasks from file if a filename was given.
        if '<tasks.yml>' in arguments:
            with open(arguments['<tasks.yml>'], 'r') as f:
                self['TASKS'] = yaml.load(f)
            del arguments['<tasks.yml>']
        # Set the rest of the parameters.
        for key, value in arguments.items():
            name, cast = param_name_and_types[key]
            self[name] = cast(value)
        self._refresh()

    def load_from_file(self, param_file):
        """Set parameters from a YAML file."""
        print('Using parameters from `{}`.\n'.format(param_file))
        with open(param_file, 'r') as f:
            file_params = yaml.load(f)
        self.update(file_params)
        self._refresh()

    def _refresh(self):
        """Call this after anything changes."""
        # (number of tasks * two directions *
        #  number of initial positions for the animat)
        self['NUM_TRIALS'] = len(self['TASKS']) * 2 * self['WORLD_WIDTH']
        # Convert world-strings into integers. Note that in the C++
        # implementation, the world is mirrored; hence the reversal of the
        # string.
        int_tasks = [(task[0], int(task[1][::-1], 2))
                     for task in self['TASKS']]
        self['HIT_MULTIPLIERS'], self['BLOCK_PATTERNS'] = zip(*int_tasks)
        # Scale raw mutual information values so they're in the range 64–128
        # before using them as an exponent (the max is either the number of
        # sensors or of motors, whichever is smaller).
        if self['FITNESS_FUNCTION'] == 'mi':
            self['FITNESS_EXPONENT_SCALE'] = 64 / min(self['NUM_SENSORS'],
                                                      self['NUM_MOTORS'])
            self['FITNESS_EXPONENT_ADD'] = 64
        # Scale raw extrinsic cause information values so they're in the range
        # 64–128 (the highest observed so far is around 14 or so, according to
        # Jaime—this assumes a max of 16).
        if self['FITNESS_FUNCTION'] == 'ex':
            self['FITNESS_EXPONENT_SCALE'] = 64 / 16
            self['FITNESS_EXPONENT_ADD'] = 64
        # Get sensor, hidden unit, and motor indices.
        self['SENSOR_INDICES'] = tuple(range(self['NUM_SENSORS']))
        self['HIDDEN_INDICES'] = tuple(
            range(self['NUM_SENSORS'], (self['NUM_NODES'] -
                                        self['NUM_MOTORS'])))
        self['MOTOR_INDICES'] = tuple(
            range(self['NUM_NODES'] - self['NUM_MOTORS'], self['NUM_NODES']))
        # Make entries accessible via dot-notation.
        self.__dict__ = self


params = Parameters(**DEFAULTS)
