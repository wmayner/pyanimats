#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# configure.py

"""
Handles the configuration of parameters from files or command-line arguments.
"""

import os
import pickle
from copy import copy
import yaml
import pyphi

import config
import constants as _


def from_args(args):
    """Load configuration values from command-line arguments."""
    c = {}
    # Prune optional args that weren't used.
    args = {key: value for key, value in args.items()
            if not (value is False or value is None)}
    # Load tasks from file if a filename was given.
    if '<tasks.yml>' in args:
        with open(args['<tasks.yml>'], 'r') as f:
            c.update({'TASKS': yaml.load(f)})
        del args['<tasks.yml>']
    # Parse the rest of the parameters.
    for key, value in args.items():
        name, cast = arg_name_and_type[key]
        c[name] = cast(value)
    # Load configuration from file if a filename was given.
    if '--config' in args:
        filename = args['--config']
        print('Loading configuration from `{}`.\n'.format(filename))
        with open(filename, 'r') as f:
            c.update(yaml.load(f))
        del args['--config']
    # Update the configuration.
    from_dict(c)
    return args


# Map command-line argument names to parameter names and types.
arg_name_and_type = {
    '--fitness': ('FITNESS_FUNCTION', str),
    '--seed': ('SEED', int),
    '--num-gen': ('NGEN', int),
    '--pop-size': ('POPSIZE', int),
    '--mut-prob': ('MUTATION_PROB', float),
    '--jumpstart': ('INIT_START_CODONS', int),
    '--scramble': ('SCRAMBLE_WORLD', bool),
    '--init-genome': ('INIT_GENOME', str),
    '--config': ('CONFIGURATION_FILE', str),
    '--dup-prob': ('DUPLICATION_PROB', float),
    '--del-prob': ('DELETION_PROB', float),
    '--max-length': ('MAX_GENOME_LENGTH', int),
    '--min-length': ('MIN_GENOME_LENGTH', int),
    '--min-dup-del': ('MIN_DUP_DEL_WIDTH', int),
    '--fit-base': ('FITNESS_BASE', float),
    '--fit-exp-add': ('FITNESS_EXPONENT_ADD', float),
    '--fit-exp-scale': ('FITNESS_EXPONENT_SCALE', float),
}


def from_dict(c):
    """Update the configuration from the given dictionary."""
    for key, value in c.items():
        setattr(config, key, value)
    _update_constants()
    return c


def _update_constants():
    """Update constants that are derived from configurable parameters."""
    # Number of trials is given by
    #   (number of tasks * two directions *
    #    number of initial positions for the animat)
    _.NUM_TRIALS = len(config.TASKS) * 2 * config.WORLD_WIDTH

    # Convert task-strings into integers. Note that in the C++ implementation,
    # the world is mirrored; hence the reversal of the string.
    int_tasks = [(task[0], int(task[1][::-1], 2)) for task in config.TASKS]
    _.HIT_MULTIPLIERS, _.BLOCK_PATTERNS = zip(*int_tasks)

    if hasattr(config, 'INIT_GENOME'):
        path = os.path.join(config.INIT_GENOME, 'lineages.pkl')
        with open(path, 'rb') as f:
            lineages = pickle.load(f)
            # Use the genome of the best individual of the most recent
            # generation.
            _.INIT_GENOME = lineages[0][0].genome
    # Insert start codons into the initial genome.
    elif config.INIT_START_CODONS > 0:
        _.INIT_GENOME = copy(_.DEFAULT_INIT_GENOME)
        gap = len(_.INIT_GENOME) // config.INIT_START_CODONS
        for i in range(config.INIT_START_CODONS):
            _.INIT_GENOME[(i * gap):(i * gap + 1)] = _.START_CODON

    # Scale raw fitness values so they're in the range 64–128
    # before using them as an exponent (this depends on which fitness function
    # is used).
    TRANSFORMS = {
        'nat': {'scale': 1, 'add': 0},
        'mi': {'scale': 64 / min(config.NUM_SENSORS, config.NUM_MOTORS),
               'add': 64},
        'mi_wvn': {'scale': 64 / min(config.NUM_SENSORS, config.NUM_MOTORS),
                   'add': 64},
        'ex': {'scale': 64 / 4, 'add': 64},
        'ex_wvn': {'scale': 64 / 1, 'add': 64},
        'sp': {'scale': 64 / 4, 'add': 64},
        'sp_wvn': {'scale': 64 / 4, 'add': 64},
        'bp': {'scale': 64 / 4, 'add': 64},
        'bp_wvn': {'scale': 63 / 4, 'add': 64},
        'state_wvn': {'scale': 64 / 8, 'add': 64},
        'mat': {'scale': 64 / 1, 'add': 64},
    }
    ff = config.FITNESS_FUNCTION
    if config.FITNESS_EXPONENT_SCALE is None:
        config.FITNESS_EXPONENT_SCALE = TRANSFORMS[ff]['scale']
    if config.FITNESS_EXPONENT_ADD is None:
        config.FITNESS_EXPONENT_ADD = TRANSFORMS[ff]['add']

    # Get sensor, hidden unit, and motor indices.
    _.SENSOR_INDICES = tuple(range(config.NUM_SENSORS))
    _.HIDDEN_INDICES = tuple(range(config.NUM_SENSORS, (config.NUM_NODES -
                                                        config.NUM_MOTORS)))
    _.MOTOR_INDICES = tuple(range(config.NUM_NODES - config.NUM_MOTORS,
                                  config.NUM_NODES))

    # Get combinations thereof.
    _.SENSOR_HIDDEN_INDICES = _.SENSOR_INDICES + _.HIDDEN_INDICES
    _.HIDDEN_MOTOR_INDICES = _.HIDDEN_INDICES + _.MOTOR_INDICES
    _.SENSOR_MOTOR_INDICES = _.SENSOR_INDICES + _.MOTOR_INDICES

    # Get their power sets.
    _.HIDDEN_POWERSET = tuple(pyphi.utils.powerset(_.HIDDEN_INDICES))
    _.SENSORS_AND_HIDDEN_POWERSET = tuple(
        pyphi.utils.powerset(_.SENSOR_INDICES + _.HIDDEN_INDICES))
    _.HIDDEN_AND_MOTOR_POWERSET = tuple(
        pyphi.utils.powerset(_.HIDDEN_INDICES + _.MOTOR_INDICES))

    # Get information about possible animat states.
    _.NUM_SENSOR_STATES = 2**config.NUM_SENSORS
    _.NUM_MOTOR_STATES = 2**config.NUM_MOTORS
    _.POSSIBLE_STATES = [pyphi.convert.loli_index2state(i, config.NUM_NODES)
                         for i in range(2**config.NUM_NODES)]

    # Get sensor locations (mapping them to the sensor index).
    if config.NUM_SENSORS == 2:
        _.SENSOR_LOCATIONS = [0, 2]
    else:
        _.SENSOR_LOCATIONS = list(range(config.NUM_SENSORS))

    def _bitlist(i, padlength):
        """Return a list of the bits of an integer, padded up to
        ``padlength``."""
        return list(map(int, bin(i)[2:].zfill(padlength)))

    _.SENSOR_MOTOR_STATES = [
        ((i, j), (_bitlist(i, config.NUM_SENSORS) +
                  _bitlist(j, config.NUM_MOTORS)))
        for i in range(_.NUM_SENSOR_STATES) for j in range(_.NUM_MOTOR_STATES)
    ]


def get_dict(full=False):
    """Return the current configuration as a dictionary. Optionally, include
    constants as well."""
    c = {}
    ignore = ['animat', 'ARGUMENTS', 'POSSIBLE_STATES',
              'HIDDEN_POWERSET', 'SENSORS_AND_HIDDEN_POWERSET',
              'HIDDEN_AND_MOTOR_POWERSET', 'SENSOR_MOTOR_STATES',
              'DEFAULT_INIT_GENOME']
    ignore_full = ['INIT_GENOME', 'math']
    for key in dir(config):
        if not key.startswith('__') and key not in ignore:
            c[key] = getattr(config, key)
    if full:
        for key in dir(_):
            if not key.startswith('__') and key not in ignore + ignore_full:
                c[key] = getattr(_, key)
    return c
