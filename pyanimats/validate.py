#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# validate.py

import numpy as np

from . import fitness_functions
from .constants import MINUTES

GENERIC_MISMATCH_MSG = """
cannot load animat: stored {attr} does not match the {attr} encoded by the
stored genome with the given experiment parameters.
""".strip().replace('\n', ' ')
CM_MISTMATCH_MSG = GENERIC_MISMATCH_MSG.format(attr='connectivity matrix')
TPM_MISTMATCH_MSG = GENERIC_MISMATCH_MSG.format(attr='TPM')
CHECK_VERSION_AND_PARAMS_MSG = """
Check that you are using the same version as when the animat was stored and
that you've provided the same experiment parameters.
""".strip().replace('\n', ' ')

REQUIRED_PARAM_FILE_KEYS = {'simulation', 'experiment'}
REQUIRED_EXPERIMENT_KEYS = {
    'rng_seed', 'fitness_function', 'popsize', 'init_genome_path',
    'init_start_codons', 'fitness_transform', 'num_sensors', 'num_hidden',
    'num_motors', 'body_length', 'world_width', 'world_height', 'task',
    'mutation_prob', 'duplication_prob', 'deletion_prob', 'min_genome_length',
    'max_genome_length', 'min_dup_del_width', 'max_dup_del_width',
    'default_init_genome_value', 'default_init_genome_length', 'deterministic'}
REQUIRED_SIMULATION_KEYS = {
    'ngen', 'checkpoint_interval', 'status_interval', 'logbook_interval',
    'sample_interval', 'all_lineages'}
REQUIRED_FITNESS_TRANSFORM_KEYS = {'base', 'scale', 'add'}


def json_animat(animat, dictionary):
    """Validate an animat loaded from JSON data.

    Ensures that the TPM and connectivity matrix stored with the animat match
    those encoded by the stored genome with the given experiment parameters.
    """
    if 'cm' in dictionary and not np.array_equal(animat.cm,
                                                 np.array(dictionary['cm'])):
        raise ValueError(' '.join([CM_MISTMATCH_MSG,
                                   CHECK_VERSION_AND_PARAMS_MSG]))
    if 'tpm' in dictionary and not np.array_equal(animat.tpm,
                                                  np.array(dictionary['tpm'])):
        raise ValueError(' '.join([TPM_MISTMATCH_MSG,
                                   CHECK_VERSION_AND_PARAMS_MSG]))


def _assert_ordering(ordering, text):
    def assertion(dictionary, name, key, threshold):
        if not ordering(dictionary[key], threshold):
            raise ValueError('invalid {}: `{}` must be {} '
                             '{}.'.format(name, key, text, threshold))
    return assertion

_assert_le = _assert_ordering(lambda a, b: a <= b, 'less than or equal to')
_assert_ge = _assert_ordering(lambda a, b: a >= b, 'greater than or equal to')
_assert_lt = _assert_ordering(lambda a, b: a < b, 'less than')
_assert_gt = _assert_ordering(lambda a, b: a > b, 'greater than')


def _assert_nonempty_dict(d, name):
    if not isinstance(d, dict):
        raise ValueError('{} must be a dictionary.'.format(name))
    if not d:
        raise ValueError('empty {}'.format(name))


def _assert_has_keys(d, required, name):
    missing = required - set(d.keys())
    if missing:
        raise ValueError(
            'invalid {}: missing `{}`.'.format(name, '`, `'.join(missing)))


def param_file(d):
    name = 'parameter_file'
    _assert_nonempty_dict(d, name)
    _assert_has_keys(d, REQUIRED_PARAM_FILE_KEYS, name)
    for k in REQUIRED_PARAM_FILE_KEYS:
        _assert_nonempty_dict(d[k], k)


def simulation(d):
    name = 'simulation parameters'
    _assert_nonempty_dict(d, name)
    _assert_has_keys(d, REQUIRED_SIMULATION_KEYS, name)
    _assert_ge(d, name, 'logbook_interval', 1)
    # Get the generational interval at which to print the evolution status.
    if d['sample_interval'] <= 0:
        d['sample_interval'] = float('inf')
    # Get the generational interval at which to print the evolution status.
    if d['status_interval'] <= 0:
        d['status_interval'] = float('inf')
    # Get the time interval at which to save checkpoints.
    d['checkpoint_interval'] = (d['checkpoint_interval'] * MINUTES)
    if d['checkpoint_interval'] <= 0:
        d['checkpoint_interval'] = float('inf')
    return d


def experiment(d):
    name = 'experiment parameters'
    _assert_nonempty_dict(d, name)
    if '_derived' in d:
        raise ValueError("the key '_derived' is reserved; please use another "
                         "key.")
    # Check that all necessary params are present.
    _assert_has_keys(d, REQUIRED_EXPERIMENT_KEYS, 'experiment parameters')
    # Evolution
    if d['fitness_function'] not in fitness_functions.metadata.keys():
        raise ValueError(
            'invalid experiment: `fitness_function` must be one of '
            '{}.'.format(list(fitness_functions.metadata.keys())))
    _assert_ge(d, name, 'popsize', 1)
    if d['fitness_transform'] is not None:
        _assert_has_keys(d['fitness_transform'],
                         REQUIRED_FITNESS_TRANSFORM_KEYS, 'fitness transform')
        _assert_gt(d['fitness_transform'], 'fitness transform', 'base', 0)
        _assert_gt(d['fitness_transform'], 'fitness transform', 'scale', 0)
        _assert_ge(d['fitness_transform'], 'fitness transform', 'add', 0)
    # TODO validate init_genome_path
    # Animat
    _assert_ge(d, name, 'num_sensors', 1)
    _assert_ge(d, name, 'num_hidden', 0)
    if d['num_motors'] not in [0, 2]:
        raise ValueError(
            'invalid experiment: must have either 0 or 2 motor units.')
    _assert_ge(d, name, 'body_length', 3)
    # Environment
    _assert_ge(d, name, 'world_width', 1)
    _assert_ge(d, name, 'world_height', 1)
    if not all(len(pattern[1]) == d['world_width'] for pattern in d['task']):
        raise ValueError(
            'invalid experiment: malformed task: each block pattern in the '
            'task must be the same length as `world_width`.')
    try:
        # Try to cast the block patterns to integers.
        [int(pattern[1], 2) for pattern in d['task']]
    except ValueError:
        raise ValueError('invalid experiment: malformed task: block patterns '
                         'must be strings consisting only of 0s and 1s.')
    # Mutation
    _assert_ge(d, name, 'mutation_prob', 0)
    _assert_le(d, name, 'mutation_prob', 1)
    _assert_ge(d, name, 'duplication_prob', 0)
    _assert_le(d, name, 'duplication_prob', 1)
    _assert_ge(d, name, 'deletion_prob', 0)
    _assert_le(d, name, 'deletion_prob', 1)
    _assert_ge(d, name, 'min_genome_length', 1)
    _assert_ge(d, name, 'min_dup_del_width', 1)
    _assert_le(d, name, 'max_dup_del_width', d['min_genome_length'])
    # Genetics
    _assert_ge(d, name, 'default_init_genome_value', 0)
    _assert_le(d, name, 'default_init_genome_value', 255)
