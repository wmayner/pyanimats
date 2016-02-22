#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# validate.py

import numpy as np

import fitness_functions


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


def _assert_ordering(ordering, name):
    def assertion(dictionary, key, threshold):
        if not ordering(dictionary[key], threshold):
            raise ValueError('invalid experiment: `{}` must be {} '
                             '{}.'.format(key, name, threshold))
    return assertion

_assert_le = _assert_ordering(lambda a, b: a <= b, 'less than or equal to')
_assert_ge = _assert_ordering(lambda a, b: a >= b, 'greater than or equal to')


def experiment_dict(d):
    if '_derived' in d:
        raise ValueError("the key '_derived' is reserved; please use another "
                         "key.")
    # TODO: check that all necessary params are present
    # Data
    _assert_ge(d, 'log_interval', 1)
    _assert_ge(d, 'num_samples', 1)
    # Evolution
    if d['fitness_function'] not in fitness_functions.metadata.keys():
        raise ValueError(
            'invalid experiment: `fitness_function` must be one of '
            '{}.'.format(list(fitness_functions.metadata.keys())))
    _assert_ge(d, 'ngen', 1)
    _assert_ge(d, 'popsize', 1)
    # TODO validate init_genome_path
    # Animat
    _assert_ge(d, 'num_sensors', 1)
    _assert_ge(d, 'num_hidden', 0)
    _assert_ge(d, 'num_motors', 1)
    _assert_ge(d, 'body_length', 3)
    # Environment
    _assert_ge(d, 'world_width', 1)
    _assert_ge(d, 'world_height', 1)
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
    _assert_ge(d, 'mutation_prob', 0)
    _assert_le(d, 'mutation_prob', 1)
    _assert_ge(d, 'duplication_prob', 0)
    _assert_le(d, 'duplication_prob', 1)
    _assert_ge(d, 'deletion_prob', 0)
    _assert_le(d, 'deletion_prob', 1)
    _assert_ge(d, 'min_genome_length', 1)
    _assert_ge(d, 'min_dup_del_width', 1)
    _assert_le(d, 'max_dup_del_width', d['min_genome_length'])
    # Genetics
    _assert_ge(d, 'default_init_genome_value', 0)
    _assert_le(d, 'default_init_genome_value', 255)
