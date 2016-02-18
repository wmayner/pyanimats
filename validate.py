#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# validate.py

import fitness_functions


def _assert_ordering(ordering, name):
    def assertion(dictionary, key, threshold):
        if not ordering(dictionary[key], threshold):
            raise ValueError('invalid experiment: `{}` must be {} '
                             '{}.'.format(key, name, threshold))
    return assertion

_assert_le = _assert_ordering(lambda a, b: a <= b, 'less than or equal to')
_assert_ge = _assert_ordering(lambda a, b: a >= b, 'greater than or equal to')


def experiment_dict(d):
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
