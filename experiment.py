#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# experiment.py

import os
import pickle
import pprint

import pyphi
import yaml
from munch import Munch

import constants
import validate


class Experiment(Munch):

    """Parameters specifying an evolutionary simulation.

    These can be accessed as attributes, i.e. with dot notation.

    **Note on magic:** there are several parameters that are derived from those
    provided upon initialization; these can also be accessed directly as
    attributes on this object, though they're stored under the ``_derived`` key
    and are not printed. See ``experiment._derived.keys()`` for a list of
    these.

    Keyword Args:
        filepath (string): A file path pointing to a YAML file containing
            experiment parameters.
        override (string): A dictionary of experiment parameters. Values in
            this dictionary overwrite values from the file.

    Example:
        >>> e = Experiment('experiments/example.yml')
        >>> e.num_sensors
        3
        >>> # This is a derived parameter that was not present in the file:
        >>> e.sensor_indices
        (0, 1, 2)
    """

    def __init__(self, override=None, filepath=None):
        dictionary = dict()
        # Store the filepath in case the user needs to remember it later.
        self.filepath = filepath
        # Load the given YAML file if provided.
        if filepath is not None:
            with open(filepath) as f:
                dictionary.update(yaml.load(f))
        # Update from the dictionary if provided.
        if override is not None:
            dictionary.update(override)
        # Validate.
        validate.experiment_dict(dictionary)
        # Derive parameters from the user-set ones.
        dictionary['_derived'] = _derive_params(dictionary)
        # Put everything in the Munch.
        self.update(dictionary)

    def __getstate__(self):
        return self.serializable()

    def __setstate__(self, state):
        self.__init__(override=state)

    def __getattr__(self, k):
        """Fall back on derived parameters if ``k`` is not an attribute."""
        try:
            return object.__getattribute__(self, k)
        except AttributeError:
            try:
                return self[k]
            except KeyError:
                try:
                    return self._derived[k]
                except KeyError:
                    raise AttributeError(k)

    def __repr__(self):
        """Return a readable representation of the experiment.

        This does not include derived parameters.
        """
        # Format with pretty print, and indent.
        return ('Experiment({\n ' +
                pprint.pformat(self.serializable(), indent=1)[1:-1] +
                '\n})')

    def serializable(self):
        """Return a serializable representation of the experiment."""
        # Exclude `_derived` parameters when serializing to JSON.
        return {k: v for k, v in self.items() if k != '_derived'}


def _derive_params(d):
    """Derive various secondary parameters from the given dictionary."""
    num_nodes = d['num_sensors'] + d['num_hidden'] + d['num_motors']
    # Load initial genome if provided.
    if d['init_genome_path']:
        path = os.path.join(d['init_genome_path'], 'lineages.pkl')
        with open(path, 'rb') as f:
            lineages = pickle.load(f)
            # Use the genome of the best individual of the most recent
            # generation.
            init_genome = lineages[0][0].genome
    else:
        # Use the default genome and inject start codons.
        init_genome = ([d['default_init_genome_value']] *
                       d['default_init_genome_length'])
        if d['init_start_codons']:
            gap = len(init_genome) // d['init_start_codons']
            for i in range(d['init_start_codons']):
                init_genome[(i * gap):(i * gap + 1)] = constants.START_CODON
    # If no fitness transform was given, use the defaults.
    if 'fitness_transform' not in d:
        # The fitness scale of mutual information depends on the number of
        # sensors/motors.
        fitness_transform = constants.FITNESS_TRANSFORMS[d['fitness_function']]
        if d['fitness_function'] in ['mi', 'mi_wvn']:
            fitness_transform = {'scale': 64 / min(d['num_sensors'],
                                                   d['num_motors']),
                                 'add': 64}
    else:
        fitness_transform = False
    sensor_indices = tuple(range(d['num_sensors']))
    hidden_indices = tuple(range(
        d['num_sensors'], d['num_sensors'] + d['num_hidden']))
    motor_indices = tuple(range(
        d['num_sensors'] + d['num_hidden'],
        d['num_sensors'] + d['num_hidden'] + d['num_motors']))
    num_sensor_states = 2**d['num_sensors']
    num_hidden_states = 2**d['num_hidden']
    num_motor_states = 2**d['num_motors']
    sensor_motor_states = [
        ((i, j), (_bitlist(i, d['num_sensors']) +
                  _bitlist(j, d['num_motors'])))
        for i in range(num_sensor_states) for j in range(num_motor_states)
    ]
    # Get sensor locations (mapping them to the sensor index).
    if d['num_sensors'] < constants.MIN_BODY_LENGTH:
        gap = constants.MIN_BODY_LENGTH - d['num_sensors']
        gap_start = constants.MIN_BODY_LENGTH // 2
        sensor_locations = (tuple(range(gap_start)) +
                            tuple(range(gap_start + gap,
                                        constants.MIN_BODY_LENGTH)))
    else:
        sensor_locations = list(range(d['num_sensors']))
    # Fill and return the dictionary.
    return {
        'num_nodes': num_nodes,
        'init_genome': init_genome,
        'fitness_transform': fitness_transform,
        # Number of trials is given by
        #   (number of tasks * two directions *
        #    number of initial positions for the animat)
        'num_trials': len(d['task']) * 2 * d['world_width'],
        'hit_multipliers': [condition[0] for condition in d['task']],
        # Convert task-strings into integers. Note that in the C++
        # implementation, the world is mirrored; hence the reversal of the
        # string.
        'block_patterns': [int(condition[1][::-1], 2)
                           for condition in d['task']],
        'sensor_indices': sensor_indices,
        'hidden_indices': hidden_indices,
        'motor_indices': motor_indices,
        'sensor_hidden_indices': sensor_indices + hidden_indices,
        'hidden_motor_indices': hidden_indices + motor_indices,
        'sensor_motor_indices': sensor_indices + motor_indices,
        # Get their power sets.
        'hidden_powerset': tuple(pyphi.utils.powerset(hidden_indices)),
        'sensors_and_hidden_powerset': tuple(
            pyphi.utils.powerset(sensor_indices + hidden_indices)),
        'hidden_and_motor_powerset': tuple(
            pyphi.utils.powerset(hidden_indices + motor_indices)),
        # Get information about possible animat states.
        'num_sensor_states': num_sensor_states,
        'num_hidden_states': num_hidden_states,
        'num_motor_states': num_motor_states,
        'num_possible_states': 2**num_nodes,
        'possible_states': [pyphi.convert.loli_index2state(i, num_nodes)
                            for i in range(2**num_nodes)],
        'sensor_motor_states': sensor_motor_states,
        'sensor_locations': sensor_locations,
    }


def _bitlist(i, padlength):
    """Return a list of the bits of an integer, padded up to ``padlength``.

    Args:
        i (int): A decimal number.
        padlength (int): The desired length of the binary representation,
            padded with leading zeros.

    Returns:
        bitlist (list(int)): The list of bits.
    """
    return list(map(int, bin(i)[2:].zfill(padlength)))
