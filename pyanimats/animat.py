#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# animat.py

"""
Class representing an individual organism in the evolution.

Wraps the C++ Animat extension, providing convenience methods for accessing
animat properties (connectivity, associated PyPhi objects, etc.).
"""

from collections import namedtuple
from copy import deepcopy
from uuid import uuid4

import numpy as np
import pyphi

from . import constants, utils, validate
from .c_animat import pyHMMAgent, pyLinearThresholdAgent
from .experiment import Experiment

Game = namedtuple('Game', ['animat_states', 'world_states', 'animat_positions',
                           'trial_results', 'correct', 'incorrect'])
Mechanism = namedtuple('Mechanism', ['inputs', 'tpm'])


class Animat:

    """
    Represents an animat.

    Args:
        experiment (Experiment): The experiment this animat is a part of.
        genome (Iterable(int)): See attribute.

    Keyword Args:
        parent (Animat): See attribute.
        gen (int): See attribute.

    Attributes:
        genome (Iterable(int)):
            A sequence of integers in the range 0–255 that will determine the
            animat's phenotype.
        parent (Animat):
            The animat's parent. Must be explicitly set upon cloning.
        gen (int):
        edges (list(tuple(int, int))):
            A list of the edges between animat nodes. May contain duplicates.
        cm (np.ndarray):
            The animat's connectivity matrix.
        tpm (np.ndarray):
            The animat's 2-D transition probability matrix.
        network (pyphi.Network):
            The animat as a PyPhi network.
        correct (int):
            The number of trials correctly completed by the animat during a
            single game. Updated every time a game is played; ``False`` if no
            game has been played yet.
        incorrect (int):
            The number of trials incorrectly completed by the animat during a
            single game. Updated every time a game is played; ``False`` if no
            game has been played yet.
    """

    def __init__(self, experiment, genome, gate=constants.HMM_GATE):
        self._experiment = experiment
        if gate == constants.HMM_GATE:
            self._c_animat = pyHMMAgent(genome, experiment.num_sensors,
                                        experiment.num_hidden,
                                        experiment.num_motors,
                                        experiment.deterministic)
        elif gate == constants.LINEAR_THRESHOLD_GATE:
            self._c_animat = pyLinearThresholdAgent(genome,
                                                    experiment.num_sensors,
                                                    experiment.num_hidden,
                                                    experiment.num_motors,
                                                    experiment.deterministic)
        self.parent = None
        self.gen = 0
        self.fitness = 1.0
        self._dirty_fitness = True
        self.raw_fitness = (float('-Inf'),)
        self._correct = False
        self._incorrect = False
        # Get a RNG.
        self.random = constants.DEFAULT_RNG
        # Don't initialize the animat's network attributes until we need to,
        # because it may be expensive.
        self._tpm = False
        self._dirty_tpm = True
        self._cm = False
        self._dirty_cm = True
        self._network = False
        self._dirty_network = True
        # Get a random unique ID.
        self._id = uuid4()

    def __str__(self):
        string = ('Animat(gen={}, genome={}, '
                  'connectivity_matrix=\n{})'.format(
                      self.gen, np.array(self.genome), self.cm))
        return string.replace('\n', '\n' + ' ' * 11)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (self.genome == other.genome and
                self._experiment == other._experiment)

    def __getattr__(self, name):
        """Fall back to experiment attributes."""
        # NOTE: this works as expected because `__getattr__` is only called as
        # a last resort (unlike `__getattribute__`, which should rarely be
        # overriden).
        try:
            return getattr(self._experiment, name)
        except AttributeError:
            raise AttributeError(
                "'Animat' object has no attribute '{}'".format(name))

    def __getstate__(self):
        # Exclude the parent pointer, network attributes, dirty flags, and RNG,
        # from the pickled object.
        state = {k: v for k, v in self.__dict__.items()
                 if k not in ['parent', 'random', '_network', '_dirty_network',
                              '_cm', '_dirty_cm', '_tpm', '_dirty_tpm']}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tpm = False
        self._dirty_tpm = True
        self._cm = False
        self._dirty_cm = True
        self._network = False
        self._dirty_network = True

    def __deepcopy__(self, memo):
        """Make an entirely new instance and copy over some attributes.

        .. note::

            This is not, strictly speaking, a deep copy: the ``parent`` and
            ``random`` attributes are not deeply copied; they're still just
            references.
        """
        copy = Animat(self._experiment, genome=self._c_animat.genome)
        copy.parent = self.parent
        copy.random = self.random
        copy.gen = deepcopy(self.gen)

        # TODO this is a kludge, fix
        copy.fitness = deepcopy(self.fitness)
        copy._dirty_fitness = deepcopy(self._dirty_fitness)
        copy.raw_fitness = deepcopy(self.raw_fitness)
        copy._correct = deepcopy(self._correct)
        copy._incorrect = deepcopy(self._incorrect)
        copy._tpm = deepcopy(self._tpm)
        copy._dirty_tpm = deepcopy(self._dirty_tpm)
        copy._cm = deepcopy(self._cm)
        copy._dirty_cm = deepcopy(self._dirty_cm)
        copy._network = deepcopy(self._network)
        copy._dirty_network = deepcopy(self._dirty_network)
        return copy

    def serializable(self, compact=True, experiment=False):
        """Return a serializable representation of the animat."""
        d = {
            'genome': self.genome,
            'gen': self.gen,
            'correct': self._correct,
            'incorrect': self._incorrect,
            'fitness': utils.rounder(self.fitness),
            'raw_fitness': utils.rounder(self.raw_fitness)
        }
        if not compact:
            d['tpm'] = self.tpm
            d['cm'] = self.cm
            if experiment:
                d['exp'] = self._experiment
        return d

    # TODO compute once
    @property
    def cm(self):
        """The animat's connectivity matrix."""
        if self._dirty_cm:
            cm = np.zeros((self.num_nodes, self.num_nodes), int)
            cm[list(zip(*self.edges))] = 1
            self._cm = cm
            self._dirty_cm = False
        return self._cm

    @property
    def tpm(self):
        """The animats's TPM."""
        if self._dirty_tpm:
            self._tpm = np.array(self._c_animat.tpm).astype(float)
            self._dirty_tpm = False
        return self._tpm

    @property
    def network(self):
        """The PyPhi network representing the animat in the given state."""
        if self._dirty_network:
            self._network = pyphi.Network(self.tpm,
                                          connectivity_matrix=self.cm)
            self._dirty_network = False
        return self._network

    @property
    def correct(self):
        """The number of correct trials in the most recently played game."""
        return self._correct

    @property
    def incorrect(self):
        """The number of incorrect trials in the most recently played game."""
        return self._incorrect

    def lineage(self, step=1):
        """Return the lineage of this animat as a generator."""
        yield self
        ancestor = self.parent
        while ancestor is not None:
            if ancestor.gen % step == 0:
                yield ancestor
            ancestor = ancestor.parent

    def mutate(self):
        """Mutate the animat's genome in-place."""
        self._c_animat.mutate(self.mutation_prob, self.duplication_prob,
                              self.deletion_prob, self.min_genome_length,
                              self.max_genome_length, self.min_dup_del_width,
                              self.max_dup_del_width)
        # Network attributes need updating.
        self._dirty_tpm = True
        self._dirty_cm = True
        self._dirty_network = True

    def play_game(self, scrambled=False):
        """Return the list of state transitions the animat goes through when
        playing the game."""
        game = self._c_animat.play_game(
            self.hit_multipliers, self.block_patterns, self.world_width,
            self.world_height, scramble_world=scrambled)
        game = Game(animat_states=game[0].reshape(self.num_trials,
                                                  self.world_height,
                                                  self.num_nodes),
                    world_states=game[1].reshape(self.num_trials,
                                                 self.world_height),
                    animat_positions=game[2].reshape(self.num_trials,
                                                     self.world_height),
                    trial_results=game[3], correct=game[4], incorrect=game[5])
        assert game.correct + game.incorrect == self.num_trials
        self._correct = game.correct
        self._incorrect = game.incorrect
        return game

    def start_codons(self):
        """Return the locations of start codons in the genome, if any."""
        genome = np.array(self.genome)
        window = utils.rolling_window(genome, len(constants.START_CODON))
        occurrences = np.all((window == constants.START_CODON), axis=1)
        return np.where(occurrences)[0]

    def as_subsystem(self, state=None):
        """Return the PyPhi subsystem consisting of all the animat's nodes."""
        if state is None:
            state = [0] * self.num_nodes
        return pyphi.Subsystem(self.network, state, range(self.num_nodes))

    def brain(self, state=None):
        """Return the PyPhi subsystem consisting of the animat's hidden
        units."""
        if state is None:
            state = [0] * self.num_nodes
        return pyphi.Subsystem(self.network, state, self.hidden_indices)

    def brain_and_sensors(self, state=None):
        """Return the PyPhi subsystem consisting of the animat's hidden
        units and sensors."""
        if state is None:
            state = [0] * self.num_nodes
        return pyphi.Subsystem(
            self.network, state, self.hidden_indices + self.sensor_indices)

    def brain_and_motors(self, state=None):
        """Return the PyPhi subsystem consisting of the animat's hidden
        units and motors."""
        if state is None:
            state = [0] * self.num_nodes
        return pyphi.Subsystem(
            self.network, state, self.hidden_indices + self.motor_indices)

    def mechanism(self, node_index, separate_on_off=False):
        """Return the TPM of a single animat node."""
        node = self.as_subsystem().nodes[node_index]
        tpm = node.tpm[1].squeeze().astype(int)
        states = [pyphi.convert.loli_index2state(i, len(node.inputs))
                  for i in range(tpm.size)]
        logical_function = list(zip(states, tpm.flatten()))
        if separate_on_off:
            # Return the states that lead to OFF separately from those that
            # lead to ON.
            off_mapping = [mapping for mapping in logical_function
                           if not mapping[1]]
            on_mapping = [mapping for mapping in logical_function
                          if mapping[1]]
            logical_function = [off_mapping, on_mapping]
        mechanism = Mechanism(inputs=node.inputs, tpm=logical_function)
        return mechanism


def _c_animat_getter(name):
    """Returns a function that gets ``name`` from the underlying animat."""
    def getter(self):
        return getattr(self._c_animat, name)
    return getter

# A list of animat attributes to expose as read-only properties
_c_animat_properties = ['genome', 'num_sensors', 'num_hidden', 'num_motors',
                        'num_nodes', 'num_states', 'deterministic',
                        'body_length', 'edges']

# Add underlying animat properties to the Animat class
for name in _c_animat_properties:
    setattr(Animat, name, property(_c_animat_getter(name)))


def from_json(dictionary, experiment=None, parent=None):
    """Initialize an animat object from a JSON dictionary.

    Checks that the stored TPM and connectivity matrix match those encoded by
    the stored genome.
    """
    if experiment is None:
        try:
            experiment = Experiment(dictionary['experiment'])
        except KeyError:
            raise ValueError('cannot load animat: no experiment was provided '
                             'and no experiment was found in the JSON data.')
    animat = Animat(experiment, dictionary['genome'])
    animat.parent = parent
    animat.gen = dictionary['gen']
    animat.fitness = dictionary['fitness']
    animat._correct = dictionary['correct']
    animat._incorrect = dictionary['incorrect']
    validate.json_animat(animat, dictionary)
    return animat
