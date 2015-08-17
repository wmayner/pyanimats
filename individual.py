#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# individual.py

"""
Class representing an individual organism in the evolution.

Wraps the C++ Animat extension, providing convenience methods for accessing
animat properties (connectivity, associated PyPhi objects, etc.).
"""

import numpy as np
from copy import deepcopy
from collections import namedtuple
import functools
import pyphi

import config
import constants as _
import utils
from animat import Animat


class ExponentialFitness:

    """
    Represents the two notions of fitness: the value that is used in
    selection (the ``exponential`` attribute and the one returned by
    ``value``), and the value we're interested in (the ``raw`` attribute and
    the one used when setting ``value``).

    We use an exponential fitness function to ensure that selection pressure is
    more even as the animats improve. When the actual fitness function is not
    exponential, this class handles transforming it to be so.
    """

    def __init__(self, value=0.0):
        self.value = value

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return 'ExponentialFitness({})'.format(self.raw)

    def __str__(self):
        return '(raw={}, exponential={})'.format(self.raw, self.exponential)

    @functools.total_ordering
    def __lt__(self, other):
        return self.value < other.value

    @property
    def value(self):
        return self.exponential

    @value.setter
    def value(self, v):
        self.raw = v
        self.exponential = config.FITNESS_BASE**(
            config.FITNESS_EXPONENT_ADD + config.FITNESS_EXPONENT_SCALE * v)


Game = namedtuple('Game', ['animat_states', 'world_states', 'animat_positions',
                           'trial_results'])
Mechanism = namedtuple('Mechanism', ['inputs', 'tpm'])


class Individual:

    """
    Represents an individual in the evolution.

    Args:
        genome (Iterable(int)): See attribute.

    Keyword Args:
        parent (Individual): See attribute.
        gen (int): See attribute.

    Attributes:
        genome (Iterable(int)):
            A sequence of integers in the range 0–255 that will determine the
            animat's phenotype.
        parent (Individual):
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
            single game. Updated every time a game is played.
        incorrect (int):
            The number of trials incorrectly completed by the animat during a
            single game. Updated every time a game is played.
    """

    def __init__(self, genome, parent=None, gen=0):
        self.parent = parent
        self.animat = Animat(genome)
        self.gen = gen
        self.fitness = ExponentialFitness()
        self._network = False
        # Mark whether the animat's phenotype and network need updating.
        self._dirty_phenotype = True
        self._dirty_network = True

    def __str__(self):
        string = ('Individual(gen={}, genome={}, '
                  'connectivity_matrix=\n{})'.format(
                      self.gen, np.array(self.genome), self.cm))
        return string.replace('\n', '\n' + ' ' * 11)

    def __repr__(self):
        return str(self)

    @property
    def genome(self):
        """The animat's genome."""
        return self.animat.genome

    @property
    def gen(self):
        """The generation this animat belongs to."""
        return self.animat.gen

    @gen.setter
    def gen(self, value):
        self.animat.gen = value

    @property
    def edges(self):
        """The animat's edge list."""
        self._update_phenotype()
        return self.animat.edges

    @property
    def cm(self):
        """The animat's connectivity matrix."""
        cm = np.zeros((config.NUM_NODES, config.NUM_NODES), int)
        cm[list(zip(*self.edges))] = 1
        return cm

    @property
    def tpm(self):
        """The animats's TPM."""
        self._update_phenotype()
        return np.array(self.animat.tpm).astype(float)

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
        """The number of correct catches/avoidances in the game."""
        return self.animat.correct

    @property
    def incorrect(self):
        """The number of incorrect catches/avoidances in the game."""
        return self.animat.incorrect

    def _update_phenotype(self):
        """Update the animat's phenotype if necessary. Returns whether an
        update was performed."""
        if self._dirty_phenotype:
            self.animat.update_phenotype()
            self._dirty_phenotype = False
            return True
        return False

    def __deepcopy__(self, memo):
        # Don't copy the underlying animat, parent, or PyPhi network.
        copy = Individual(genome=self.animat.genome, parent=self.parent)
        for key, val in self.__dict__.items():
            if key not in ('animat', 'parent', '_network', '_dirty_network'):
                copy.__dict__[key] = deepcopy(val, memo)
        return copy

    def start_codons(self):
        """Return the locations of start codons in the genome, if any."""
        start_codon = [42, 213]
        genome = np.array(self.genome)
        window = utils.rolling_window(genome, len(start_codon))
        occurrences = np.all((window == start_codon), axis=1)
        return np.where(occurrences)[0]

    def as_subsystem(self, state=None):
        """Return the PyPhi subsystem consisting of all the animat's nodes."""
        if state is None:
            state = [0] * config.NUM_NODES
        return pyphi.Subsystem(self.network, state, range(config.NUM_NODES))

    def brain(self, state=None):
        """Return the PyPhi subsystem consisting of the animat's hidden
        units."""
        if state is None:
            state = [0] * config.NUM_NODES
        return pyphi.Subsystem(self.network, state, _.HIDDEN_INDICES)

    def brain_and_sensors(self, state=None):
        """Return the PyPhi subsystem consisting of the animat's hidden
        units and sensors."""
        if state is None:
            state = [0] * config.NUM_NODES
        return pyphi.Subsystem(
            self.network, state, _.HIDDEN_INDICES + _.SENSOR_INDICES)

    def brain_and_motors(self, state=None):
        """Return the PyPhi subsystem consisting of the animat's hidden
        units and motors."""
        if state is None:
            state = [0] * config.NUM_NODES
        return pyphi.Subsystem(
            self.network, state, _.HIDDEN_INDICES + _.MOTOR_INDICES)

    def mutate(self):
        """Mutate the animat's genome in-place."""
        self.animat.mutate(config.MUTATION_PROB, config.DUPLICATION_PROB,
                           config.DELETION_PROB, config.MIN_GENOME_LENGTH,
                           config.MAX_GENOME_LENGTH)
        self._dirty_phenotype = True
        self._dirty_network = True

    def play_game(self, scrambled=None):
        """Return the list of state transitions the animat goes through when
        playing the game. Optionally also returns the world states and the
        positions of the animat."""
        self._update_phenotype()
        if scrambled is None:
            scrambled = config.SCRAMBLE_WORLD
        game = self.animat.play_game(_.HIT_MULTIPLIERS, _.BLOCK_PATTERNS,
                                     scramble_world=scrambled)
        assert self.animat.correct + self.animat.incorrect == _.NUM_TRIALS
        return Game(animat_states=game[0].reshape(_.NUM_TRIALS,
                                                  config.WORLD_HEIGHT,
                                                  config.NUM_NODES),
                    world_states=game[1].reshape(_.NUM_TRIALS,
                                                 config.WORLD_HEIGHT),
                    animat_positions=game[2].reshape(_.NUM_TRIALS,
                                                     config.WORLD_HEIGHT),
                    trial_results=game[3])

    def lineage(self):
        """Return a generator for the lineage of this individual."""
        yield self.animat
        ancestor = self.parent
        while ancestor is not None:
            yield ancestor.animat
            ancestor = ancestor.parent

    def mechanism(self, node_index):
        """Return the TPM of a single animat node."""
        node = self.as_subsystem().nodes[node_index]
        tpm = node.tpm[1].squeeze().astype(int)
        states = [pyphi.convert.holi_index2state(i, len(node.inputs))
                  for i in range(tpm.size)]
        return Mechanism(inputs=node.inputs,
                         tpm=np.array(list(zip(states, tpm.flatten()))))
