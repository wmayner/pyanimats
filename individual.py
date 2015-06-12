#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# individual.py

import numpy as np
from copy import deepcopy
from parameters import params
from animat import Animat


class Individual:

    def __init__(self, genome, parent=None, gen=0):
        self.parent = parent
        self.animat = Animat(genome)
        self.gen = gen
        # Mark whether the animat's phenotype needs updating.
        self._dirty_phenotype = True

    @property
    def genome(self):
        """The animat's genome."""
        return self.animat.genome

    @property
    def gen(self):
        """The generation the animat was born."""
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
        cm = np.zeros((params.NUM_NODES, params.NUM_NODES), int)
        cm[list(zip(*self.edges))] = 1
        return cm

    @property
    def tpm(self):
        """The animats's TPM."""
        self._update_phenotype()
        return np.array(self.animat.tpm).astype(float)

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
        # Don't copy the animat or the parent.
        copy = Individual(genome=self.animat.genome, parent=self.parent)
        for key, val in self.__dict__.items():
            if key not in ('animat', 'parent'):
                copy.__dict__[key] = deepcopy(val, memo)
        return copy

    def mutate(self):
        """Mutate the animat's genome in-place."""
        self.animat.mutate(params.MUTATION_PROB, params.DUPLICATION_PROB,
                           params.DELETION_PROB, params.MIN_GENOME_LENGTH,
                           params.MAX_GENOME_LENGTH)
        self._dirty_phenotype = True

    def play_game(self):
        """Return the list of state transitions the animat goes through when
        playing the game."""
        self._update_phenotype()
        transitions = self.animat.play_game(
            params.HIT_MULTIPLIERS,
            params.BLOCK_PATTERNS,
            scramble_world=params.SCRAMBLE_WORLD)
        # TODO remove this assertion at some point for speed
        # Check that everything adds up.
        assert self.animat.correct + self.animat.incorrect == params.NUM_TRIALS
        return transitions

    def lineage(self):
        """Return a generator for the lineage of this individual."""
        yield self.animat
        ancestor = self.parent
        while ancestor is not None:
            yield ancestor.animat
            ancestor = ancestor.parent
