#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# individual.py

import numpy as np
from copy import deepcopy
from parameters import SEED, INIT_GENOME
import animat
animat.seed(SEED)
from animat import Animat


class Individual:

    def __init__(self, genome=INIT_GENOME, parent=None):
        self.parent = parent
        self.animat = Animat(genome)
        self.animat.update_phenotype()
        self.cm = np.zeros((animat.NUM_NODES, animat.NUM_NODES), int)
        self.cm[list(zip(*self.animat.edges))] = 1

    @property
    def genome(self):
        return self.animat.genome

    @property
    def edges(self):
        return self.animat.edges

    def __deepcopy__(self, memo):
        # Don't copy the animat or the parent.
        copy = Individual(genome=self.animat.genome, parent=self.parent)
        for key, val in self.__dict__.items():
            if key not in ('animat', 'parent'):
                copy.__dict__[key] = deepcopy(val, memo)
        return copy

    def lineage(self):
        """Return a generator for the lineage of this individual."""
        yield self.animat
        ancestor = self.parent
        while ancestor is not None:
            yield ancestor.animat
            ancestor = ancestor.parent
