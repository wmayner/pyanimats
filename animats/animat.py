#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# animat.py

from bitarray import bitarray

from .constants import START_CODON, ZERO_STATE
from .plg import ProbabilisticLogicGate

LEFT = bitarray('10')
RIGHT = bitarray('01')


class Animat:

    """A virtual organism."""

    def __init__(self, genome):
        self.genome = bytearray(genome)
        self.state = bitarray(ZERO_STATE)
        self.correct, self.incorrect = 0, 0
        self.fitness = None

        # Find all the start codons.
        gene_indices = []
        G = len(genome)
        for i in range(G):
            if genome[i: (i + 2) % G] == START_CODON:
                gene_indices.append((i + 2) % G)
        # Make a PLG for each one.
        self.gates = [ProbabilisticLogicGate(genome, index) for index in
                      gene_indices]

    def update_state(self):
        next_state = bitarray(ZERO_STATE)
        for gate in self.gates:
            next_state |= gate.get_next_state(self.state)
        self.state = next_state
        return self.state

    def __repr__(self):
        return 'Animat(' + str(self.genome) + ')'

    def __str__(self):
        return ('Animat(\n\t' + '\n\t'.join([
            "Genome: " + str(self.genome),
            "State: " + str(self.state),
            "PLGs: \n" + '\n'.join(map(str, self.gates))
        ]) + '\n)')
