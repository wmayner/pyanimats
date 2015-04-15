#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# distutils: language = c++


from libcpp.vector cimport vector
from libcpp cimport bool

INIT_GENOME_SIZE = 5000


cdef extern from 'HMM.hpp':
    cdef cppclass HMM:
        HMM(vector[unsigned char] &genome, int start)

        vector[unsigned int] sums
        vector[unsigned char] ins, outs
        unsigned char numInputs, numOutputs

        void update(unsigned char *currentStates, unsigned char *nextStates)


cdef extern from 'Agent.hpp':
    cdef cppclass Agent:
        Agent()
        vector[HMM] hmmus
        vector[unsigned char] genome
        Agent *ancestor
        unsigned int nrPointingAtMe
        int hits
        int NUM_NODES

        void setupEmptyAgent(int nucleotides)
        void setupPhenotype()
        void injectStartCodons(int n)
        void resetBrain()
        void updateStates()


cdef extern from 'Game.hpp':
    cdef vector[vector[int]] executeGame(Agent* agent, vector[int] patterns,
                                         bool scrambleWorld)


cdef class Animat:
    # Hold the C++ instance that we're wrapping.
    cdef Agent *thisptr

    def __cinit__(self):
        self.thisptr = new Agent()
        self.thisptr.setupEmptyAgent(INIT_GENOME_SIZE)
        self.thisptr.injectStartCodons(4)

    def __dealloc__(self):
        del self.thisptr

    @property
    def genome(self):
        return self.thisptr.genome

    @property
    def hits(self):
        return self.thisptr.hits

    def play_game(self, patterns, scramble_world=False):
        return executeGame(self.thisptr, patterns, scramble_world)
