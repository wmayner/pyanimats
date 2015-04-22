#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# distutils: language = c++

# animat.pyx


from libcpp.vector cimport vector
from libcpp cimport bool

cimport cython


cdef extern from 'Agent.hpp':

    void srand(int s)

    cdef cppclass Agent:
        Agent(vector[unsigned char] genome)

        vector[unsigned char] genome
        int hits

        void injectStartCodons(int n)
        void generatePhenotype()
        void mutateGenome(double mutProb, double dupProb, double delProb, int
                          minGenomeLength, int maxGenomeLength);

cdef extern from 'Game.hpp':
    cdef vector[vector[int]] executeGame(Agent* agent, vector[int] patterns,
                                         bool scrambleWorld)

@cython.freelist(1000)
cdef class Animat:
    # Hold the C++ instance that we're wrapping.
    cdef Agent *thisptr

    def __cinit__(self, genome, hits=0):
        self.thisptr = new Agent(genome)
        self.thisptr.hits = hits

    def __dealloc__(self):
        del self.thisptr

    def __deepcopy__(self, memo):
        return Animat(self.genome)

    def __copy__(self):
        return self.__deepcopy__()

    def __reduce__(self):
        return (Animat, (self.thisptr.genome, self.thisptr.hits))

    property genome:

        def __get__(self):
            return self.thisptr.genome

    property hits:

        def __get__(self):
            return self.thisptr.hits

        def __set__(self, v):
            self.thisptr.hits = v

    def update_phenotype(self):
        self.thisptr.generatePhenotype()

    def mutate(self, mutProb, dupProb, delProb, minGenomeLength,
               maxGenomeLength):
        self.thisptr.mutateGenome(mutProb, dupProb, delProb, minGenomeLength,
                                  maxGenomeLength);

    def play_game(self, patterns, scramble_world=False):
        self.thisptr.hits = 0
        return executeGame(self.thisptr, patterns, scramble_world)


def seed(s):
    """Initialize the C++ random number generator with the given seed."""
    srand(s)
