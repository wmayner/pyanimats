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
        int correct
        int incorrect

        void injectStartCodons(int n)
        void generatePhenotype()
        void mutateGenome(
            double mutProb, double dupProb, double delProb, int
            minGenomeLength, int maxGenomeLength);

cdef extern from 'Game.hpp':
    cdef vector[vector[int]] executeGame(
        Agent* agent, vector[int] hitMultipliers, vector[int] patterns, bool
        scrambleWorld);

@cython.freelist(1000)
cdef class Animat:
    # Hold the C++ instance that we're wrapping.
    cdef Agent *thisptr

    def __cinit__(self, genome, correct=0, incorrect=0):
        self.thisptr = new Agent(genome)
        self.thisptr.correct = correct
        self.thisptr.incorrect = incorrect

    def __dealloc__(self):
        del self.thisptr

    def __deepcopy__(self, memo):
        return Animat(self.genome)

    def __copy__(self):
        return self.__deepcopy__()

    def __reduce__(self):
        return (Animat, (self.thisptr.genome, self.thisptr.correct))

    property genome:

        def __get__(self):
            return self.thisptr.genome

    property correct:

        def __get__(self):
            return self.thisptr.correct

        def __set__(self, v):
            self.thisptr.correct = v

    property incorrect:

        def __get__(self):
            return self.thisptr.incorrect

        def __set__(self, v):
            self.thisptr.incorrect = v

    def update_phenotype(self):
        self.thisptr.generatePhenotype()

    def mutate(self, mutProb, dupProb, delProb, minGenomeLength,
               maxGenomeLength):
        self.thisptr.mutateGenome(mutProb, dupProb, delProb, minGenomeLength,
                                  maxGenomeLength);

    def play_game(self, hit_multipliers, patterns, scramble_world=False):
        self.thisptr.correct = 0
        return executeGame(self.thisptr, hit_multipliers, patterns,
                           scramble_world)


def seed(s):
    """Initialize the C++ random number generator with the given seed."""
    srand(s)
