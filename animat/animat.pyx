#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# distutils: language = c++

# animat.pyx


from libcpp.vector cimport vector
from libcpp cimport bool


cdef extern from 'Agent.hpp':

    void srand(int s)

    vector[unsigned char] mutateGenome(
        vector[unsigned char] genome, double mutProb, double dupProb, double
        delProb, int minGenomeLength, int maxGenomeLength);

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


# cdef extern from 'mutate.hpp':
#     vector[unsigned char] mutateGenome(
#         vector[unsigned char] genome, double mutProb, double dupProb, double
#         delProb, int minGenomeLength, int maxGenomeLength)


# cdef class Individual:
#     cdef vector[unsigned char] genome

#     def __cinit__(self, genome, fitness):
#         self.genome = genome
#         self.fitness = fitness

#     def __dealloc__(self):
#         del self.fitness

#     def mutate(self, mut_prob, dup_prob, del_prob, min_genome_length,
#                max_genome_length):
#         mutateGenome(self.genome, mut_prob, dup_prob, del_prob,
#                      min_genome_length, max_genome_length)



cdef class Animat:
    # Hold the C++ instance that we're wrapping.
    cdef Agent *thisptr

    def __cinit__(self, genome, hits=0):
        self.thisptr = new Agent(genome)
        self.thisptr.hits = hits

    def __dealloc__(self):
        del self.thisptr

    def __copy__(self):
        return Animat(self.genome)

    def __deepcopy__(self, memo):
        return self.__copy__()

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

    def play_game(self, patterns, scramble_world=False):
        self.thisptr.hits = 0
        return executeGame(self.thisptr, patterns, scramble_world)

    def update_phenotype(self):
        self.thisptr.generatePhenotype()

    def mutate(self, mutProb, dupProb, delProb, minGenomeLength,
               maxGenomeLength):
        self.thisptr.mutateGenome(mutProb, dupProb, delProb, minGenomeLength,
                                  maxGenomeLength);


def mutate(genome, mutProb, dupProb, delProb, minGenomeLength,
           maxGenomeLength):
    return mutateGenome(genome, mutProb, dupProb, delProb, minGenomeLength,
                        maxGenomeLength);



def seed(s):
    """Initialize the C++ random number generator with the given seed."""
    srand(s)
