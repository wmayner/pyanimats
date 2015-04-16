#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# distutils: language = c++

# animat.pyx


from libcpp.vector cimport vector
from libcpp cimport bool


cdef extern from 'Agent.hpp':
    cdef cppclass Agent:
        Agent(vector[unsigned char] genome)

        vector[unsigned char] genome
        int hits

        void injectStartCodons(int n)


cdef extern from 'Game.hpp':
    cdef vector[vector[int]] executeGame(Agent* agent, vector[int] patterns,
                                         bool scrambleWorld)


cdef class Animat:
    # Hold the C++ instance that we're wrapping.
    cdef Agent *thisptr

    def __cinit__(self, genome):
        self.thisptr = new Agent(genome)

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
