#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False

# animat.pyx


from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdlib cimport free
from cpython cimport PyObject, Py_INCREF 

cimport cython

import numpy as np
cimport numpy as cnp


ctypedef unsigned char uchar

# Expose #defined constants to Python.
cdef extern from 'constants.hpp':
    cdef int _NUM_NODES 'NUM_NODES'
    cdef bool _DETERMINISTIC 'DETERMINISTIC'
    cdef int _WORLD_HEIGHT 'WORLD_HEIGHT'
    cdef int _WORLD_WIDTH 'WORLD_WIDTH'
    cdef int _NUM_STATES 'NUM_STATES'
    cdef int _NUM_SENSORS 'NUM_SENSORS'
    cdef int _NUM_MOTORS 'NUM_MOTORS'
NUM_NODES = _NUM_NODES
DETERMINISTIC = _DETERMINISTIC
WORLD_HEIGHT = _WORLD_HEIGHT
WORLD_WIDTH = _WORLD_WIDTH
NUM_STATES = _NUM_STATES
NUM_SENSORS = _NUM_SENSORS
NUM_MOTORS = _NUM_MOTORS


cdef extern from 'Agent.hpp':
    void srand(int s)
    cdef cppclass Agent:
        Agent(vector[uchar] genome)

        vector[uchar] genome
        int gen
        int correct
        int incorrect

        void injectStartCodons(int n)
        void generatePhenotype()
        void mutateGenome(
            double mutProb, double dupProb, double delProb, int
            minGenomeLength, int maxGenomeLength)
        vector[vector[int]] getEdges()
        vector[vector[bool]] getTransitions()


cdef extern from 'Game.hpp':
    cdef void executeGame(
        vector[uchar] animatStates, vector[int] worldStates, vector[int]
        animatPositions, Agent* agent, vector[int] hitMultipliers, vector[int]
        patterns, bool scrambleWorld);


cdef extern from 'asvoid.hpp':
    void *asvoid(vector[uchar] *buf)
    void *asvoid(vector[int] *buf)


class StdVectorBase:
    pass


# See https://groups.google.com/d/topic/cython-users/13Bo4zXb930/discussion
cdef class UnsignedCharWrapper:

    cdef vector[uchar] *buf 

    def __cinit__(UnsignedCharWrapper self, n): 
        self.buf = NULL 

    def __init__(UnsignedCharWrapper self, cnp.intp_t n): 
        self.buf = new vector[uchar](n) 

    def __dealloc__(UnsignedCharWrapper self): 
        if self.buf != NULL: 
            del self.buf 

    def asarray(UnsignedCharWrapper self): 
        """Interpret the vector as an np.ndarray without copying the data.""" 
        base = StdVectorBase() 
        intbuf = <cnp.uintp_t> asvoid(self.buf) 
        n = <cnp.intp_t> self.buf.size()
        dtype = np.dtype(np.uint8) 
        base.__array_interface__ = dict( 
            data=(intbuf, False), 
            descr=dtype.descr, 
            shape=(n,),
            strides=(dtype.itemsize,), 
            typestr=dtype.str, 
            version=3,
        ) 
        base.vector_wrapper = self 
        return np.asarray(base) 


cdef class Int32Wrapper:

    cdef vector[int] *buf 

    def __cinit__(int self, n): 
        self.buf = NULL 

    def __init__(int self, cnp.intp_t n): 
        self.buf = new vector[int](n) 

    def __dealloc__(int self): 
        if self.buf != NULL: 
            del self.buf 

    def asarray(int self): 
        """Interpret the vector as an np.ndarray without copying the data.""" 
        base = StdVectorBase() 
        intbuf = <cnp.uintp_t> asvoid(self.buf) 
        n = <cnp.intp_t> self.buf.size()
        dtype = np.dtype(np.int32) 
        base.__array_interface__ = dict( 
            data=(intbuf, False), 
            descr=dtype.descr, 
            shape=(n,),
            strides=(dtype.itemsize,), 
            typestr=dtype.str, 
            version=3,
        ) 
        base.vector_wrapper = self 
        return np.asarray(base) 


# TODO(wmayner) does this help?
@cython.freelist(60000)
cdef class Animat:
    # Hold the C++ instance that we're wrapping.
    cdef Agent *thisptr

    def __cinit__(self, genome, gen=0, correct=0, incorrect=0):
        self.thisptr = new Agent(genome)
        self.thisptr.gen = gen
        self.thisptr.correct = correct
        self.thisptr.incorrect = incorrect

    def __dealloc__(self):
        del self.thisptr

    def __deepcopy__(self, memo):
        return Animat(self.genome, gen=self.gen,
                      correct=self.thisptr.correct,
                      incorrect=self.thisptr.incorrect)

    def __copy__(self):
        return self.__deepcopy__()

    def __reduce__(self):
        return (Animat, (self.thisptr.genome, self.thisptr.gen,
                         self.thisptr.correct, self.thisptr.incorrect))

    property genome:

        def __get__(self):
            return self.thisptr.genome

    property gen:

        def __get__(self):
            return self.thisptr.gen

        def __set__(self, v):
            self.thisptr.gen = v

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

    property edges:

        def __get__(self):
            return self.thisptr.getEdges()

    property tpm:

        def __get__(self):
            return self.thisptr.getTransitions()

    def update_phenotype(self):
        self.thisptr.generatePhenotype()

    def mutate(self, mutProb, dupProb, delProb, minGenomeLength,
               maxGenomeLength):
        self.thisptr.mutateGenome(mutProb, dupProb, delProb, minGenomeLength,
                                  maxGenomeLength);

    def play_game(self, hit_multipliers, patterns, scramble_world=False):
        # Reset the animat's hit and miss counts every time the game is played.
        self.thisptr.correct = 0
        self.thisptr.incorrect = 0
        # Calculate the size of the state transition vector, which has an entry
        # for every node state of every timestep of every trial, and initialize.
        num_timesteps = len(patterns) * 2 * WORLD_WIDTH * WORLD_HEIGHT
        cdef UnsignedCharWrapper animat_states = \
            UnsignedCharWrapper(num_timesteps * NUM_NODES)
        cdef Int32Wrapper world_states = Int32Wrapper(num_timesteps)
        cdef Int32Wrapper animat_positions = Int32Wrapper(num_timesteps)
        # Play the game, updating the animats hit and miss counts and filling
        # the given transition vector with the states the animat went through.
        executeGame(animat_states.buf[0], world_states.buf[0],
                    animat_positions.buf[0], self.thisptr, hit_multipliers,
                    patterns, scramble_world)
        # Return the state transitions and world states as NumPy arrays.
        return (animat_states.asarray(), world_states.asarray(),
                animat_positions.asarray())

         
def seed(s):
    """Initialize the C++ random number generator with the given seed."""
    srand(s)
