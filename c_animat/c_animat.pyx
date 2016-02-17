#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False

# c_animat.pyx


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
    cdef int _CORRECT_CATCH 'CORRECT_CATCH'
    cdef int _WRONG_CATCH 'WRONG_CATCH'
    cdef int _CORRECT_AVOID 'CORRECT_AVOID'
    cdef int _WRONG_AVOID 'WRONG_AVOID'
    cdef int _MIN_BODY_LENGTH 'MIN_BODY_LENGTH'
    cdef int _START_CODON_NUCLEOTIDE_ONE 'START_CODON_NUCLEOTIDE_ONE'
    cdef int _START_CODON_NUCLEOTIDE_TWO 'START_CODON_NUCLEOTIDE_ONE'
CORRECT_CATCH = _CORRECT_CATCH
WRONG_CATCH = _WRONG_CATCH
CORRECT_AVOID = _CORRECT_AVOID
WRONG_AVOID = _WRONG_AVOID
MIN_BODY_LENGTH = _MIN_BODY_LENGTH
START_CODON_NUCLEOTIDE_ONE = _START_CODON_NUCLEOTIDE_ONE
START_CODON_NUCLEOTIDE_TWO = _START_CODON_NUCLEOTIDE_ONE


cdef extern from 'Agent.hpp':
    void srand(int s)
    cdef cppclass Agent:
        Agent(vector[uchar] genome, int numSensors, int numHidden, int
              numMotors, bool deterministic)

        int mNumSensors
        int mNumHidden
        int mNumMotors
        int mNumNodes
        int mNumStates
        int mBodyLength
        bool mDeterministic

        vector[uchar] genome

        void injectStartCodons(int n)
        void generatePhenotype()
        void mutateGenome(
            double mutProb, double dupProb, double delProb, int
            minGenomeLength, int maxGenomeLength, int minDupDelLength, 
            int maxDupDelLength)
        vector[vector[int]] getEdges()
        vector[vector[bool]] getTransitions()


cdef extern from 'Game.hpp':
    cdef vector[int] executeGame(
        vector[uchar] animatStates, vector[int] worldStates, 
        vector[int] animatPositions, vector[int] trialResults, Agent* agent,
        vector[int] hitMultipliers, vector[int] patterns, int worldWidth, 
        int worldHeight, bool scrambleWorld)


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


def seed(s):
    """Initialize the C++ random number generator with the given seed."""
    srand(s)


# TODO(wmayner) does this help?
@cython.freelist(60000)
cdef class cAnimat:
    # Hold the C++ instance that we're wrapping.
    cdef Agent *thisptr

    def __cinit__(self, genome, numSensors, numHidden, numMotors,
                  deterministic):
        self.thisptr = new Agent(genome, numSensors, numHidden, numMotors,
                                 deterministic)
        self.thisptr.generatePhenotype()

    def __dealloc__(self):
        del self.thisptr

    def __deepcopy__(self, memo):
        return cAnimat(self.genome, self.mNumSensors, self.mNumHidden,
                       self.mNumMotors, self.mDeterministic)

    def __copy__(self):
        return self.__deepcopy__()

    def __reduce__(self):
        # When pickling, simply regenerate an instance.
        # NOTE: This means that changes in the implementation of cAnimat that
        # occur between pickling and unpickling can cause a SILENT change in
        # behavior!
        return (cAnimat, (self.thisptr.genome, self.thisptr.mNumSensors,
                          self.thisptr.mNumHidden, self.thisptr.mNumMotors,
                          self.thisptr.mDeterministic))

    property genome:
        def __get__(self):
            return self.thisptr.genome

    property num_sensors:
        def __get__(self):
            return self.thisptr.mNumSensors

    property num_hidden:
        def __get__(self):
            return self.thisptr.mNumHidden

    property num_motors:
        def __get__(self):
            return self.thisptr.mNumMotors

    property num_nodes:
        def __get__(self):
            return self.thisptr.mNumNodes

    property num_states:
        def __get__(self):
            return self.thisptr.mNumStates

    property deterministic:
        def __get__(self):
            return self.thisptr.mDeterministic

    property body_length:
        def __get__(self):
            return self.thisptr.mBodyLength

    property edges:
        def __get__(self):
            return self.thisptr.getEdges()

    property tpm:
        def __get__(self):
            return self.thisptr.getTransitions()

    def mutate(self, mutProb, dupProb, delProb, minGenomeLength,
               maxGenomeLength, minDupDelLength, maxDupDelLength):
        self.thisptr.mutateGenome(mutProb, dupProb, delProb, minGenomeLength,
                                  maxGenomeLength, minDupDelLength,
                                  maxDupDelLength);
        # Update the animat's phenotype after changing the genome.
        self.thisptr.generatePhenotype();

    def play_game(self, hit_multipliers, patterns, worldWidth, worldHeight,
                  scramble_world=False):
        # Calculate the size of the state transition vector, which has an entry
        # for every node state of every timestep of every trial, and initialize.
        num_trials = len(patterns) * 2 * worldWidth
        num_timesteps = num_trials * worldHeight
        cdef UnsignedCharWrapper animat_states = \
            UnsignedCharWrapper(num_timesteps * self.num_nodes)
        cdef Int32Wrapper world_states = Int32Wrapper(num_timesteps)
        cdef Int32Wrapper animat_positions = Int32Wrapper(num_timesteps)
        cdef Int32Wrapper trial_results = Int32Wrapper(num_trials)
        # Play the game, updating the animats hit and miss counts and filling
        # the given transition vector with the states the animat went through.
        correct, incorrect = executeGame(
            animat_states.buf[0], world_states.buf[0], animat_positions.buf[0],
            trial_results.buf[0], self.thisptr, hit_multipliers, patterns,
            worldWidth, worldHeight, scramble_world)
        # Return the state transitions and world states as NumPy arrays.
        return (animat_states.asarray(), world_states.asarray(),
                animat_positions.asarray(), trial_results.asarray(), correct,
                incorrect)
