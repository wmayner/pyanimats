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
cimport numpy as np
np.import_array()

import ctypes


ctypedef unsigned char NodeState
ctypedef unsigned char Nucleotide

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
        Agent(vector[Nucleotide] genome)

        vector[Nucleotide] genome
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
    cdef int* executeGame(
        Agent* agent, vector[int] hitMultipliers, vector[int] patterns, bool
        scrambleWorld);


cdef class ArrayWrapper:
    """Creates a NumPy array from already allocated memory."""
    # See https://gist.github.com/GaelVaroquaux/1249305
    cdef void* data_ptr
    cdef int size
    cdef int dtype

    cdef set_data(self, int size, int dtype, void* data_ptr):
        """Set the data of the array.

        This cannot be done in the constructor as it must recieve C-level
        arguments.

        Parameters:
            size (int) Length of the array.
            data_ptr (void*): Pointer to the data.
        """
        self.data_ptr = data_ptr
        self.size = size
        self.dtype = dtype

    def __array__(self):
        """Here we use the ``__array__`` method, that is called when NumPy
        tries to get an array from the object."""
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.size
        # Create a 1D array, of length 'size'
        ndarray = np.PyArray_SimpleNewFromData(1, shape, self.dtype,
                                               self.data_ptr)
        return ndarray

    def __dealloc__(self):
        """Frees the array. This is called by Python when all the references to
        the object are gone. """
        free(<void*>self.data_ptr)


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
        self.thisptr.correct = 0
        self.thisptr.incorrect = 0

        cdef int* stateTransitions = executeGame(
            self.thisptr, hit_multipliers, patterns, scramble_world)

        num_trials = len(patterns) * 2 * WORLD_WIDTH
        size = num_trials * WORLD_HEIGHT * NUM_NODES

        array_wrapper = ArrayWrapper()
        array_wrapper.set_data(size, np.NPY_UINT8, <void*> stateTransitions) 

        cdef np.ndarray result = np.array(array_wrapper, copy=False)
        # Assign our object to the 'base' of the ndarray object.
        result.base = <PyObject*> array_wrapper
        # Increment the reference count, as the above assignement was done in
        # C, and Python does not know that there is this additional reference.
        Py_INCREF(array_wrapper)
        return result


def seed(s):
    """Initialize the C++ random number generator with the given seed."""
    srand(s)
