#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# analysis.py

import pickle
import numpy as np
from glob import glob

import evolve
from parameters import FITNESS_BASE


# Get CPP fitnesses.
cpp_lods = {}
for f in glob('../animats/results/current/seed-*/seed-*_LOD.csv'):
    cpp_lods[f] = np.genfromtxt(f, delimiter=',', dtype=int, skip_header=1)

cpp_correct = [lod[-1][1] for lod in cpp_lods.values()]
cpp_fit = np.array([FITNESS_BASE**correct for correct in cpp_correct])

# Get Python fitnesses.
py_data = {}
for f in glob('./results/*final*'):
    with open(f, 'rb') as dill:
        py_data[f] = pickle.load(dill)

py_fit = np.array([ind[1].values[0] for d in py_data.values() for ind in d])
