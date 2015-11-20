#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# make_pickle_for_pyphi.py

import os
import numpy as np
import pyphi
import pickle

from analyze import *
from individual import Individual
from utils import unique_rows
from fitness_functions import nat

# Make sure that analyze has the right parameters set for the correct animats to be loaded
# load config
#SEEDS = range(0,10)
SEEDS = [0]

# Take from analyze
OUTPUT_DIR = RESULT_PATH

def animat_from_LOD_to_dict(animat_info, seed, nLOD, save=False):
    animat = Individual(animat_info.genome, gen=animat_info.gen)

    game = animat.play_game()
    # all unique states of the animat during the 128 trials with 36 steps, sorted by occurence
    game_unique_states = unique_rows(game.animat_states, counts = True)

    # make 2 column list of state and respective state count
    unique_states = game_unique_states[0]
    rank_unique_states = game_unique_states[1]

    ranked_states = [[unique_states[i].tolist(), rank_unique_states[i]] for i in range(len(unique_states))]
      
    data = {
        'fitness': nat(animat),  
        'net': pyphi.Network(animat.tpm, connectivity_matrix=animat.cm),
        'ranked_states': ranked_states,
        'generation': animat.gen
        }

    if save:
        # directory + seed + LOD? + generation
        # TODO: LOD missing, ask Will
        # find task
        # Save the results in a pickled file for analysis with Python.
        animat_name = 'seed-' + str(seed) + '_LOD-' + str(nLOD) + '_gen-' + str(animat.gen)
        output_filepath = OUTPUT_DIR + '/seed-' + str(seed) + '/' + animat_name + '.pkl'
        #import pdb; pdb.set_trace()
        with open(output_filepath, 'wb') as f:
            pickle.dump(data, f)
        
        with open(output_filepath, 'rb') as f: 
            results = pickle.load(f)    
        print(results['fitness'])

    #import pdb; pdb.set_trace()
    

def save_LOD(LOD, nLOD, seed):
    # save a network file for each animat
    #import pdb; pdb.set_trace()
    [animat_from_LOD_to_dict(animat_info, seed, nLOD, save=True) for animat_info in LOD]


# Run everything if this file is being executed.
if __name__ == "__main__":
    for s in SEEDS:
        # for the different seeds
        animat_population = load("lineages", seed=s)
        
        # animat_population has one line of descent (LOD) 0 is final generation, end is first generation
        [save_LOD(animat_population[nLOD], nLOD, s) for nLOD in range(len(animat_population))]
