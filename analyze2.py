'''

A module for loading and manipulating results

NOTE:
This is similar(ish) to `analyze.py` and they should be combined at some point, but I'm building this out separately for now

---------------------------

# Output of pyanimats.py #

lineages.pkl
((<Animat>, ...), ...)

hof.pkl
[<Animat>, ...]

logbook
[{'gen': 0}, ...]

Animat
  - correct
  - gen
  - incorrect
  - play_game
  - update_phenotype
  - edges
  - genome
  - mutate
  - tpm 


Desired Outputs:
  - list of states as it played the game
  - fitness
  - tpm
  - cm
  - pyphi results

'''


import os

import pickle
import json

from utils import unique_rows
from fitness_functions import nat
from individual import Individual


from analyze import * # TODO (josh): why the heck does this need to be imported, and not even used, in order for stuff to work?
import pyphi

# Constants ###########################################
PKL_DIR = "./test/end to end/raw_results/0.0.22/initial_tests/seed-153"
LINEAGES = "lineages.pkl"


# Utility Functions ###################################
def load_pkl(directory=PKL_DIR, filename=LINEAGES):
    ''' load a pkl file as an object in memory '''
    path = os.path.join(directory, filename)
    with open(path, 'rb') as f:
        loaded = pickle.load(f)
    return loaded

def expand_list_by_func(seed_list, columns):
    '''
    Takes a list of things, and "expands" it into a table of columns,
    where each column contains some interesting data.
    This is useful for turning things like a list of genomes into
    a table containing fitness values, phi, etc.

    Args:
      seed_list: a list of things
      columns: a dictionary that maps column names to a function that
               acts over some arguments (eg the genome, or an Individual)
               ex:
                {
                'phi': pyphi.compute.big_phi,
                'fitness': lambda x, y: x/(x+y), (seed.correct, seed.incorrect)),
                ...
                }
    
    Returns:
      A list of dicts
    '''
    return_list = []
    for seed in seed_list:
        row = {}
        for key, func in columns.items():
            row[key] = func(seed)

        return_list.append(row)
    return return_list

def extract_value_from_list(listy, *keys):
    '''
    When you have a list of similar shaped dicts, and
    want one value from each item, use this.

    Args:
      listy: a list of dicts, ex:
             [{'a':..., 'b':..., 'c':{ 'inside': ...}}, <same shaped dicts>]
      *keys: the path to the value you want

    Returns:
      a list of the values at the end of the keys "path"

    '''

    ret = []
    for item in listy:
        XXX






# Functions for generating data from Animats #################################
# Normally, these would be user defined, and used in the `columns` dict

def individual_to_ranked_states(individual):
    '''
    take an individual, get the states they've gone through
    while playing a game, count the occurences, and return
    a sorted list of [<state>, occurences]
    '''
    game = individual.play_game()
    # all unique states of the animat during the 128 trials with 36 steps, sorted by occurence
    game_unique_states = unique_rows(game.animat_states, counts = True)

    # make 2 column list of state and respective state count
    unique_states = game_unique_states[0]
    rank_unique_states = game_unique_states[1]

    ranked_states = [[unique_states[i].tolist(), rank_unique_states[i]] for i in range(len(unique_states))]
    return ranked_states

def individual_vars(animat):
    ''' get a dictionary of vars from an Animat object '''
    individual = Individual(animat.genome, gen=animat.gen)
    data = {
        'fitness': nat(individual),  
        # 'net': pyphi.Network(individual.tpm, connectivity_matrix=individual.cm),
        # 'ranked_states': individual_to_ranked_states(individual),
        # 'generation': individual.gen
    }
    return data



# Load some stuff for easy access in iPython #################################
lineages = load_pkl()

columns = {
    # 'gen': lambda x: x.gen,
    # 'genome_sum': lambda x: sum(x.genome),
    # 'percent_correct': lambda x: x.correct / (x.correct + x.incorrect),
    'individual_vars': individual_vars,
}

g = expand_list_by_func(lineages[0][::100], columns)
