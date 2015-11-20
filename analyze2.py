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


------------------
Plot task:
python pyanimats.py \
./raw_results/0.0.22/plot_trial/seed-$i \
./tasks/1-3-1-3.yml \
-n 10000 \
-i 100 \
-s $i


'''


import os

import pickle
import json

from utils import unique_rows
from fitness_functions import nat
from individual import Individual


from analyze import * # TODO (josh): wh does this need to be imported, and not even used, in order for stuff to work? 
import pyphi




# Utility Functions ###################################
def load_pkl(path):
    ''' load a pkl file as an object in memory '''
    # path = os.path.join(directory, filename)
    with open(path, 'rb') as f:
        loaded = pickle.load(f)
    return loaded

def load_pkls(paths):
    ''' loads pkls as a list of objects '''
    ret = []
    for path in paths:
        ret.append(load_pkl(path))
    return ret

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
      * see tests for usage.
    '''
    return_list = []
    for seed in seed_list:
        row = {}
        for key, func in columns.items():
            row[key] = func(seed)

        return_list.append(row)
    return return_list


def extract_list_from_dicts(dicts, *keys):
    '''
    When you have a list of similar shaped dicts, and
    want one value from each item, use this.

    Args:
      listy: a list of dicts, ex:
             [{'a':..., 'b':..., 'c':{ 'inside': ...}}, <same shaped dicts>]
      *keys: the path to the value you want

    Returns:
      a list of the values at the end of the key's "path"
      * see test for example

    '''

    ret = []
    for item in dicts:
        temp = item
        for key in keys:
            temp = temp[key]
        ret.append(temp)
    return ret






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
        'net': pyphi.Network(individual.tpm, connectivity_matrix=individual.cm),
        'ranked_states': individual_to_ranked_states(individual),
        'generation': individual.gen
    }
    return data



# Load some stuff for easy access in iPython #################################
SEEDS = range(10)
PKL_DIR = "./raw_results/0.0.22/plot_trial/seed-%d"
FILENAME = "lineages.pkl"
COLUMNS = {
    'gen': lambda x: x.gen,
    'genome_sum': lambda x: sum(x.genome),
    'percent_correct': lambda x: x.correct / (x.correct + x.incorrect),
    'individual_vars': individual_vars,
}


# list of paths to pkls
all_paths = [os.path.join(PKL_DIR%seed,
                          FILENAME)
                          for seed in SEEDS]

# list of objects from the lineages, reference to what's in those pkls:
# list = [<seed1>, <seed2>, ...]
# <seed> = [<lineage1>, <lineage2>, ...]
# <lineage> = [<gen1's Animat>, <gen2's>, ...]
all_lineages = load_pkls(all_paths)

# list of first lineages for each seed
all_first_lineages = extract_list_from_dicts(all_lineages, 0)

# list of expanded lineages, through out every SKIP generation
SKIP = 1 # 1 = don't skip any
expanded_lineages = [expand_list_by_func(lineage[::SKIP], COLUMNS)
                     for lineage in all_first_lineages]

# list of fitnesses along each seed's first lineage
fitnesses_by_seed = [extract_list_from_dicts(table,
                                             'individual_vars',
                                             'fitness')
                     for table in expanded_lineages]

# avg fitness across seeds, by generation
fitness_avgs = [sum(gen)/len(gen)
                for gen in zip(*fitnesses_by_seed)]

# reverse, so first index is first generation
fitness_avgs = fitness_avgs[::-1]

with open('fitness_avgs.csv', 'w') as f:
    for i, avg in enumerate(fitness_avgs):
        f.write(str(i) + ',' + str(avg) + '\n')


