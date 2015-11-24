'''

Check that the results from running `./e2etest` are the same between the most recent run, and the last run

NOTE: This is a terrible way to test things. I'm only doing it like this during refactoring to set up a "guardrail", as I try to despaghettify the code.


Notes:
Seeding is never being called
* just call it anywhere
* animats.pyx srand for setting the seed in C

'''

import sys
sys.path.append("../../")
from individual import Individual

import os
import pickle
import unittest
import json




class E2ETest(unittest.TestCase):

    def test_e2e(self):
        # Locate the results dirs
        main_results_dir = "raw_results"
        results_dirs = list(os.listdir(main_results_dir))
        results_dirs = sorted(results_dirs)[::-1] # sort and reverse


        
        # consider the very first run
        lineages = []
        lineages.append(load_pkl(os.path.join(main_results_dir,
                                              results_dirs[-1],
                                              "seed-0/lineages.pkl"
                                          )))
        # consider the most recent two runs
        for i in range(2):
            lineages.append(load_pkl(os.path.join(main_results_dir,
                                             results_dirs[i],
                                             "seed-0/lineages.pkl"
                                         )))

        attrs = [
            "correct",
            "gen",
            "incorrect",
            #"play_game",
            #"update_phenotype",
            "edges",
            #"genome",
            #"mutate",
            #"tpm",
        ]
        a = lineages[0][0] # the first lineage in the first run
        for i in range(len(lineages)-1): # compare each lineage against the first
            b = lineages[i][0]

            self.assertEqual( len(a), len(b))
            for aanim, banim in zip(a, b):
                for attr in attrs:
                    print(attr, getattr(aanim, attr), getattr(banim, attr))
                    
                    self.assertEqual(
                        getattr(aanim, attr),
                        getattr(banim, attr)
                    )
                print()
        # import ipdb; ipdb.set_trace()



# Utility Functions ##############################
def load_pkl(path):
    ''' load a pkl file as an object in memory '''
    # path = os.path.join(directory, filename)
    with open(path, 'rb') as f:
        loaded = pickle.load(f)
    return loaded

def load_json(path):
    with open(path, 'rb') as f:
        loaded = json.load(f)
    return loaded
