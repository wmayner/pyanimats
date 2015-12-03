
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
import logging
import time
import re



class E2ETest(unittest.TestCase):

    def test_e2e(self):
        # Locate the results dirs
        main_results_dir = "raw_results"
        results_dirs = list(
            filter(
                os.path.isdir, map(lambda x: main_results_dir + '/' + x,
                                   os.listdir(main_results_dir)
                               )
            )
        )

        #print(list(os.listdir(main_results_dir)))
        #print (list(filter(os.path.isdir, os.listdir(main_results_dir))))
        results_dirs = sorted(results_dirs)[::-1] # sort and reverse
        take = [0, 1, -1] # take these array indexes to compare

        
        lineages = []
        # consider the most recent two runs
        take_folders = [results_dirs[i] for i in take]
        all_paths = [os.path.join(t, "seed-0/lineages.pkl") for t in take_folders]
        
        for path in all_paths:
            lineages.append(load_pkl(path))

        attrs = [
            "correct",
            "gen",
            "incorrect",
            #"play_game",
            #"update_phenotype",
            "edges",
            "genome",
            #"mutate",
            "tpm",
        ]

        log = logging.getLogger("End2End Test")
        
        for seconds in take_folders:
            t = time.strftime('%Y-%m-%d %H:%M:%S',
                              time.localtime(int(re.findall('[0-9]+', seconds)[0]))) # pull the seconds out of "raw_results/12345"
            log.info('DATES USED: ' + t )
            
        for path in all_paths:
            log.info('FOLDERS USED: ' + path )
            
        a = lineages[0][0] # the first lineage in the first run

        log.info('# ANIMATS: ' + str(len(a)))
        
        for i in range(len(lineages)-1): # compare each lineage against the first
            b = lineages[i][0]

            self.assertEqual( len(a), len(b))
            for aanim, banim in zip(a, b):
                for attr in attrs:
                    self.assertEqual(
                        getattr(aanim, attr),
                        getattr(banim, attr)
                    )
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
