#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pyanimats.py

"""
PyAnimats
---------
Evolve animats.

Usage:
    pyanimats.py <output_dir> [options]
    pyanimats.py -h | --help
    pyanimats.py -v | --version
    pyanimats.py --list-fitness-funcs
    pyanimats.py --num-sensors

Options:
    -s, --seed=SEED             Random number generator seed [default: 0]
    -f, --fitness=FUNC          Fitness function [default: nat]
    -m, --mut-prob=PROB         Nucleotide mutation probability
                                  [default: 0.005]
    -p, --pop-size=SIZE         Population size [default: 100]
    -d, --log-interval=FREQ     Logbook recording interval (generations)
                                  [default: 1]
    -i, --num-samples=NUM       Number of individuals to sample from evolution
                                  (0 saves entire lineage) [default: 0]
    -t, --snapshot=FREQ         Snapshot interval (seconds) [default: 0]
    -o, --min-snapshots=NUM     Minimum number of snapshots to take
                                  [default: 0]
    -l, --stdout-interval=FREQ  Status-printing interval (generations)
                                  [default: 1]
    -j, --jumpstart=NUM         Begin with this many start codons [default: 0]
    -g, --init-genome=PATH      Path to a lineage file for an intial genome
    -a, --all-lineages          Save lineages of entire final population
    -c, --config=PATH           Path to a configuration file to load
        --scramble              Randomly rearrange the world in each trial
        --dup-prob=PROB         Duplication probability [default: 0.05]
        --del-prob=PROB         Deletion probability [default: 0.02]
        --max-length=LENGTH     Maximum genome length [default: 10000]
        --min-length=LENGTH     Minimum genome length [default: 1000]
        --min-dup-del=LENGTH    Minimum length of duplicated/deleted genome
                                  part [default: 15]
        --fit-base=FLOAT        Base used in the fitness function (see
                                  --list-fitness-funcs) [default: 1.02]
        --fit-exp-add=FLOAT     Add this term to the fitness exponent
        --fit-exp-scale=FLOAT   Scale raw fitness values before they're used as
                                  an exponent
        --profile=PATH          Profile performance and store results at PATH
                                  [default: profiling/profile.pstats]
"""

__version__ = '0.0.23'

import os
import pickle
import json
import yaml
import random
import utils
from time import time
from pprint import pprint
import numpy as np
import cProfile

import fitness_functions
from individual import Individual
from deap import base, tools

import configure


PROFILING = False


def select(individuals, k):
    """Select *k* individuals from the given list of individuals using the
    variant of roulette-wheel selection used in the old C++ code.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.

    This function uses the :func:`~random.random` function from the built-in
    :mod:`random` module."""
    max_fitness = max([ind.fitness.value for ind in individuals])
    chosen = []
    for i in range(k):
        done = False
        while not done:
            candidate = random.choice(individuals)
            done = random.random() <= (candidate.fitness.value /
                                       max_fitness)
        chosen.append(candidate)
    return chosen


def mutate(ind):
    ind.mutate()
    return (ind,)
mutate.__doc__ = Individual.mutate.__doc__


def main(arguments):
    # Load the main YAML file
    
    EXPERIMENT_DIR = "test/end_to_end/raw_results" # this should be the main/only thing passed to pyanimats.py

    # ToDo: add user arguments into `experiment` object
    
    experiment_yaml = "experiment.yml"
    with open(os.path.join(EXPERIMENT_DIR, experiment_yaml), 'r') as f:
        experiment = yaml.load(f)
    experiment['arguments'] = arguments
    

    # Set the global random seed
    random.seed(experiment['seed'])

    



    
    # Handle configuration
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    # Final output and snapshots will be written here.
    OUTPUT_DIR = arguments['<output_dir>'] # ToDo: this still needs to be fixed, should reference the folder where experiments.yml is
    # OUTPUT_DIR = EXPERIMENT_DIR
    del arguments['<output_dir>']

    # Ensure profile directory exists and set profile flag.
    PROFILING = False
    if "profile" in experiment:
        profile_filepath = experiment['profile']
        if profile_filepath:
            PROFILING = True
            utils.ensure_exists(os.path.dirname(profile_filepath))
    del arguments['--profile']


    
    # Logbooks will be updated at this interval.
    LOGBOOK_RECORDING_INTERVAL = int(arguments['--log-interval'])
    del arguments['--log-interval']

    # Individuals will be recorded in the lineage at this interval.
    NUM_INDIVIDUAL_SAMPLES = int(arguments['--num-samples'])
    del arguments['--num-samples']

    # Status will be printed at this interval.
    STATUS_PRINTING_INTERVAL = int(arguments['--stdout-interval'])
    del arguments['--stdout-interval']

    # Get the minimum number of snapshots to be taken.
    MIN_SNAPSHOTS = int(arguments['--min-snapshots'])
    del arguments['--min-snapshots']

    # Get the interval at which to take snapshots.
    SNAPSHOT_TIME_INTERVAL = float(arguments['--snapshot'])
    if SNAPSHOT_TIME_INTERVAL <= 0:
        SNAPSHOT_TIME_INTERVAL = float('inf')
    del arguments['--snapshot']

    # Whether or not to save every individual in the population, or just the best one.
    SAVE_ALL_LINEAGES = arguments['--all-lineages']
    del arguments['--all-lineages']

    # Load and print configuration.
    configure.from_args(arguments)
    print('Configuration:')
    pprint(experiment)

    # Snapshots will be written to disk at this interval.
    if MIN_SNAPSHOTS <= 0:
        SNAPSHOT_GENERATION_INTERVAL = float('inf')
    else:
        SNAPSHOT_GENERATION_INTERVAL = experiment['ngen'] // MIN_SNAPSHOTS

    # Helper functions
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def save_data(output_dir, gen, config, pop, logbook, hof, elapsed):
        '''
        Function for saving data at the end of a run, including:
          - lineages.pkl
          - logbook.pkl
          - hof.pkl
        '''
        # Ensure output directory exists.
        utils.ensure_exists(output_dir)
        # Collect lineages.
        if SAVE_ALL_LINEAGES:
            to_save = pop
        else:
            to_save = [max(pop, key=lambda ind: ind.fitness.value)]
        step = (1 if NUM_INDIVIDUAL_SAMPLES <= 0
                else max(gen // NUM_INDIVIDUAL_SAMPLES, 1))
        lineages = tuple(tuple(ind.lineage())[::step] for ind in to_save)
        # Save config and metadata as JSON.
        data_json = {
            'config': experiment,
            'metadata': {
                'elapsed': round(elapsed, 2),
                'version': __version__
            }
        }
        for key in data_json:
            with open(os.path.join(output_dir, str(key) + '.json'), 'w') as f:
                json.dump(data_json[key], f, indent=2, separators=(',', ': '))
        # Pickle everything else.
        data_pickle = {
            'lineages': lineages,
            'logbook': logbook,
            'hof': [ind.animat for ind in hof],
        }
        for key in data_pickle:
            with open(os.path.join(output_dir, str(key) + '.pkl'), 'wb') as f:
                pickle.dump(data_pickle[key], f)

    # Setup
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    toolbox = base.Toolbox()

    # Register the various genetic algorithm components to the toolbox.
    toolbox.register('individual', Individual, experiment)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', fitness_functions.__dict__[experiment['fitness_function']])
    toolbox.register('select', select)
    toolbox.register('mutate', mutate)

    # Create statistics trackers.
    fitness_stats = tools.Statistics(key=lambda ind: ind.fitness.raw)
    fitness_stats.register('max', np.max)

    real_fitness_stats = tools.Statistics(key=lambda ind: ind.fitness.value)
    real_fitness_stats.register('max', np.max)

    correct_stats = tools.Statistics(key=lambda ind: (ind.animat.correct,
                                                      ind.animat.incorrect))
    correct_stats.register('correct', lambda x: np.max(x, 0)[0])
    correct_stats.register('incorrect', lambda x: np.max(x, 0)[1])

    # Stats objects for alternate matching measures.
    alt_fitness_stats = tools.Statistics(key=lambda ind: ind.alt_fitness)
    alt_fitness_stats.register('weighted', lambda x: np.max(x, 0)[0])
    alt_fitness_stats.register('unweighted', lambda x: np.max(x, 0)[1])

    # Initialize a MultiStatistics object for convenience that allows for only
    # one call to `compile`.
    if experiment['fitness_function'] == 'mat':
        mstats = tools.MultiStatistics(correct=correct_stats,
                                       fitness=fitness_stats,
                                       real_fitness=real_fitness_stats,
                                       alt_fitness=alt_fitness_stats)
    else:
        mstats = tools.MultiStatistics(correct=correct_stats,
                                       fitness=fitness_stats,
                                       real_fitness=real_fitness_stats)

    # Initialize logbooks and hall of fame.
    logbook = tools.Logbook()
    hall_of_fame = tools.HallOfFame(maxsize=experiment['popsize'])

    def print_status(line, time):
        print('[Seed {}] '.format(experiment['seed']), end='')
        print(line, utils.compress(time))

    print('\nSimulating {} generations...\n'.format(experiment['ngen']))

    if PROFILING:
        pr = cProfile.Profile()
        pr.enable()
    sim_start = time()

    # Simulation
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def multi_fit_evaluate(pop, gen):
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fitness in zip(pop, fitnesses):
            ind.fitness.value = fitness[0]
            ind.alt_fitness = fitness[1:]

    def single_fit_evaluate(pop, gen):
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fitness in zip(pop, fitnesses):
            ind.fitness.value = fitness

    evaluate = (multi_fit_evaluate if experiment['fitness_function'] == 'mat'
                else single_fit_evaluate)

    def record(pop, gen):
        hall_of_fame.update(pop)
        if gen % LOGBOOK_RECORDING_INTERVAL == 0:
            record = mstats.compile(pop)
            logbook.record(gen=gen, **record)

    def process_gen(pop, gen):
        # Selection.
        pop = toolbox.select(pop, len(pop))
        # Cloning.
        offspring = [toolbox.clone(ind) for ind in pop]
        for ind in offspring:
            # Tag offspring with new generation number.
            ind.gen = gen
        # Variation.
        for i in range(len(offspring)):
            toolbox.mutate(offspring[i])
            offspring[i].parent = pop[i]
        # Evaluation.
        evaluate(offspring, gen)
        # Recording.
        record(offspring, gen)
        return offspring

    # Create initial population.
    population = toolbox.population(n=experiment['popsize'])

    log_duration_start = time()
    # Evaluate the initial population.
    evaluate(population, 0)
    # Record stats for initial population.
    record(population, 0)
    # Print first lines of logbook.
    first_lines = str(logbook).split('\n')
    header_lines = ['[Seed {}] '.format(experiment['seed']) + l
                    for l in first_lines[:-1]]
    print('\n'.join(header_lines))
    print_status(first_lines[-1], time() - log_duration_start)

    log_duration_start = time()
    snap_duration_start = time()
    snapshot = 1
    for gen in range(1, experiment['ngen'] + 1):
        # Evolution.
        population = process_gen(population, gen)
        # Reporting.
        if gen % STATUS_PRINTING_INTERVAL == 0:
            # Get time since last report was printed.
            log_duration_end = time()
            print_status(logbook.__str__(startindex=gen),
                         log_duration_end - log_duration_start)
            log_duration_start = time()
        # Snapshotting.
        current_time = time()
        if (current_time - snap_duration_start >= SNAPSHOT_TIME_INTERVAL
                or gen % SNAPSHOT_GENERATION_INTERVAL == 0):
            print('[Seed {}] –\tRecording snapshot {}... '.format(experiment['seed'],
                                                               snapshot), end='')
            dirname = os.path.join(OUTPUT_DIR,
                                   'snapshot-{}-gen-{}'.format(snapshot, gen))
            save_data(dirname, gen, config=experiment,
                      pop=population, logbook=logbook, hof=hall_of_fame,
                      elapsed=(current_time - sim_start))
            print('done.')
            snapshot += 1
            snap_duration_start = time()

    # Finish
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    sim_end = time()
    if PROFILING:
        pr.disable()
        pr.dump_stats(profile_filepath)

    print('\nSimulated {} generations in {}.'.format(
        experiment['ngen'], utils.compress(sim_end - sim_start)))

    # Write final results to disk.
    save_data(OUTPUT_DIR, gen, config=experiment, pop=population,
              logbook=logbook, hof=hall_of_fame, elapsed=(sim_end - sim_start))


from docopt import docopt
if __name__ == '__main__':
    # Get command-line arguments from docopt.
    arguments = docopt(__doc__, version=__version__)
    
    main(arguments)
