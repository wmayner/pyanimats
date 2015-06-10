#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# evolve.py

"""
PyAnimats
---------
Evolve animats.

Usage:
    evolve.py <output_dir> <tasks.yml> <params.yml> [options]
    evolve.py <output_dir> <tasks.yml> [options]
    evolve.py --list-fitness-funcs
    evolve.py -h | --help
    evolve.py -v | --version

Options:
    -h, --help                Show this
    -v, --version             Show version
        --list-fitness-funcs  List available fitness functions
    -n, --num-gen=NGEN        Number of generations to simulate [default: 10]
    -s, --seed=SEED           Random number generator seed [default: 0]
    -f, --fitness=FUNC        Fitness function [default: natural]
    -l, --log-freq=FREQ       Status printing interval [default: 1]
    -p, --pop-size=SIZE       Population size [default: 100]
    -m, --mut-prob=PROB       Nucleotide mutation probability [default: 0.005]
        --scramble            Randomly rearrange the world in each trial
        --dup-prob=PROB       Duplication probability [default: 0.05]
        --del-prob=PROB       Deletion probability [default: 0.02]
        --max-length=LENGTH   Maximum genome length [default: 10000]
        --min-length=LENGTH   Minimum genome length [default: 1000]
        --min-dup-del=LENGTH  Minimum length of duplicated/deleted genome part
                                [default: 15]
        --nat-fit-base=FLOAT  Base used in the natural fitness function (see
                                --list-fitness-funcs) [default: 1.02]
        --profile=PATH        Profile performance and store results at PATH
                                [default: profiling/profile.pstats]

Note: command-line arguments override parameters in the <params.yml> file.
"""

__version__ = '0.0.4'

import os
import pickle
import random
import numpy
from time import time
import cProfile

from parameters import params
import fitness_functions
from individual import Individual
from deap import creator, base, tools


PROFILING = False


def select(individuals, k):
    """Select *k* individuals from the given list of individuals using the
    variant of roulette-wheel selection used in the old C++ code.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.

    This function uses the :func:`~random.random` function from the built-in
    :mod:`random` module."""
    max_fitness = max([ind.fitness.values[0] for ind in individuals])
    chosen = []
    for i in range(k):
        done = False
        while not done:
            candidate = random.choice(individuals)
            done = random.random() <= (candidate.fitness.values[0] /
                                       max_fitness)
        chosen.append(candidate)
    return chosen


def mutate(ind):
    ind.mutate()
    return (ind,)
mutate.__doc__ = Individual.mutate.__doc__


def main(arguments):

    if arguments['--list-fitness-funcs']:
        fitness_functions.print_functions()
        return

    # Ensure output directory exists.
    output_dir = arguments['<output_dir>']
    del arguments['<output_dir>']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure profile directory exists and set profile flag.
    profile_filepath = arguments['--profile']
    if profile_filepath:
        PROFILING = True
        profile_dir = os.path.dirname(profile_filepath)
        if not os.path.exists(profile_dir):
            os.makedirs(profile_dir)
    del arguments['--profile']

    # Status will be printed at this interval.
    LOG_FREQ = int(arguments['--log-freq'])
    del arguments['--log-freq']

    # Load parameters.
    if arguments['<params.yml>']:
        params.load_from_file(arguments['<params.yml>'])
        del arguments['<params.yml>']
    params.load_from_args(arguments)

    print('Parameters:')
    print(params)

    # Setup
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    toolbox = base.Toolbox()

    # Register the various genetic algorithm components to the toolbox.
    creator.create('Fitness', base.Fitness, weights=(1.0,))
    creator.create('Individual', Individual, fitness=creator.Fitness)
    toolbox.register('individual', creator.Individual, params.INIT_GENOME)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate',
                     fitness_functions.__dict__[params.FITNESS_FUNCTION])
    toolbox.register('select', select)
    toolbox.register('mutate', mutate)

    # Create statistics trackers.
    fitness_stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    fitness_stats.register('avg', numpy.mean)
    fitness_stats.register('std', numpy.std)
    fitness_stats.register('min', numpy.min)
    fitness_stats.register('max', numpy.max)

    correct_stats = tools.Statistics(key=lambda ind: (ind.animat.correct,
                                                      ind.animat.incorrect))
    correct_stats.register('correct', lambda x: numpy.max(x, 0)[0])
    correct_stats.register('incorrect', lambda x: numpy.max(x, 0)[1])

    # Initialize logbooks and hall of fame.
    logbook1, logbook2 = tools.Logbook(), tools.Logbook()
    hof = tools.HallOfFame(maxsize=params.POPSIZE)

    print('\nSimulating {} generations...'.format(params.NGEN))

    if PROFILING:
        pr = cProfile.Profile()
        pr.enable()
    start = time()

    # Simulation
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    population = toolbox.population(n=params.POPSIZE)

    # Evaluate the initial population.
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fitness in zip(population, fitnesses):
        ind.fitness.values = fitness
    # Record stats for initial population.
    hof.update(population)
    record = fitness_stats.compile(population)
    logbook1.record(gen=0, **record)
    record = correct_stats.compile(population)
    logbook2.record(gen=0, **record)

    def process_gen(population):
        # Selection.
        population = toolbox.select(population, len(population))
        # Cloning.
        offspring = [toolbox.clone(ind) for ind in population]
        for ind in offspring:
            # Tag offspring with new generation number.
            ind.gen = gen
        # Variation.
        for i in range(len(offspring)):
            toolbox.mutate(offspring[i])
            offspring[i].parent = population[i]
        # Evaluation.
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
        # Recording.
        hof.update(offspring)
        record = fitness_stats.compile(offspring)
        logbook1.record(gen=gen, **record)
        record = correct_stats.compile(offspring)
        logbook2.record(gen=gen, **record)
        return offspring

    # Evolution.
    for gen in range(1, params.NGEN + 1):
        population = process_gen(population)
        if gen % LOG_FREQ == 0:
            print('[Generation] {}  [Max Correct] {}  [Max Incorrect] {}  '
                  '[Avg. Fitness]  {}'.format(
                      str(gen).rjust(len(str(params.NGEN))),
                      str(logbook2[-1]['correct']).rjust(3),
                      str(logbook2[-1]['incorrect']).rjust(3),
                      logbook1[-1]['max']))

    # Finish
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    end = time()
    if PROFILING:
        pr.disable()
        pr.dump_stats(profile_filepath)

    print("Simulated {} generations in {} seconds.".format(
        params.NGEN, round(end - start, 2)))

    # Save data.
    data = {
        'params': params,
        'lineages': [tuple(ind.lineage()) for ind in population],
        'logbooks': {
            'fitness': logbook1,
            'correct': logbook2,
        },
        'hof': [ind.animat for ind in hof],
        'metadata': {
            'elapsed': end - start,
            'version': __version__
        }
    }
    for key in data:
        with open(os.path.join(output_dir, '{}.pkl'.format(key)), 'wb') as f:
            pickle.dump(data[key], f)


from docopt import docopt
if __name__ == '__main__':
    # Get command-line arguments from docopt.
    arguments = docopt(__doc__, version='PyAnimats v{}'.format(__version__))
    main(arguments)
