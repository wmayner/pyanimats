#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# evolve.py

import os
import pickle
import random
import numpy
from time import time
import cProfile

import parameters
from parameters import (NGEN, POPSIZE, SEED, TASKS, FITNESS_BASE,
                        SCRAMBLE_WORLD, NUM_TRIALS)

random.seed(SEED)
from individual import Individual

from deap import creator, base, tools
toolbox = base.Toolbox()


__version__ = '0.0.1'


RESULTS_DIR = 'results/current/seed-{}'.format(SEED)
PROFILING = False
# Status will be printed at this interval.
LOG_FREQ = 1000


# Convert world-strings into integers. Note that in the implementation, the
# world is mirrored; hence the reversal of the string.
TASKS = [(task[0], int(task[1][::-1], 2)) for task in TASKS]


def evaluate(ind):
    # Simulate the animat in the world with the given tasks.
    hit_multipliers, patterns = zip(*TASKS)
    ind.play_game(hit_multipliers, patterns, scramble_world=SCRAMBLE_WORLD)
    assert ind.correct + ind.incorrect == NUM_TRIALS
    # We use an exponential fitness function because the selection pressure
    # lessens as animats get close to perfect performance in the game; thus we
    # need to weight additional improvements more as the animat gets better in
    # order to keep the selection pressure more even.
    return (FITNESS_BASE**ind.animat.correct,)
toolbox.register('evaluate', evaluate)


def mutate(ind):
    ind.mutate()
    return (ind,)
toolbox.register('mutate', mutate)


def select(individuals, k):
    """Select *k* individuals from the input *individuals* using the variant of
    roulette-wheel selection used in the C++ code.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
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


creator.create('Fitness', base.Fitness, weights=(1.0,))
creator.create('Individual', Individual, fitness=creator.Fitness)
toolbox.register('individual', creator.Individual)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('select', select)

fitness_stats = tools.Statistics(key=lambda ind: ind.fitness.values)
fitness_stats.register('avg', numpy.mean)
fitness_stats.register('std', numpy.std)
fitness_stats.register('min', numpy.min)
fitness_stats.register('max', numpy.max)

correct_stats = tools.Statistics(key=lambda ind: (ind.animat.correct,
                                                  ind.animat.incorrect))
correct_stats.register('correct/incorrect', lambda x: numpy.max(x, 0))

logbook1 = tools.Logbook()
logbook2 = tools.Logbook()

hof = tools.HallOfFame(maxsize=POPSIZE)

if __name__ == '__main__':
    parameters.print_parameters()

    if PROFILING:
        pr = cProfile.Profile()
        pr.enable()
    start = time()

    population = toolbox.population(n=POPSIZE)

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
    for gen in range(1, NGEN + 1):
        population = process_gen(population)
        if gen % LOG_FREQ == 0:
            print('[Generation] {}  [Max Correct] {}  [Max Incorrect] {}  '
                  '[Avg. Fitness] {}'.format(
                      str(gen).rjust(len(str(NGEN))),
                      str(logbook2[-1]['correct/incorrect'][0]).rjust(3),
                      str(logbook2[-1]['correct/incorrect'][1]).rjust(3),
                      logbook1[-1]['max']))

    end = time()
    if PROFILING:
        pr.disable()
        pr.dump_stats('profiling/profile.pstats')

    print("Simulated {} generations in {} seconds.".format(
        NGEN, round(end - start, 2)))

    # Save data.
    data = {
        'params': parameters.params,
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
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    for key in data:
        with open(os.path.join(RESULTS_DIR, '{}.pkl'.format(key)), 'wb') as f:
            pickle.dump(data[key], f)

    parameters.print_parameters()
