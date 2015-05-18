#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# evolve.py

import os
from pprint import pprint
import pickle
import random
import numpy
from time import time
import cProfile

import parameters
from parameters import (NGEN, POPSIZE, SEED, TASKS, FITNESS_BASE,
                        SCRAMBLE_WORLD)

random.seed(SEED)
from individual import Individual

from deap import creator, base, tools
toolbox = base.Toolbox()


PROFILING = False
# Status will be printed at this interval.
LOG_FREQ = 100


# Convert world-strings into integers. Note that in the implementation, the
# world is mirrored; hence the reversal of the string.
print('SEED:', SEED)
print('TASKS:')
pprint(TASKS)
TASKS = [(task[0], int(task[1][::-1], 2)) for task in TASKS]


def evaluate(ind):
    # Simulate the animat in the world with the given tasks.
    hit_multipliers, patterns = zip(*TASKS)
    ind.play_game(hit_multipliers, patterns, scramble_world=SCRAMBLE_WORLD)
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

hof = tools.HallOfFame(maxsize=100)

if __name__ == '__main__':

    population = toolbox.population(n=POPSIZE)

    if PROFILING:
        pr = cProfile.Profile()

    # Evaluate the initial population.
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    hof.update(population)
    record = fitness_stats.compile(population)
    logbook1.record(gen=0, **record)
    record = correct_stats.compile(population)
    logbook2.record(gen=0, **record)

    if PROFILING:
        pr.enable()
    start = time()

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
        record = fitness_stats.compile(population)
        logbook1.record(gen=gen, **record)
        record = correct_stats.compile(population)
        logbook2.record(gen=gen, **record)
        return offspring

    # Evolution.
    for gen in range(1, NGEN + 1):
        population = process_gen(population)
        if gen % LOG_FREQ == 0:
            print('[Generation {}] Max Correct / Max Incorrect: {} Avg. '
                  'Fitness: {}'.format(str(gen).rjust(len(str(NGEN))),
                                       logbook2[-1]['correct/incorrect'],
                                       logbook1[-1]['max']))

    end = time()
    if PROFILING:
        pr.disable()
        pr.dump_stats('profiling/profile.pstats')

    print("Simulated {} generations in {} seconds.".format(
        NGEN, round(end - start, 2)))

    # Save data.
    data = {
        'params': parameters.param_dict,
        'lineages': [tuple(ind.lineage()) for ind in population],
        'logbooks': {
            'fitness': logbook1,
            'correct': logbook2,
        },
        'hof': [ind.animat for ind in hof],
        'metadata': {
            'elapsed': end - start,
            'version': '0.0.0'
        }
    }
    RESULTS_DIR = 'results/current/seed-{}'.format(SEED)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    for key in data:
        with open(os.path.join(RESULTS_DIR,
                               'seed-{}_{}.pkl'.format(SEED, key)),
                  'wb') as f:
            pickle.dump(data[key], f)
