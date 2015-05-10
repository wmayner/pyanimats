#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# evolve.py

from pprint import pprint
import pickle
import random
import numpy
from time import time
import cProfile

import parameters
from parameters import (
    NGEN, POPSIZE, SEED, TASKS, INIT_GENOME, MUTATION_PROB, DUPLICATION_PROB,
    DELETION_PROB, MAX_GENOME_LENGTH, MIN_GENOME_LENGTH, MIN_DUP_DEL_WIDTH,
    MAX_DUP_DEL_WIDTH, FITNESS_BASE
)

random.seed(SEED)
from individual import Individual

from deap import creator, base, tools
toolbox = base.Toolbox()


PROFILING = True
# Status will be printed at this interval.
LOG_FREQ = 1000


# Convert world-strings into integers. Note that in the implementation, the
# world is mirrored; hence the reversal of the string.
print('TASKS:')
pprint(TASKS)
TASKS = [(task[0], int(task[1][::-1], 2)) for task in TASKS]


def evaluate(ind):
    ind.animat.update_phenotype()
    # Simulate the animat in the world.
    hit_multipliers, patterns = zip(*TASKS)
    ind.animat.play_game(hit_multipliers, patterns)
    # We use an exponential fitness function because the selection pressure
    # lessens as animats get close to perfect performance in the game; thus we
    # need to weight additional improvements more as the animat gets better in
    # order to keep the selection pressure more even.
    return (FITNESS_BASE**ind.animat.correct,)
toolbox.register('evaluate', evaluate)


def mutate(ind, mut_prob, dup_prob, del_prob, min_genome_length,
           max_genome_length):
    ind.animat.mutate(mut_prob, dup_prob, del_prob, min_genome_length,
                      max_genome_length)
    return (ind,)
toolbox.register('mutate', mutate,
                 mut_prob=MUTATION_PROB,
                 dup_prob=DUPLICATION_PROB,
                 del_prob=DELETION_PROB,
                 min_genome_length=MIN_GENOME_LENGTH,
                 max_genome_length=MAX_GENOME_LENGTH)


creator.create('Fitness', base.Fitness, weights=(1.0,))
creator.create('Individual', Individual, fitness=creator.Fitness)
toolbox.register('individual', creator.Individual)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('select', tools.selRoulette)

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

population = toolbox.population(n=POPSIZE)


if __name__ == '__main__':

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
            print('[Generation {}] Correct/Incorrect: {} Fitness: {}'.format(
                str(gen).rjust(len(str(NGEN))),
                logbook2[-1]['correct/incorrect'],
                logbook1[-1]['max'])
            )

    end = time()
    if PROFILING:
        pr.disable()
        pr.dump_stats('profiling/profile.pstats')

    print("Simulated {} generations in {} seconds.".format(
        NGEN, round(end - start, 2)))

    # Save data
    data = {
        'parameters': parameters.param_dict,
        'lineages': [tuple(ind.lineage()) for ind in population],
        'logbooks': {
            'fitness': logbook1,
            'correct': logbook2,
        },
        'hof': [ind.animat for ind in hof],
        'elapsed': end - start,
        'version': '0.0.0'
    }
    with open('results/seed_{}.pkl'.format(SEED), 'wb') as f:
        pickle.dump(data, f)
