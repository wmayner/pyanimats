import random
import numpy
from time import time
# import networkx as nx

import cProfile

from deap import creator, base, tools, algorithms
from animats import Animat, World

INIT_GENOME_SIZE = 5000

creator.create('FitnessCatch', base.Fitness, weights=(1.0,))
creator.create('Individual', bytearray, fitness=creator.FitnessCatch)

toolbox = base.Toolbox()

toolbox.register('random_byte', random.randint, 0, 255)
toolbox.register('individual', tools.initRepeat, creator.Individual,
                 toolbox.random_byte, INIT_GENOME_SIZE)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)


FITNESS_BASE = 1.02

TASKS = (
    (1, '1110000000000000'),
)

def evaluate(genome):
    animat = Animat(genome)
    # Simulate the animat in the world.
    world = World(animat, TASKS)
    world.run()

    # We use an exponential fitness function because the selection pressure
    # lessens as animats get close to perfect performance in the game; thus we
    # need to weight additional improvements more as the animat gets better in
    # order to keep the selection pressure more even.
    return (FITNESS_BASE**animat.correct,)


toolbox.register('evaluate', evaluate)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
toolbox.register('select', tools.selTournament, tournsize=3)

history = tools.History()
toolbox.decorate('mate', history.decorator)
toolbox.decorate('mutate', history.decorator)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('avg', numpy.mean)
stats.register('std', numpy.std)
stats.register('min', numpy.min)
stats.register('max', numpy.max)

logbook = tools.Logbook()

POPSIZE = 100
population = toolbox.population(n=POPSIZE)

NGEN = 1000

pr = cProfile.Profile()

# Evaluate the individuals with an invalid fitness
invalid_ind = [ind for ind in population if not ind.fitness.valid]
fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
for ind, fit in zip(invalid_ind, fitnesses):
    ind.fitness.values = fit
record = stats.compile(population)
logbook.record(gen=0, popsize=POPSIZE, **record)

start = time()
pr.enable()
for gen in range(1, NGEN+1):
    # Selection
    offspring = toolbox.select(population, len(population))
    # Variation (crossover and mutation)
    offspring = algorithms.varAnd(offspring, toolbox, cxpb=0.5, mutpb=0.1)

    # Reevaluate the individuals with an invalid fitness, i.e. those that were
    # changed
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    population = offspring

    record = stats.compile(population)
    logbook.record(gen=gen, popsize=POPSIZE, **record)
pr.disable()
end = time()
pr.dump_stats('profiling/profile.pstats')

print("Simulated {} generations in {} seconds.".format(NGEN,
                                                       round(end - start, 2)))

# hof = tools.HallOfFame(maxsize=10)
# best = hof[0]
# graph = nx.DiGraph(history.getGenealogy(best, max_depth=5))
