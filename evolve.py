import numpy
from time import time
from copy import copy, deepcopy
import cProfile


PROFILING = True

# Simulation parameters
NGEN = 60000
POPSIZE = 100
SEED = 0
# Evolution parameters
INIT_GENOME = [127] * 5000
MUTATION_PROB = 0.002
DUPLICATION_PROB = 0.05
DELETION_PROB = 0.02
MAX_GENOME_LENGTH = 10000
MIN_GENOME_LENGTH = 1000
MIN_DUP_DEL_WIDTH = 15
MAX_DUP_DEL_WIDTH = 511


from deap import creator, base, tools
toolbox = base.Toolbox()

import animat
animat.seed(SEED)
from animat import Animat


class Individual:

    def __init__(self, genome=INIT_GENOME, parent=None):
        self.animat = Animat(genome)
        self.parent = parent

    def __deepcopy__(self, memo):
        # Don't copy the animat or the parent.
        copy = Individual(genome=self.animat.genome, parent=self.parent)
        for key, val in self.__dict__.items():
            if not key in ('animat', 'parent'):
                copy.__dict__[key] = deepcopy(val, memo)
        return copy


FITNESS_BASE = 1.02
TASKS = (
    ( 1, '1000000000000000'),
    (-1, '1110000000000000'),
    ( 1, '1000000000000000'),
    (-1, '1110000000000000'),
)
# Convert world-strings into integers. Note that in the implementation, the
# world is mirrored; hence the reversal of the string.
TASKS = [(task[0], int(task[1][::-1], 2)) for task in TASKS]
print(TASKS)
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

correct_stats = tools.Statistics(key=lambda ind: (ind.animat.correct, ind.animat.incorrect))
correct_stats.register('correct/incorrect', lambda x: numpy.max(x, 0))

logbook1 = tools.Logbook()
logbook2 = tools.Logbook()

hof = tools.HallOfFame(maxsize=10)

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
        print('[Generation {}] Correct/Incorrect: {} Fitness: {}'.format(gen,
                                                logbook2[-1]['correct/incorrect'],
                                                logbook1[-1]['max']))
        return offspring

    # Evolution.
    for gen in range(1, NGEN):
        population = process_gen(population)

    end = time()
    if PROFILING:
        pr.disable()
        pr.dump_stats('profiling/profile.pstats')

    print("Simulated {} generations in {} seconds.".format(
        NGEN, round(end - start, 2)))
