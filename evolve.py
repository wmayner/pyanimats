#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# evolve.py

"""Implements the genetic algorithm."""

import pickle
import random
from copy import deepcopy
from time import perf_counter as timer

import numpy as np
from deap import base, tools
from munch import Munch

import c_animat
import fitness_functions
import utils
import validate
from animat import Animat
from experiment import Experiment
from phylogeny import Phylogeny


class Evolution:

    """An evolutionary simulation."""

    def __init__(self, experiment, simulation):
        self.version = utils.get_version()
        self.experiment = (experiment if isinstance(experiment, Experiment)
                           else Experiment(experiment))
        self.simulation = Munch(validate.simulation(simulation))
        self.generation = 0
        self.elapsed = 0
        # Get our own RNG.
        self.random = random.Random()
        # Seed the random number generators.
        self.random.seed(self.experiment.rng_seed)
        c_animat.seed(self.experiment.rng_seed)
        # Get their states to pass to the evolution.
        self.python_rng_state = self.random.getstate()
        self.c_rng_state = c_animat.get_rng_state()
        # Initialize the DEAP toolbox.
        self.toolbox = base.Toolbox()
        # Register the various genetic algorithm components to the toolbox.
        self.toolbox.register('animat', Animat, self.experiment,
                              self.experiment.init_genome)
        self.toolbox.register('population', tools.initRepeat, list,
                              self.toolbox.animat)
        # Initialize logbooks and hall of fame.
        self.logbook = tools.Logbook()
        # Create initial population.
        self.population = self.toolbox.population(n=self.experiment.popsize)
        # Create statistics trackers.
        fitness_stats = tools.Statistics(key=lambda animat: animat.fitness.raw)
        fitness_stats.register('max', np.max)
        real_fitness_stats = tools.Statistics(key=lambda animat:
                                              animat.fitness.value)
        real_fitness_stats.register('max', np.max)
        correct_stats = tools.Statistics(key=lambda animat: (animat.correct,
                                                             animat.incorrect))
        correct_stats.register('correct', lambda x: np.max(x, 0)[0])
        correct_stats.register('incorrect', lambda x: np.max(x, 0)[1])
        # Stats objects for alternate matching measures.
        alt_fitness_stats = tools.Statistics(key=lambda animat:
                                             animat.alt_fitness)
        alt_fitness_stats.register('weighted', lambda x: np.max(x, 0)[0])
        alt_fitness_stats.register('unweighted', lambda x: np.max(x, 0)[1])
        # Initialize a MultiStatistics object for convenience that allows for
        # only one call to `compile`.
        if self.experiment.fitness_function == 'mat':
            self.mstats = tools.MultiStatistics(
                correct=correct_stats, fitness=fitness_stats,
                real_fitness=real_fitness_stats, alt_fitness=alt_fitness_stats)
        else:
            self.mstats = tools.MultiStatistics(
                correct=correct_stats, fitness=fitness_stats,
                real_fitness=real_fitness_stats)
        # Initialize evaluate function.
        fitness_function = \
            fitness_functions.__dict__[self.experiment.fitness_function]

        def multi_fit_evaluate(population):
            animats = [a for a in population if a._dirty_fitness]
            for animat in animats:
                fitness = fitness_function(animat)
                animat.fitness.set(fitness[0])
                animat.alt_fitness = fitness[1:]

        def single_fit_evaluate(population):
            animats = [a for a in population if a._dirty_fitness]
            for animat in animats:
                animat.fitness.set(fitness_function(animat))

        self.evaluate = (
            multi_fit_evaluate if self.experiment.fitness_function == 'mat'
            else single_fit_evaluate)

    def update_simulation(self, opts):
        self.simulation.update(opts)
        self.simulation = validate.simulation(self.simulation)

    def __getstate__(self):
        # Copy the instance attributes.
        state = self.__dict__.copy()
        # Remove unpicklable attributes.
        del state['evaluate']
        del state['mstats']
        # Save the population as a Phylogeny to recover lineages later.
        state['population'] = Phylogeny(state['population'])
        return state

    def __setstate__(self, state):
        # Convert the Phylogeny back to a normal list.
        state['population'] = list(state['population'])
        # Initialize from the saved experiment and simulation.
        self.__init__(state['experiment'], state['simulation'])
        # Update with the saved state.
        self.__dict__.update(state)

    def select(self, animats, k):
        """Select *k* animats from a list of animats.

        Uses fitness-proportionate selection.

        Args:
            animats (Iterable): The population of animats to select from.
            k (int): The number of animats to select from the population.

        Returns
            list: The selected animats.
        """
        max_fitness = max(animat.fitness.value for animat in animats)
        chosen = []
        for i in range(k):
            done = False
            while not done:
                candidate = self.random.choice(animats)
                done = self.random.random() <= (candidate.fitness.value /
                                                max_fitness)
            chosen.append(candidate)
        return chosen

    def print_status(self, line, elapsed):
        """Print a status uptdate to the screen."""
        print('[Seed {}] {}{}'.format(self.experiment.rng_seed, line,
                                      utils.compress(elapsed)))

    def record(self, population, gen):
        if gen % self.simulation.logbook_interval == 0:
            record = self.mstats.compile(population)
            self.logbook.record(gen=gen, **record)

    def new_gen(self, population, gen):
        # Update generation number.
        self.generation = gen
        # Selection.
        population = self.select(population, len(population))
        # Cloning.
        # TODO: why does directly cloning the population prevent evolution?!
        offspring = [deepcopy(x) for x in population]
        # Variation.
        for i, animat in enumerate(offspring):
            # Update parent reference.
            animat.parent = population[i]
            # Update generation number.
            animat.gen = gen
            # Mutate.
            animat.mutate()
            # Determine whether fitness needs updating.
            animat._dirty_fitness = not np.array_equal(animat.tpm,
                                                       animat.parent.tpm)
        # Evaluation.
        self.evaluate(offspring)
        # Recording.
        self.record(offspring, gen)
        return offspring

    def run(self, checkpoint_file, ngen=None):
        """Evolve."""
        if ngen is None:
            ngen = self.simulation.ngen
        # Get the range of generations to simulate.
        generations = range(self.generation + 1, ngen + 1)
        # Return immediately if there are no generations to simulate.
        if not generations:
            return 0.0

        # Set the random number generator states.
        self.random.setstate(self.python_rng_state)
        c_animat.set_rng_state(self.c_rng_state)

        # Initial evalutation.
        if self.generation == 0:
            self.evaluate(self.population)
            self.record(self.population, 0)
            # Print first lines of the logbook.
            if 0 < self.simulation.status_interval < float('inf'):
                first_lines = str(self.logbook).split('\n')
                header_lines = [
                    '[Seed {}] {}'.format(self.experiment.rng_seed, l)
                    for l in first_lines[:-1]]
                print('\n' + '\n'.join(header_lines))

        last_status, last_checkpoint = [timer()] * 2

        for gen in generations:
            self.generation = gen
            # Evolution.
            self.population = self.new_gen(self.population, gen)
            # Reporting.
            if gen % self.simulation.status_interval == 0:
                # Get time since last report was printed.
                elapsed_since_last_status = timer() - last_status
                self.print_status(self.logbook.__str__(startindex=-1),
                                  elapsed_since_last_status)
                last_status = timer()
            # Checkpointing.
            elapsed_since_last_checkpoint = timer() - last_checkpoint
            if (elapsed_since_last_checkpoint >=
                    self.simulation.checkpoint_interval):
                print('[Seed {}] Saving checkpoint to `{}`... '.format(
                    self.experiment.rng_seed, checkpoint_file),
                    end='', flush=True)
                self.elapsed += timer() - last_checkpoint
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(self, f)
                last_checkpoint = timer()
                print('done.')

        self.elapsed += timer() - last_checkpoint

        # Save final checkpoint.
        print('[Seed {}] Saving final checkpoint to `{}`... '.format(
            self.experiment.rng_seed, checkpoint_file),
            end='', flush=True)
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(self, f)
        print('done.\n')

        return self.elapsed

    def serializable(self, all_lineages=None):
        if all_lineages is None:
            all_lineages = self.simulation.all_lineages
        # Determine the generational interval.
        gen_interval = max(
            self.simulation.ngen // self.simulation.output_samples, 1)
        # Get the lineage(s).
        if not all_lineages:
            fittest = max(self.population, key=lambda a: a.fitness.value)
            lineage = fittest.serializable_lineage(interval=gen_interval,
                                                   experiment=False)
        else:
            lineage = [a.serializable_lineage(interval=gen_interval,
                                              experiment=False)
                       for a in self.population]
        # Set up the serializable object.
        return {
            'experiment': self.experiment,
            'simulation': self.simulation,
            'lineage': lineage,
            'logbook': {
                'gen': self.logbook.select('gen'),
                'fitness': self.logbook.chapters['fitness'].select('max'),
                'correct': self.logbook.chapters['correct'].select('correct'),
                'incorrect': (self.logbook.chapters['correct']
                              .select('incorrect')),
            },
            'elapsed': round(self.elapsed, 2),
            'version': utils.get_version(),
        }
