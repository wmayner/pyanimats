#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# evolve.py

"""Implements the genetic algorithm."""

import datetime
import gzip
import pickle
import random
from copy import deepcopy
from time import perf_counter as timer

import dateutil.parser
import numpy as np
from deap import base, tools
from munch import Munch

from . import animat, c_animat, fitness_functions, utils, validate
from .fitness_transforms import ExponentialMultiFitness
from .animat import Animat
from .experiment import Experiment
from .phylogeny import Phylogeny
from .utils import rounder


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
        self.logbook.header = ['gen', 'fitness', 'game']
        self.logbook.chapters['fitness'].header = ['raw', 'exp']
        # Create initial population.
        self.population = self.toolbox.population(n=self.experiment.popsize)
        # If we're using an expensive fitness function, then check if the TPM
        # has changed before re-evaluating fitness (with cheap functions, like
        # `nat`, it's actually more expensive to generate the TPM and check it)
        self.CHECK_FOR_TPM_CHANGE = any(
            f not in fitness_functions.CHEAP
            for f in self.experiment.fitness_function)
        # Transform the fitness function.
        self.fitness_function = ExponentialMultiFitness(
            self.experiment.fitness_function,
            self.experiment.fitness_transform,
            self.experiment.fitness_ranges)
        # Create statistics trackers.
        fitness_stats = tools.Statistics(
            key=lambda a: (a.fitness, a.raw_fitness))
        fitness_stats.register('raw', lambda x: rounder(max(x)[1]))
        fitness_stats.register('exp', lambda x: rounder(max(x)[0]))
        game_stats = tools.Statistics(key=lambda a: (a.fitness, a.correct))
        game_stats.register('fittest', lambda x: max(x)[1])
        game_stats.register('weakest', lambda x: min(x)[1])
        # Initialize a MultiStatistics object for convenience that allows for
        # only one call to `compile`.
        self.mstats = tools.MultiStatistics(fitness=fitness_stats,
                                            game=game_stats)

    def evaluate(self, population):
        animats = [a for a in population if a._dirty_fitness]
        for a in animats:
            a.fitness, a.raw_fitness = self.fitness_function(a)

    def update_simulation(self, opts):
        self.simulation.update(opts)
        # TODO don't change user-set stuff
        self.simulation = validate.simulation(self.simulation)

    def __getstate__(self):
        # Copy the instance attributes.
        state = self.__dict__.copy()
        # Remove unpicklable attributes.
        del state['mstats']
        del state['fitness_function']
        # Save the population as a Phylogeny to recover lineages later.
        state['population'] = Phylogeny(state['population'],
                                        step=self.simulation.sample_interval)
        return state

    def __setstate__(self, state):
        # Convert the Phylogeny back to a normal list.
        state['population'] = list(state['population'])
        # Re-initialize references to our RNG on the animats.
        for a in state['population']:
            a.random = state['random']
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
        max_fitness = max(animat.fitness for animat in animats)
        chosen = []
        for i in range(k):
            done = False
            while not done:
                candidate = self.random.choice(animats)
                done = self.random.random() <= (candidate.fitness /
                                                max_fitness)
            chosen.append(candidate)
        return chosen

    def print_status(self, line, elapsed):
        """Print a status uptdate to the screen."""
        print('[Seed {}]\t{}{}'.format(self.experiment.rng_seed, line,
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
        for i, a in enumerate(offspring):
            # Use our RNG.
            a.random = self.random
            # Update parent reference.
            a.parent = population[i]
            # Update generation number.
            a.gen = gen
            # Mutate.
            a.mutate()
            # Check whether fitness needs updating (if desired and CM is
            # nontrivial).
            if self.CHECK_FOR_TPM_CHANGE and not a.cm.sum() == 0:
                a._dirty_fitness = not np.array_equal(a.tpm, a.parent.tpm)
            else:
                a._dirty_fitness = True
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

        if self.generation == 0:
            # Inject start codons.
            if self.experiment.init_start_codons:
                for a in self.population:
                    a.inject_start_codons(self.experiment.init_start_codons)

            self.evaluate(self.population)
            self.record(self.population, self.generation)

            # Print first lines of the logbook.
            if 0 < self.simulation.status_interval < float('inf'):
                first_lines = str(self.logbook).split('\n')
                header_lines = [
                    '[Seed {}]\t{}'.format(self.experiment.rng_seed, l)
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
                with gzip.open(checkpoint_file, 'wb') as f:
                    pickle.dump(self, f)
                last_checkpoint = timer()
                print('done.')

        self.elapsed += timer() - last_checkpoint

        # Save final checkpoint.
        print('[Seed {}]\tSaving final checkpoint to `{}`... '.format(
            self.experiment.rng_seed, checkpoint_file),
            end='', flush=True)
        with gzip.open(checkpoint_file, 'wb') as f:
            pickle.dump(self, f)
        print('done.\n')

        return self.elapsed

    def serializable(self, all_lineages=None):
        if all_lineages is None:
            all_lineages = self.simulation.all_lineages
        # Get the lineage(s).
        if not all_lineages:
            fittest = max(self.population, key=lambda a: a.fitness)
            lineage = fittest.lineage(step=self.simulation.sample_interval)
        else:
            lineage = [a.lineage(step=self.simulation.sample_interval)
                       for a in self.population]
        # Set up the serializable object.
        return {
            'experiment': self.experiment,
            'simulation': self.simulation,
            'lineage': lineage,
            'logbook': {
                'fitness': self.logbook.chapters['fitness'].select('exp'),
                'raw_fitness': self.logbook.chapters['fitness'].select('raw'),
                'game': self.logbook.chapters['game'].select('fittest'),
            },
            'elapsed': round(self.elapsed, 2),
            'version': utils.get_version(),
            'time': datetime.datetime.now().isoformat(),
        }


def from_json(d):
    """Initialize an Evolution object from a JSON dictionary."""
    d = Munch(d)
    d.simulation = Munch(d.simulation)
    d.experiment = Experiment(d.experiment)
    d.time = dateutil.parser.parse(d.time)
    # Restore population
    lineage = list(
        map(lambda a: animat.from_json(a, experiment=d['experiment']),
            d['lineage']))
    for i in range(len(lineage) - 1):
        lineage[i].parent = lineage[i + 1]
    d.lineage = lineage
    return d
