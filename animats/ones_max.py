'''
A simple Individual who represents the basics of what
an Individual should be. This is a sort of "template"
for future Individuals.

It's genome is represented by a binary string, eg:

010011011101

and it's fitness function maximizes the number of
1s, eg:

111110111011

Note, the way I've implemented this, it can't really get a perfect score ever
because good genes mutate as frequently as bad.

'''


import random
import functools
from copy import deepcopy


class ExponentialFitness:

    """
    Represents the two notions of fitness: the value that is used in
    selection (the ``exponential`` attribute and the one returned by
    ``value``), and the value we're interested in (the ``raw`` attribute and
    the one used when setting ``value``).

    We use an exponential fitness function to ensure that selection pressure is
    more even as the animats improve. When the actual fitness function is not
    exponential, this class handles transforming it to be so.
    """

    def __init__(self, experiment, value=0.0):
        self.experiment = experiment
        self.value = value

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return 'ExponentialFitness({})'.format(self.raw)

    def __str__(self):
        return '(raw={}, exponential={})'.format(self.raw, self.exponential)

    @functools.total_ordering
    def __lt__(self, other):
        return self.value < other.value

    @property
    def value(self):
        return self.exponential

    @value.setter
    def value(self, v):
        self.raw = v + 1  # +1 avoids `div by 0` errors
        self.exponential = v + 1


class Animat:
    """
    ToDo: Individuals shouldn't need some representation of
    Animats, if they don't need to. This is just a place-
    holder class to allow this Individual class to operate
    like the original HMU class. Should be refactored
    """
    correct = 0
    incorrect = 0
    genome = []


class Individual:
    """
    Represents an individual in the evolution

    Args:

    Keyword Args:

    Attributes:
    """
    MUTATION_PROB = .05

    def __init__(self, genome=[0]*100, parent=None, gen=0):
        self.fitness = ExponentialFitness(self.experiment)
        self.animat = Animat()

        # It's the animat's genome that makes it to
        # `lineages.pkl` at the end, so save it on the animat
        self.animat.genome = genome

        self.parent = parent
        self.gen = gen
        self.genome = genome

    @classmethod
    def initializeClass(cls, experiment):
        cls.experiment = experiment

    def mutate(self):
        """ probabilistically mutate at each locus """
        for i in range(len(self.genome)):
            # flip the bit if that loci should mutate
            self.genome[i] = (self.genome[i] + 1) % 2\
                if (random.random() < self.MUTATION_PROB) \
                else self.genome[i]

    def play_game(self):
        """ """
        self.animat.correct = sum(self.genome)
        self.animat.incorrect = len(self.genome) - sum(self.genome)

    def lineage(self):
        """Return a generator for the lineage of this individual."""
        yield self.animat
        ancestor = self.parent
        while ancestor is not None:
            yield ancestor.animat
            ancestor = ancestor.parent

    def __deepcopy__(self, memo):
        # Don't copy the underlying animat, parent, or PyPhi network.
        copy = Individual(genome=self.genome,
                          parent=self.parent)
        for key, val in self.__dict__.items():
            if key not in ('animat', 'parent', '_network', '_dirty_network'):
                copy.__dict__[key] = deepcopy(val, memo)
        return copy
