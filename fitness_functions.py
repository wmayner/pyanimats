#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fitness_functions.py

"""
Fitness functions for driving animat evolution.
"""

from parameters import TASKS, NATURAL_FITNESS_BASE, SCRAMBLE_WORLD, NUM_TRIALS


# Convert world-strings into integers. Note that in the implementation, the
# world is mirrored; hence the reversal of the string.
TASKS = [(task[0], int(task[1][::-1], 2)) for task in TASKS]
HIT_MULTIPLIERS, BLOCK_PATTERNS = zip(*TASKS)


def natural(ind):
    """Essentially the number of correct trials (each additional correct trial
    is weighted exponentially higher than the last in order to keep selection
    pressure more even).

    This is accomplished by using the ``NATURAL_FITNESS_BASE`` parameter as the
    base and the number of correct trials as the exponent."""
    # Simulate the animat in the world with the given tasks.
    ind.play_game(HIT_MULTIPLIERS, BLOCK_PATTERNS,
                  scramble_world=SCRAMBLE_WORLD)
    assert ind.correct + ind.incorrect == NUM_TRIALS
    # We use an exponential fitness function because the selection pressure
    # lessens as animats get close to perfect performance in the game; thus we
    # need to weight additional improvements more as the animat gets better in
    # order to keep the selection pressure more even.
    return (NATURAL_FITNESS_BASE**ind.animat.correct,)


def mutual_information(ind):
    pass
