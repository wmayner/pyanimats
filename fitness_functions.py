#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fitness_functions.py

"""
Fitness functions for driving animat evolution.
"""

from parameters import params


# A registry of available fitness functions
functions = {}


def register(f):
    """Register a fitness function to the directory."""
    functions[f.__name__] = f.__doc__
    return f


def print_functions():
    """Display a list of available fitness functions with their
    descriptions."""
    print('')
    for name, doc in functions.items():
        print(name + '\n    ' + doc)
        print('')


@register
def natural(ind):
    """Animats are evaluated based on the number of trials they successfully
    complete. Each additional correct trial is weighted exponentially higher
    than the last in order to keep selection pressure more even.

    This is accomplished by using the ``NATURAL_FITNESS_BASE`` parameter as the
    base and the number of correct trials as the exponent."""
    ind.play_game()
    return (params.NATURAL_FITNESS_BASE**ind.animat.correct,)


@register
def mutual_information(ind):
    # TODO implement mutual information
    """MUTUAL INFORMATION DESCRIPTION"""
    pass
