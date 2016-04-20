#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# constants.py

"""
Container for package-level constants.
"""

import math
import random

from . import c_animat

MIN_BODY_LENGTH = c_animat.MIN_BODY_LENGTH
DEFAULT_RNG = random.Random()
NAT_TO_BIT_CONVERSION_FACTOR = 1 / math.log(2)
START_CODON = [c_animat.START_CODON_NUCLEOTIDE_ONE,
               c_animat.START_CODON_NUCLEOTIDE_TWO]
HIT_TYPE = {c_animat.CORRECT_CATCH: 'CORRECT_CATCH',
            c_animat.WRONG_CATCH: 'WRONG_CATCH',
            c_animat.CORRECT_AVOID: 'CORRECT_AVOID',
            c_animat.WRONG_AVOID: 'WRONG_AVOID'}

MINUTES = 60
HOURS = 60 * MINUTES
DAYS = 24 * HOURS
WEEKS = 7 * DAYS

# Scale raw fitness values so they're mostly in the desired range before using
# them as an exponent (this depends on which fitness function is used).
DESIRED_RANGE = {'min': 64, 'max': 128}


def normalization_params(minimum, maximum):
    """Return the scale and constant factor"""
    add = DESIRED_RANGE['min'] - minimum
    scale = (DESIRED_RANGE['max'] - add) / maximum
    return {'add': add, 'scale': scale}


DEFAULT_BASE = 1.02
RANGES = {
    'nat':        (0, 128),
    'mi':         (0, 2),
    'mi_nat':     (0, 1),
    'mi_wvn':     (0, 1),
    'mi_wvn_nat': (0, 1),
    'ex':         (0, 4),
    'ex_nat':     (0, 1),
    'ex_wvn':     (0, 1),
    'ex_wvn_nat': (0, 1),
    'sp':         (0, 4),
    'sp_nat':     (0, 1),
    'sp_wvn':     (0, 1),
    'sp_wvn_nat': (0, 1),
    'bp':         (0, 4),
    'bp_nat':     (0, 1),
    'bp_wvn':     (0, 4),
    'bp_wvn_nat': (0, 1),
    'sd_wvn':     (0, 8),
    'sd_wvn_nat': (0, 1),
    'mat':        (0, 1),
    'mat_nat':    (0, 1),
}
FITNESS_TRANSFORMS = {k: normalization_params(r[0], r[1])
                      for k, r in RANGES.items()}
for k, params in FITNESS_TRANSFORMS.items():
    params.update(base=DEFAULT_BASE)
