#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# constants.py

"""
Container for package-level constants.
"""

import math
import random

import c_animat
from c_animat import MIN_BODY_LENGTH

DEFAULT_RNG = random.Random()
NAT_TO_BIT_CONVERSION_FACTOR = 1 / math.log(2)
START_CODON = [c_animat.START_CODON_NUCLEOTIDE_ONE,
               c_animat.START_CODON_NUCLEOTIDE_TWO]
HIT_TYPE = {c_animat.CORRECT_CATCH: 'CORRECT_CATCH',
            c_animat.WRONG_CATCH: 'WRONG_CATCH',
            c_animat.CORRECT_AVOID: 'CORRECT_AVOID',
            c_animat.WRONG_AVOID: 'WRONG_AVOID'}
# Scale raw fitness values so they're mostly in the range 64–128 before using
# them as an exponent (this depends on which fitness function is used).
FITNESS_TRANSFORMS = {
    'nat':    {'base': 1.02, 'scale': 1,      'add': 0},
    'mi':     {'base': 1.02, 'scale': 64,     'add': 64},
    'mi_wvn': {'base': 1.02, 'scale': 64,     'add': 64},
    'ex':     {'base': 1.02, 'scale': 64 / 4, 'add': 64},
    'ex_wvn': {'base': 1.02, 'scale': 64 / 1, 'add': 64},
    'sp':     {'base': 1.02, 'scale': 64 / 4, 'add': 64},
    'sp_wvn': {'base': 1.02, 'scale': 64 / 1, 'add': 64},
    'bp':     {'base': 1.02, 'scale': 64 / 4, 'add': 64},
    'bp_wvn': {'base': 1.02, 'scale': 63 / 4, 'add': 64},
    'sd_wvn': {'base': 1.02, 'scale': 64 / 8, 'add': 64},
    'mat':    {'base': 1.02, 'scale': 64 / 1, 'add': 64},
}
MINUTES = 60
HOURS = 60 * MINUTES
DAYS = 24 * HOURS
WEEKS = 7 * DAYS
