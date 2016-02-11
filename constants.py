#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# constants.py

"""
Container for package-level constants.
"""

import math

import animat
from animat import MIN_BODY_LENGTH

NAT_TO_BIT_CONVERSION_FACTOR = 1 / math.log(2)
START_CODON = [animat.START_CODON_NUCLEOTIDE_ONE,
               animat.START_CODON_NUCLEOTIDE_TWO]
HIT_TYPE = {animat.CORRECT_CATCH: 'CORRECT_CATCH',
            animat.WRONG_CATCH: 'WRONG_CATCH',
            animat.CORRECT_AVOID: 'CORRECT_AVOID',
            animat.WRONG_AVOID: 'WRONG_AVOID'}
# Scale raw fitness values so they're mostly in the range 64–128 before using
# them as an exponent (this depends on which fitness function is used).
FITNESS_TRANSFORMS = {
    'nat':       {'scale': 1,      'add': 0},
    'mi':        {'scale': 64,     'add': 64},
    'mi_wvn':    {'scale': 64,     'add': 64},
    'ex':        {'scale': 64 / 4, 'add': 64},
    'ex_wvn':    {'scale': 64 / 1, 'add': 64},
    'sp':        {'scale': 64 / 4, 'add': 64},
    'sp_wvn':    {'scale': 64 / 1, 'add': 64},
    'bp':        {'scale': 64 / 4, 'add': 64},
    'bp_wvn':    {'scale': 63 / 4, 'add': 64},
    'state_wvn': {'scale': 64 / 8, 'add': 64},
    'mat':       {'scale': 64 / 1, 'add': 64},
}
