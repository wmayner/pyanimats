#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# constants.py

"""
Container for static constants and values derived from parameters.

Populated by :mod:`configure`.
"""

import math

import animat
from animat import MIN_BODY_LENGTH

NAT_TO_BIT_CONVERSION_FACTOR = 1 / math.log(2)
START_CODON = [animat.START_CODON_NUCLEOTIDE_ONE,
               animat.START_CODON_NUCLEOTIDE_TWO]
DEFAULT_INIT_GENOME = [127] * 5000
INIT_GENOME = [127] * 5000
HIT_TYPE = {animat.CORRECT_CATCH: 'CORRECT_CATCH',
            animat.WRONG_CATCH: 'WRONG_CATCH',
            animat.CORRECT_AVOID: 'CORRECT_AVOID',
            animat.WRONG_AVOID: 'WRONG_AVOID'}
