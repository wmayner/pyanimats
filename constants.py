#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# constants.py

"""
Container for static constants and values derived from parameters.

Populated by :mod:`configure`.
"""

import animat


START_CODON = [42, 213]
DEFAULT_INIT_GENOME = [127] * 5000
INIT_GENOME = [127] * 5000
HIT_TYPE = {
    animat.CORRECT_CATCH: 'CORRECT_CATCH',
    animat.WRONG_CATCH: 'WRONG_CATCH',
    animat.CORRECT_AVOID: 'CORRECT_AVOID',
    animat.WRONG_AVOID: 'WRONG_AVOID',
}
