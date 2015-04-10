#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# constants.py

from bitarray import bitarray


# Animat stuff
DETERMINISTIC = True
ALLOW_FEEDBACK_FROM_MOTORS = False

NUM_NODES = 8
# Sensors are always the first nodes.
NUM_SENSORS = 2
# Motors are always the last nodes.
NUM_MOTORS = 2
# Hidden units are always in the middle.
NUM_HIDDEN = NUM_NODES - NUM_SENSORS - NUM_MOTORS

BODY_SIZE = 3
SENSOR_LOCATIONS = (0, 2)

ZERO_STATE = bitarray([0]) * NUM_NODES


# PLG stuff
START_CODON = bytearray([42, 213])

PLG_MAX_IN_OUT = 4

# PLG properties paired with lengths of the region they're encoded by in the
# genome, ordered as they are in the genome.
CODON_LENGTHS = (
    ('num_inputs', 1),
    ('num_outputs', 1),
    ('input_ids', PLG_MAX_IN_OUT),
    ('output_ids', PLG_MAX_IN_OUT),
)

BYTE_MAX = 255
