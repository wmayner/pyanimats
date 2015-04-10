#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# plg.py

from bitarray import bitarray

import numpy as np
import math
from .constants import (NUM_NODES, NUM_SENSORS, NUM_MOTORS, PLG_MAX_IN_OUT,
                        CODON_LENGTHS, BYTE_MAX, DETERMINISTIC, ZERO_STATE,
                        ALLOW_FEEDBACK_FROM_MOTORS)


def byte_to_num_nodes(byte):
    """Convert a byte to a number in ``range(0, NUM_NODES + 1)``."""
    return math.floor(byte / ((BYTE_MAX + 1) / (PLG_MAX_IN_OUT + 1)))


def byte_to_node_index(byte, num_nodes):
    """Convert a byte to an index of a node in ``range(0, num_nodes)``."""
    return round(((byte * num_nodes) / (BYTE_MAX + 0.1)) - 0.5)


def byte_to_input_index(byte):
    """Convert a byte to an index of an input node (if
    ``ALLOW_FEEDBACK_FROM_MOTORS`` is ``False``, motors cannot be inputs)."""
    if ALLOW_FEEDBACK_FROM_MOTORS:
        return byte_to_node_index(byte, NUM_NODES)
    else:
        return byte_to_node_index(byte, NUM_NODES - NUM_MOTORS)


def byte_to_output_index(byte):
    """Convert a byte to an index of an output node (sensors cannot be
    outputs)."""
    return byte_to_node_index(byte, NUM_NODES - NUM_SENSORS) + NUM_SENSORS


class ProbabilisticLogicGate:
    # States are `int`s where the ith bit is the state of the ith node.
    # NOTE: The genome must be at least as long as the sum of the lengths of
    # the codons.

    def __init__(self, genome, start):
        self.state = 0

        # To use when indexing into the genome, since it's circular.
        def wrap(i):
            return i % len(genome)

        # To use for reading a section of the genome (which may wrap around).
        def read_region(start, end):
            if start > end:
                return genome[start:] + genome[:end]
            else:
                return genome[start:end]

        codon = {}
        # Scan through the genome up to the probability codon, getting the
        # relevant regions that code for the different PLG properties.
        for codon_name, codon_length in CODON_LENGTHS:
            end = wrap(start + codon_length)
            codon[codon_name] = read_region(start, end)
            # Update the next codon's start index.
            start = end

        # Read the nucleotides specifying the number of inputs/outputs.
        self.num_inputs = byte_to_num_nodes(codon['num_inputs'][0])
        self.num_outputs = byte_to_num_nodes(codon['num_outputs'][0])

        # Convert the nucleotides in the input/output codons to node indices.
        input_ids = [
            byte_to_input_index(byte) for byte in codon['input_ids']]
        output_ids = [
            byte_to_output_index(byte) for byte in codon['output_ids']]
        # Only take the first `num_inputs`/`num_outputs` nodes specified by the
        # input/output codons, removing duplicates.
        self.input_ids = sorted(set(input_ids[:self.num_inputs]))
        self.output_ids = sorted(set(output_ids[:self.num_outputs]))

        # Adjust number of inputs/outputs in case duplicates were removed.
        self.num_inputs = len(self.input_ids)
        self.num_outputs = len(self.output_ids)

        # Now that we know the number of inputs and outputs, scan to get the
        # probabilities.
        M = 2**self.num_inputs
        N = 2**self.num_outputs
        end = wrap(start + M * N)
        probabilities = np.array(read_region(start, end)).reshape([M, N])

        # Make the TPM from the list of probability-nucleotides.
        if DETERMINISTIC:
            # For each past state, take the maximum probability next state as
            # being certain.
            maximums = np.argmax(probabilities, 1)
            probabilities[:, :] = 0
            probabilities[np.arange(M), maximums] = 1
            self.tpm = probabilities
        else:
            # Ensure each state has some probability.
            # TODO(wmayner) why was this done in the C++ code?
            probabilities = probabilities + 1
            # Normalize rows so they sum to 1.
            row_sums = np.sum(probabilities, 1, keepdims=True)
            self.tpm = probabilities / row_sums

    def get_next_state(self, past_full_state):
        """Returns the next state of this PLG as a bitarray of each node's
        state."""
        input_state = 0
        # Get the past states of the PLG inputs as an integer.
        for i, input_id in enumerate(self.input_ids):
            input_state |= (past_full_state[input_id] << i)
        # Get the next states of the PLG outputs as an integer.
        output_state = np.where(self.tpm[input_state] == 1)[0][0]
        # Get and return the next states of all the animat nodes as a bitarray.
        next_full_state = bitarray(ZERO_STATE)
        for i, output_id in enumerate(self.output_ids):
            # If the ith output node is on, set the output node's bit in the
            # full state.
            if (output_state & (1 << i)):
                next_full_state[i] = 1
        return next_full_state

    def __repr__(self):
        return ('\t\tProbabilisticLogicGate(\n\t\t\t' +
                '\n\t\t\t'.join([
                    "Number of inputs: " + str(self.num_inputs),
                    "Number of outputs: " + str(self.num_outputs),
                    "Input IDs: " + str(self.input_ids),
                    "Output IDs: " + str(self.output_ids),
                    "TPM: " + str(self.tpm)
                ]) + '\n\t\t)')

    def __str__(self):
        return repr(self)
