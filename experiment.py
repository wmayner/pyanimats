#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# experiment.py

import pprint

import yaml
from munch import Munch


class Experiment(Munch):

    """A parameter set specifying an evolutionary simulation.

    Parameters can be accessed as attributes, i.e. with dot notation.

    Keyword Args:
        filepath (string): A file path pointing to a YAML file containing
            experiment parameters.
        dictionary (string): A dictionary of experiment parameters. Values in
            this dictionary overwrite values from the file.

    Example:
        >>> e = Experiment('experiments/example.yml')
        >>> e.num_sensors
        3
    """

    def __init__(self, filepath=None, dictionary=None):
        # Store the filepath in case the user needs to remember it later.
        self.filepath = filepath
        # Load the given YAML file if provided.
        if filepath is not None:
            with open(filepath) as f:
                self.update(yaml.load(f))
        # Update from the dictionary if provided.
        if dictionary is not None:
            self.update(dictionary)

    def __repr__(self):
        return ('Experiment({\n ' +
                indent(pprint.pformat(dict(self), indent=1)[1:-1], amount=1) +
                '\n})')


def indent(lines, amount=2, chr=' '):
    """Indent a string.

    Prepends whitespace to every line in the passed string. (Lines are
    separated by newline characters.)

    Args:
        lines (str): The string to indent.

    Keyword Args:
        amount (int): The number of columns to indent by.
        chr (char): The character to to use as the indentation.

    Returns:
        str: The indented string.
    """
    lines = str(lines)
    padding = amount * chr
    return padding + ('\n' + padding).join(lines.split('\n'))
