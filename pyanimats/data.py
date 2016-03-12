#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# data.py

"""Data management."""

import json

from munch import Munch

import animat
from experiment import Experiment


def load_dict(d):
    """Initialize an animat object from a JSON dictionary.

    Checks that the stored TPM and connectivity matrix match those encoded by
    the stored genome.
    """
    d = Munch(d)
    d.experiment = Experiment(d.experiment)
    # Restore population
    lineage = list(
        map(lambda a: animat.from_json(a, experiment=d['experiment']),
            d['lineage']))
    for i in range(len(lineage) - 1):
        lineage[i].parent = lineage[i + 1]
    d.lineage = lineage
    return d


def load(filepath):
    with open(filepath, 'r') as f:
        return load_dict(json.load(f))
