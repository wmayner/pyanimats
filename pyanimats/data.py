#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# data.py

"""Data management."""

import gzip
import json

import dateutil.parser
from munch import Munch

from . import animat
from .experiment import Experiment


def load_evolution(d):
    d = Munch(d)
    d.simulation = Munch(d.simulation)
    d.experiment = Experiment(d.experiment)
    d.time = dateutil.parser.parse(d.time)
    # Restore population
    lineage = list(
        map(lambda a: animat.from_json(a, experiment=d['experiment']),
            d['lineage']))
    for i in range(len(lineage) - 1):
        lineage[i].parent = lineage[i + 1]
    d.lineage = lineage
    return d


def load(filepath, gzipped=False):
    if gzipped:
        _open = gzip.open
    else:
        _open = open
    with _open(filepath, 'r') as f:
        return load_dict(json.load(f))
