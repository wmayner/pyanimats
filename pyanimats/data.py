#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# data.py

"""Data management."""

import os
import gzip
import json
from glob import glob

from tqdm import tqdm

from . import evolve


def load(filepath, convert=True, compressed=False, last=False):
    _, ext = os.path.splitext(filepath)
    if compressed or ext in ['.gz', '.gzip']:
        with gzip.open(filepath, 'rb') as f:
            d = json.loads(f.read().decode('ascii'))
    else:
        with open(filepath, 'rt') as f:
            d = json.load(f)
    if convert or last:
        d = evolve.from_json(d)
        if last:
            d = d['lineage'][0]
    return d


def load_all(directory, pattern=os.path.join('output*.json*'), last=False,
             **kwargs):
    """Recursively load files in ``directory`` matching ``pattern``."""
    d = []
    paths = glob(os.path.join(directory, pattern))
    for path in tqdm(paths, leave=False, dynamic_ncols=True):
        try:
            d.append(load(path, last=last, **kwargs))
        except Exception as e:
            print('Could not load file `{}`.\n'
                  '  Error: {}\n'
                  '  Skipping...'.format(path, e))

    def sort_func(x):
        if last:
            return x.rng_seed
        else:
            return x.experiment.rng_seed

    return sorted(d, key=sort_func)
