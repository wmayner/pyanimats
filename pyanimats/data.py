#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# data.py

"""Data management."""

import os
import gzip
import json
from glob import glob

from . import evolve


def load(filepath, gzipped=False):
    if gzipped:
        _open = gzip.open
    else:
        _open = open
    with _open(filepath, 'r') as f:
        return evolve.from_json(json.load(f))


def load_all(directory, pattern=os.path.join('**', 'output.json'),
             convert=True):
    """Recursively load files matching ``pattern``."""
    data = []
    paths = glob(os.path.join(directory, pattern))
    for path in paths:
        print('Loading {}...'.format(path))
        try:
            with open(path, 'r') as f:
                data.append(json.load(f))
        except Exception as e:
            print('  Error: {}'.format(e))
            print('  Could not load file; skipping...')
    if convert:
        data = list(map(evolve.from_json, data))
    return data
