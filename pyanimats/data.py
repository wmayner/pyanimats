#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# data.py

"""Data management."""

import os
import gzip
import json
from glob import glob

from . import evolve


def load(filepath, convert=True, compressed=False, quiet=False, last=False,
         progress=False):
    if not quiet:
        prefix = '\r' if progress else ''
        print(prefix + 'Loading {}...'.format(filepath), end='', flush=True)
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


def load_all(directory, pattern=os.path.join('output*.json*'), **kwargs):
    """Recursively load files in ``directory`` matching ``pattern``."""
    d = []
    for path in glob(os.path.join(directory, pattern)):
        try:
            d.append(load(path, progress=True, **kwargs))
        except Exception as e:
            print('\n  Error: {}'.format(e))
            print('  Could not load file; skipping...')
    return d
