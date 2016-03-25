#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# data.py

"""Data management."""

import gzip
import json

from . import evolve


def load(filepath, gzipped=False):
    if gzipped:
        _open = gzip.open
    else:
        _open = open
    with _open(filepath, 'r') as f:
        return evolve.from_json(json.load(f))
