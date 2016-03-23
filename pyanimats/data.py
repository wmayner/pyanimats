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


def load(filepath, gzipped=False):
    if gzipped:
        _open = gzip.open
    else:
        _open = open
    with _open(filepath, 'r') as f:
        return load_evolution(json.load(f))
