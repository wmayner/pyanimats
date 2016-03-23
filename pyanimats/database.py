#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# database.py

"""Methods for interacting with a TinyDB database of PyAnimats output."""

import json
import os
from glob import glob

from . import evolve


def true(x):
    return True


def get(db, *args, **kwargs):
    if not args:
        args = [true]
    return evolve.from_json(db.get(*args, **kwargs))


def search(db, *args, **kwargs):
    if not args:
        args = [true]
    return map(evolve.from_json, db.search(*args, **kwargs))


def insert_all(database, directory, pattern=os.path.join('**', 'output.json'),
               transform=None):
    """Recursively insert files matching ``pattern``."""
    with database as db:
        data = []
        paths = glob(os.path.join(directory, pattern))
        for path in paths:
            print('Loading {}...'.format(path))
            with open(path, 'r') as f:
                data.append(json.load(f))
        if transform:
            print('Transforming data...')
            data = [transform(d) for d in data]
        print('Inserting data...')
        return db.insert_multiple(data)


def in_range(value, begin, end):
    return begin < value < end
