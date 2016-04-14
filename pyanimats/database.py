#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# database.py

"""Methods for interacting with a TinyDB database of PyAnimats output."""

import os

from . import data
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
        d = data.load_all(directory, pattern, convert=False)
        if transform:
            print('Transforming data...')
            d = [transform(x) for x in d]
        print('Inserting data...')
        return db.insert_multiple(d)


def in_range(value, begin, end):
    return begin < value < end
