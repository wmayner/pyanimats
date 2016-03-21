#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# db.py

"""Methods for interacting with a TinyDB database of PyAnimats output."""

import json
import os
from glob import glob

import tinydb
from tinydb import TinyDB

from . import data


def true(x):
    return True


# TODO figure out `.all`?
class PyAnimatsTable(tinydb.database.Table):

    """Convert PyAnimats JSON output back into PyAnimats objects."""

    def all(self, *args, **kwargs):
        return [data.load_evolution(x) for x in super().all(*args, **kwargs)]

    def get(self, *args, **kwargs):
        if not args:
            args = [true]
        return super().get(*args, **kwargs)


# Use PyAnimatsTable as the default table for TinyDB databases.
TinyDB.table_class = PyAnimatsTable


def insert_all(db, directory, pattern=os.path.join('**', 'output.json')):
    """Recursively insert files matching ``pattern``."""
    paths = glob(os.path.join(directory, pattern))
    data = []
    for path in paths:
        print('Loading {}...'.format(path))
        with open(path, 'r') as f:
            data.append(json.load(f))
    print('Inserting data...')
    return db.insert_multiple(data)


def in_range(value, begin, end):
    return begin < value < end
