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


# TODO figure out `.all`?
class AnimatTable(tinydb.database.Table):

    """Convert PyAnimats JSON output back into PyAnimats objects."""

    def search(self, *args, **kwargs):
        return list(map(data.load_dict, super().search(*args, **kwargs)))

    def get(self, *args, **kwargs):
        return data.load_dict(super().get(*args, **kwargs))


# Use AnimatTable as the default table for TinyDB databases.
TinyDB.table_class = AnimatTable


def insert_all(db, directory, pattern='output.json'):
    """Recursively insert files matching ``pattern``."""
    paths = glob(os.path.join(directory, '**', pattern))
    paths += glob(os.path.join(directory, '**', pattern))
    data = []
    for path in paths:
        print('Loading {}...'.format(path))
        with open(path, 'r') as f:
            data.append(json.load(f))
    print('Inserting data...')
    return db.insert_multiple(data)
