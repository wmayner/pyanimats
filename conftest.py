#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os


collect_ignore = [
    "setup.py",
]
# Also ignore everything that git ignores.
git_ignore = os.path.join(os.path.dirname(__file__), '.gitignore')
collect_ignore += list(filter(None, open(git_ignore).read().split('\n')))
