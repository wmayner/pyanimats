#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension('core',
              sources=[
                  'lib/core.pyx',
                  'lib/Agent.cpp',
                  'lib/HMM.cpp',
                  'lib/Game.cpp'
              ],
              language='c++')
]

setup(
    name="animats",
    ext_modules=cythonize(extensions)
)
