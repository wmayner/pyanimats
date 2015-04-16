#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension('animat',
              sources=[
                  'animat/animat.pyx',
                  'animat/Agent.cpp',
                  'animat/HMM.cpp',
                  'animat/Game.cpp'
              ],
              language='c++')
]

setup(
    name="animat",
    ext_modules=cythonize(extensions)
)
