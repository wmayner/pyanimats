#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize


# See http://stackoverflow.com/a/21621689/1085344
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        if hasattr(__builtins__, '__NUMPY_SETUP__'):
            __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


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

setup_requires = [
    'numpy'
]

install_requires = setup_requires + [
    'yaml'
    'deap'
    'docopt'
    'pyphi'
]

setup(
    name="animat",
    ext_modules=cythonize(extensions),
    cmdclass={'build_ext': build_ext},
    setup_requires=setup_requires,
    install_requires=install_requires,
)
