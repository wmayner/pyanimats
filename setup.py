#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


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
    Extension('pyanimats/c_animat',
              sources=[
                  'pyanimats/c_animat/c_animat.pyx',
                  'pyanimats/c_animat/rng.cpp',
                  'pyanimats/c_animat/Game.cpp',
                  'pyanimats/c_animat/AbstractAgent.cpp',
                  'pyanimats/c_animat/HMM.cpp',
                  'pyanimats/c_animat/HiddenMarkovAgent.cpp',
                  'pyanimats/c_animat/LinearThreshold.cpp',
                  'pyanimats/c_animat/LinearThresholdAgent.cpp',
              ],
              language='c++',
              extra_compile_args=['-std=c++11'])
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
    name="pyanimats",
    ext_modules=cythonize(extensions),
    cmdclass={'build_ext': build_ext},
    setup_requires=setup_requires,
    install_requires=install_requires,
)
