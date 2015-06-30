#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# plot.py

import os
import numpy as np
import matplotlib.pyplot as plt

import analyze
from fitness_functions import LaTeX_NAMES as fit_funcnames


CASE_NAME = '0.0.10/sp/3-4-6-5/sensors-3/jumpstart-0/gen-4000'
RESULT_DIR = 'raw_results'
ANALYSIS_DIR = 'compiled_results'
RESULT_PATH = os.path.join(RESULT_DIR, CASE_NAME)
ANALYSIS_PATH = os.path.join(ANALYSIS_DIR, CASE_NAME)
FILENAMES = {
    'config': 'config.pkl',
    'hof': 'hof.pkl',
    'logbook': 'logbook.pkl',
    'lineages': 'lineages.pkl',
    'metadata': 'metadata.pkl',
}

# Utilities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def close():
    """Close a matplotlib figure window."""
    plt.close()


def _get_task_name(tasks):
    return '[' + ',\ '.join(str(task[1].count('1')) for task in tasks) + ']'


def _get_desc(config, seed=False, num_seeds=False):
    if not seed and not num_seeds:
        raise Exception('Must provide either a single seed number or the '
                        'number of seeds.')
    return (str(config['NGEN']) + '\ generations,\ ' +
            ('{}\ seeds'.format(num_seeds) if num_seeds
             else 'seed\ {}'.format(seed)) + ',\ task\ ' +
            _get_task_name(config['TASKS']) + ',\ population\ size\ '
            + str(config['POPSIZE']))


def _get_correct_trials_axis_label(config):
    return ('$\mathrm{Correct\ trials\ (out\ of\ ' + str(config['NUM_TRIALS'])
            + ')}$')


# Correct counts
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_final_correct(case_name=CASE_NAME, force=False,
                       bins=np.arange(64, 128, 2), fontsize=20, title=''):
    data = analyze.get_final_correct(case_name, force)
    correct_counts, config = data['correct_counts'], data['config']
    fig = plt.figure(figsize=(14, 12))
    plt.hist(correct_counts, bins, normed=True, facecolor='blue', alpha=0.8)
    plt.xlabel(_get_correct_trials_axis_label(config), labelpad=20,
               fontsize=fontsize)
    plt.ylabel('$\mathrm{Normalized\ number\ of\ animats}$', labelpad=20,
               fontsize=fontsize)
    plt.title(title + '$\mathrm{Histogram\ of\ animat\ performance:\ '
              + _get_desc(config, num_seeds=len(correct_counts))
              + '}$', fontsize=fontsize)
    plt.grid(True)
    fig.show()
    return fig, data


# LOD Evolution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_lods(case_name=CASE_NAME, force=False, gen_interval=500, seed=0,
              all_seeds=False, avg=False, fontsize=20, title='',
              chapter='fitness', stat='max'):
    data = analyze.get_lods(case_name, force, gen_interval, seed, all_seeds,
                            chapter, stat)
    lods, config = data['lods'], data['config']
    fig = plt.figure(figsize=(14, 12))
    if avg:
        plt.plot(np.arange(lods.shape[1]) * gen_interval, lods.mean(0))
    else:
        for row in lods:
            plt.plot(np.arange(lods.shape[1]) * gen_interval, row)
    plt.xlabel('$\mathrm{Generation}$', labelpad=20, fontsize=fontsize)
    if chapter == 'correct':
        ylabel = _get_correct_trials_axis_label(config)
    elif chapter == 'fitness':
        ylabel = ('$\mathrm{' + fit_funcnames[config.FITNESS_FUNCTION] + '}$')
    plt.ylabel(ylabel, labelpad=20, fontsize=fontsize)

    plt.title(title + '$\mathrm{' + ('Average\ a' if avg else 'A') +
              'nimat\ fitness:\ ' + _get_desc(config, num_seeds=len(lods))
              + '}$', fontsize=fontsize)
    plt.grid(True)
    fig.show()
    return fig, data
