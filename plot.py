#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# plot.py

import pickle
import matplotlib.pyplot as plt

DATA_FILE = 'correct_counts.pkl'

with open(DATA_FILE, 'rb') as f:
    data = pickle.load(f)

n, bins, patches = plt.hist(data, 12, normed=True, facecolor='blue', alpha=0.8)

plt.xlabel('$\mathrm{Fitness}$')
plt.ylabel('$\mathrm{Number\ of\ Animats}$')
plt.title('$\mathrm{Histogram\ of\ Animat Performance:\ 30,000\ '
          'generations,\ population\ size\ 100}$')
plt.grid(True)

plt.show()
