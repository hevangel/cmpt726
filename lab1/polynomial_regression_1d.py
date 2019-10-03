#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
import importlib
import pandas as pd

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
#x = a1.normalize_data(x)

N_TRAIN = 100
bias = 0

w = {}
tr_err = {}
t_est = {}
te_err = {}

for col in range(8,15+1):
    print('column:', col)

    x_train = x[0:N_TRAIN,col]
    x_test = x[N_TRAIN:,col]
    t_train = targets[0:N_TRAIN]
    t_test = targets[N_TRAIN:]

    (w[col], tr_err[col]) = a1.linear_regression(x_train, t_train, 'polynomial', 0, 3, bias)
    (t_est[col], te_err[col]) = a1.evaluate_regression(x_test, t_test, w[col], 'polynomial', 3, bias)

# importlib.reload(a1)

# Produce a plot of results.
#plt.rcParams.update({'font.size': 15})
#plt.plot(list(tr_err.keys()), list(tr_err.values()))
#plt.plot(list(te_err.keys()), list(te_err.values()))
#plt.ylabel('RMS')
#plt.legend(['Training error','Testing error'])
#plt.title('Fit with polynomials, no regularization')
#plt.xlabel('Polynomial degree')
#plt.show()
