#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
import importlib

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)

N_TRAIN = 100
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

# Complete the linear_regression and evaluate_regression functions of the assignment1.py
# Pass the required parameters to these functions

w = {}
tr_err = {}
t_est = {}
te_err = {}

for i in range(1,6+1):
    print('degree', i)
    (w[i], tr_err[i]) = a1.linear_regression(x_train, t_train, 'polynomial', 0, i)
    (t_est[i], te_err[i]) = a1.evaluate_regression(x_test, t_test, w[i], 'polynomial', i)

# importlib.reload(a1)

# Produce a plot of results.
plt.rcParams.update({'font.size': 15})
plt.plot(list(tr_err.keys()), list(tr_err.values()))
plt.plot(list(te_err.keys()), list(te_err.values()))
plt.ylabel('RMS')
plt.legend(['Training error','Testing error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()
