#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
#x = a1.normalize_data(x)

N_TRAIN = 100
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

# Complete the linear_regression and evaluate_regression functions of the assignment1.py
# Pass the required parameters to these functions

# list of train_err/test_err for polynomial_degree = 3
train_err = []
test_err = []
for feature_idx in range(8, 16):
    x = values[:, feature_idx - 1]
    x_train = x[0:N_TRAIN, :]
    x_test = x[N_TRAIN:, :]
    (w, tr_err) = a1.linear_regression(x_train, t_train, 'polynomial', degree = 3)
    (t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, 'polynomial', degree = 3)

    train_err.append(tr_err)
    test_err.append(te_err)

# Produce a plot of results.
plt.rcParams.update({'font.size': 15})
plt.plot(list(train_err.keys()), list(train_err.values()))
plt.plot(list(test_err.keys()), list(test_err.values()))
plt.ylabel('RMS')
plt.legend(['Training error','Testing error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()
