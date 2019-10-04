#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
import importlib
from collections import defaultdict

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)

N_TRAIN = 100
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

bias = 1
degree = 2
reg_lambda_list = [0, 0.01, 0.1, 1, 10, 100, 1000, 10000]

# Complete the linear_regression and evaluate_regression functions of the assignment1.py
# Pass the required parameters to these functions

w = defaultdict(dict)
tr_err = defaultdict(dict)
v_est = defaultdict(dict)
v_err = defaultdict(dict)
v_err_avg = {}
t_est = {}
te_err = {}

for reg_lambda in reg_lambda_list:
    print('reg_lambda', reg_lambda)
    for cross in range(0,10):
        validate_range = list(range(cross*10,(cross+1)*10))
        train_range = [i for i in range(0,100) if i not in validate_range]

        x_train2 = x_train[train_range,:]
        x_validate = x_train[validate_range]
        t_train2 = t_train[train_range]
        t_validate = t_train[validate_range]

        (w[reg_lambda][cross], tr_err[reg_lambda][cross]) = a1.linear_regression(x_train2, t_train2, 'polynomial',
                                                                reg_lambda=reg_lambda, degree=degree, bias=bias)
        (v_est[reg_lambda][cross], v_err[reg_lambda][cross]) = a1.evaluate_regression(x_validate, t_validate,
                                                                w[reg_lambda][cross], 'polynomial', degree=degree, bias=bias)
    v_err_avg[reg_lambda] = sum(v_err[reg_lambda].values())/10
    min_w = sorted(v_err[reg_lambda].items(), key=lambda item: item[1])[0][0]
    (t_est[reg_lambda], te_err[reg_lambda]) = a1.evaluate_regression(x_test, t_test, w[reg_lambda][min_w],
                                                                     'polynomial', degree=degree, bias=bias)

# importlib.reload(a1)

# Produce a plot of results.
plt.rcParams.update({'font.size': 15})
plt.semilogx(list(te_err.keys()), list(te_err.values()))
plt.semilogx(list(v_err_avg.keys()), list(v_err_avg.values()))
plt.show()
