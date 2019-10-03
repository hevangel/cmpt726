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

bias = 1
N_TRAIN = 100
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

w = {}
tr_err = {}
t_est = {}
te_err = {}


for col in range(8,15+1):
    print('column:', col)

    x_train = values[0:N_TRAIN,col]
    x_test = values[N_TRAIN:,col]

    (w[col], tr_err[col]) = a1.linear_regression(x_train, t_train, 'polynomial', 0, 3, bias)
    (t_est[col], te_err[col]) = a1.evaluate_regression(x_test, t_test, w[col], 'polynomial', 3, bias)

# importlib.reload(a1)


# Produce a plot of results.
plt.close('all')

df = pd.DataFrame([tr_err, te_err], index=['tr_err', 'te_err'])
df.T.plot.bar()
plt.show()

# visualize_1d
t = np.squeeze(targets, axis=1).tolist()[0]
df = {}
for col in range(11,13+1):
    xval = np.squeeze(x[:, col], axis=1).tolist()[0]
    (t_est, te_err) = a1.evaluate_regression(x[:,col], t, w[col], 'polynomial', 3, bias)
    t2 = np.squeeze(t_est, axis=1).tolist()[0]
    df[col] = pd.DataFrame({'t':t, 't2':t2}, index=xval)
    ax = df[col].reset_index().plot.scatter(x='index', y='t')
    df[col].reset_index().plot.scatter(x='index', y='t2', color='Red', ax=ax)
    plt.title(col)
    plt.show()



