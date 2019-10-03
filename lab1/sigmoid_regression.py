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
bias = 1

w = {}
tr_err = {}
t_est = {}
te_err = {}

col = 11
mu_list = [100, 10000]

x_train = x[0:N_TRAIN, col]
x_test = x[N_TRAIN:, col]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]


for mu in mu_list:
    print('mu =', mu)
    (w[mu], tr_err[mu]) = a1.linear_regression(x_train, t_train, 'sigmoid', mu=mu, s=2000.0)
    (t_est[mu], te_err[mu]) = a1.evaluate_regression(x_test, t_test, w[mu], 'sigmoid', mu=mu, s=2000.0)

# importlib.reload(a1)

## Produce a plot of results.
#plt.close('all')
#
## visualize_1d
df = {}
t2 = {}
xval = np.squeeze(x[:, col], axis=1).tolist()[0]
t = np.squeeze(targets, axis=1).tolist()[0]
for mu in mu_list:
    (t_est, te_err) = a1.evaluate_regression(x[:,col], t, w[mu], 'sigmoid', mu=mu, s=2000.0)
    t2[mu] = np.squeeze(t_est, axis=1).tolist()[0]

df = pd.DataFrame({'t':t, 't2':t2[100], 't3':t2[10000]}, index=xval)
ax = df.reset_index().plot.scatter(x='index', y='t')
df.reset_index().plot.scatter(x='index', y='t2', color='Red', ax=ax)
df.reset_index().plot.scatter(x='index', y='t3', color='Green', ax=ax)
plt.show()



