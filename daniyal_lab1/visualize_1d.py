#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

# Plot a curve showing learned function.
# Use linspace to get a set of samples on which to evaluate
#x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
#x_ev = np.linspace(np.asscalar(min(min(x_train[:,f]),min(x_test[:,f]))),
#                   np.asscalar(max(max(x_train[:,f]),max(x_test[:,f]))), num=500)
x1_ev = np.linspace(0, 10, num=500)
x2_ev = np.linspace(0, 10, num=50)

# TO DO::
# Perform regression on the linspace samples.
# Put your regression estimate here in place of y_ev.
y1_ev = np.random.random_sample(x1_ev.shape)
y2_ev = np.random.random_sample(x2_ev.shape)
y1_ev = 100*np.sin(x1_ev)
y2_ev = 100*np.sin(x2_ev)

plt.plot(x1_ev,y1_ev,'r.-')
plt.plot(x2_ev,y2_ev,'bo')
plt.title('Visualization of a function and some data points')
plt.show()