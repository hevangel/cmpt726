#!/usr/bin/env python

# Run logistic regression training.

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import assignment2 as a2
import math
import random

# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 5000
tol = 0.00001

# Step size for gradient descent.
#eta = 0.5
#etas = [0.5, 0.3, 0.1, 0.05, 0.01]
etas = [0.5]
eta_e_all = {}

# Load data.
data = np.genfromtxt('data.txt')

# Data matrix, with column of ones at end.
X = data[:, 0:3]

# Target values, 0 for class 1, 1 for class 2.
t = data[:, 3]

# For plotting data
class1 = np.where(t == 0)
X1 = X[class1]
class2 = np.where(t == 1)
X2 = X[class2]

for eta in etas:

  # Initialize w.
  w = np.array([0.1, 0, 0])
  w_old = w

  # Error values over all iterations.
  e_all = []

  DATA_FIG = 1

  # Set up the slope-intercept figure
  #SI_FIG = 2
  #plt.figure(SI_FIG, figsize=(8.5, 6))
  #plt.rcParams.update({'font.size': 15})
  #plt.title('Separator in slope-intercept space')
  #plt.xlabel('slope')
  #plt.ylabel('intercept')
  #plt.axis([-5, 5, -10, 0])


  for iter in range(0, max_iter):

    # Compute output using current w on all data X.
    #index = iter % X.shape[0]
    index = random.randint(0, X.shape[0]-1)

    y = sps.expit(np.dot(X[index], w))

    # e is the error, negative log-likelihood (Eqn 4.90)
    #e = -np.multiply(t[index], np.log(y)) + np.multiply((1-t[index]), np.log(1-y))
    e = -(t[index]*math.log(y) + (1-t[index])*math.log(1-y))

    # Add this error to the end of error vector.
    e_all.append(e)

    # Gradient of the error, using Eqn 4.91
    grad_e = np.multiply((y - t[index]), X[index].T)

    # Update w, *subtracting* a step in the error derivative since we're minimizing
    w_old = w
    w = w - eta*grad_e

    # Plot current separator and data.  Useful for interactive mode / debugging.
    plt.figure(DATA_FIG)
    plt.clf()
    plt.plot(X1[:,0],X1[:,1],'b.')
    plt.plot(X2[:,0],X2[:,1],'g.')
    if t[index] == 0:
      plt.plot(X[index,0], X[index,1], 'bx')
    else:
      plt.plot(X[index,0], X[index,1], 'gx')
    a2.draw_sep(w)
    a2.draw_sep(w_old, line='y-')
    plt.axis([-5, 15, -10, 10])
    plt.show()

    # Add next step of separator in m-b space.
    #plt.figure(SI_FIG)
    #a2.plot_mb(w, w_old)
    #plt.show()

    # Print some information.
    print('epoch {0:d}, y={1:.2f}, t={2}, negative log-likelihood {3:.4f}, w={4}'.format(iter, y, t[index], e, w.T))
    temp = 1

    # Stop iterating if error doesn't change more than tol.
    #if iter > 0:
    #  if np.absolute(e-e_all[iter-1]) < tol:
    #    break

  eta_e_all[eta] = e_all


# Plot error over iterations
TRAIN_FIG = 3
plt.figure(TRAIN_FIG, figsize=(8.5, 6))
for eta in etas:
  plt.plot(eta_e_all[eta], label=eta)
plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression')
plt.xlabel('Epoch')
plt.legend()
plt.show()
