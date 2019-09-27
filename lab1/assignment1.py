"""Basic code for assignment 1."""

import numpy as np
import pandas as pd
from scipy import nanmean

def load_unicef_data():
    """Loads Unicef data from CSV file.

    Retrieves a matrix of all rows and columns from Unicef child mortality
    dataset.

    Args:
      none

    Returns:
      Country names, feature names, and matrix of values as a tuple (countries, features, values).

      countries: vector of N country names
      features: vector of F feature names
      values: matrix N-by-F
    """
    fname = 'SOWC_combined_simple.csv'

    # Uses pandas to help with string-NaN-numeric data.
    data = pd.read_csv(fname, na_values='_', encoding='latin1')
    # Strip countries title from feature names.
    features = data.axes[1][1:]
    # Separate country names from feature values.
    countries = data.values[:,0]
    values = data.values[:,1:]
    # Convert to numpy matrix for real.
    values = np.asmatrix(values,dtype='float64')

    # Modify NaN values (missing values).
    mean_vals = nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(mean_vals, inds[1])
    return (countries, features, values)


def normalize_data(x):
    """Normalize each column of x to have mean 0 and variance 1.
    Note that a better way to normalize the data is to whiten the data (decorrelate dimensions).  This can be done using PCA.

    Args:
      input matrix of data to be normalized

    Returns:
      normalized version of input matrix with each column with 0 mean and unit variance

    """
    mvec = x.mean(0)
    stdvec = x.std(axis=0)
    
    return (x - mvec)/stdvec
    


def linear_regression(x, t, basis, reg_lambda=0, degree=0, mu=0, s=1):
    """Perform linear regression on a training set with specified regularizer lambda and basis

    Args:
      x is training inputs
      t is training targets
      reg_lambda is lambda to use for regularization tradeoff hyperparameter
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      mu,s are parameters of Gaussian basis

    Returns:
      w vector of learned coefficients
      train_err RMS error on training set
      """

    # Construct the design matrix.
    # Pass the required parameters to this function
    phi = design_matrix(basis)
            
    # Learning Coefficients
    if reg_lambda > 0:
        # regularized regression
        w = None
    else:
        # no regularization
        w = None

    # Measure root mean squared error on training data.
    train_err = None

    return (w, train_err)



def design_matrix(basis=None):
    """ Compute a design matrix Phi from given input datapoints and basis.

    Args:
        ?????

    Returns:
      phi design matrix
    """

    if basis == 'polynomial':
        phi = None
    elif basis == 'sigmoid':
        phi = None
    else: 
        assert(False), 'Unknown basis %s' % basis

    return phi


def evaluate_regression():
    """Evaluate linear regression on a dataset.

    Args:
      ?????

    Returns:
      t_est values of regression on inputs
      err RMS error on training set if t is not None
      """

    t_est = None
    err = None

    return (t_est, err)
