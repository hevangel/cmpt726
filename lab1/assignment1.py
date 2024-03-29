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
    


def linear_regression(x, t, basis, reg_lambda=0, degree=0, bias=0, mu=0, s=1):
    """Perform linear regression on a training set with specified regularizer lambda and basis

    Args:
      x is training inputs
      t is training targets
      reg_lambda is lambda to use for regularization tradeoff hyperparameter
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      bias has bias term or not
      mu,s are parameters of Gaussian basis

    Returns:
      w vector of learned coefficients
      train_err RMS error on training set
      """

    # Construct the design matrix.
    # Pass the required parameters to this function
    phi = design_matrix(basis, degree, x, bias, mu, s)
    phi_pinv = np.linalg.pinv(phi)
            
    # Learning Coefficients
    if reg_lambda > 0:
        # regularized regression
        w = np.linalg.inv(phi.T.dot(phi) + reg_lambda * np.identity(phi.shape[1])).dot(phi.T).dot(t)
    else:
        # no regularization
        w = phi_pinv.dot(t)

    # Measure root mean squared error on training data.
    new_t = phi.dot(w)
    train_err = np.sqrt(sum(np.square(t - new_t))/x.shape[0])[0,0]

    return (w, train_err)



def design_matrix(basis=None, degree=None, x=None, bias=None, mu=None, s=None):
    """ Compute a design matrix Phi from given input datapoints and basis.

    Args:
        degree is the degree of the polynomials
        x is the training data
        bias - 0/1 bias term

    Returns:
      phi design matrix:w

    """

    if basis == 'polynomial':
        phi = np.ones((x.shape[0],bias))
        for i in range(1,degree+1):
            phi = np.hstack((phi, np.power(x,i)))
    elif basis == 'sigmoid':
        phi = np.ones((x.shape[0],bias))
        for m in mu:
            phi = np.hstack((phi, 1/(1+np.exp((m-x)/s))))
    else:
        assert(False), 'Unknown basis %s' % basis

    return phi


def evaluate_regression(x, t, w, basis, degree=0, bias=0, mu=0, s=1):
    """Evaluate linear regression on a dataset.

    Args:
      x - inputs
      t - output
      w - weight matrix
      degree - degree of the polynomial
      bias - has bias term or not

    Returns:
      t_est values of regression on inputs
      err RMS error on training set if t is not None
      """

    phi = design_matrix(basis, degree, x, bias, mu, s)
    t_est = phi.dot(w)
    err = np.sqrt(sum(np.square(t - t_est)) / x.shape[0])[0, 0]

    return (t_est, err)
