# SFU CMPT726 Machine Learning Assignments


## Assignment 1 - Regression

In this question you will train models for regression and analyze a dataset. Start by downloading
the code and dataset from the website.

The dataset is created from data provided by UNICEF’s State of theWorld’s Children 2013 report:
http://www.unicef.org/sowc2013/statistics.html

Child mortality rates (number of children who die before age 5, per 1000 live births) for 195
countries, and a set of other indicators are included.

### 1. Getting started

Run the provided script polynomial regression.py to load the dataset and names of countries
/ features.

Answer the following questions about the data. Include these answers in your report.
1. Which country had the highest child mortality rate in 1990? What was the rate?
2. Which country had the highest child mortality rate in 2011? What was the rate?
3. Some countries are missing some features (see original .xlsx/.csv spreadsheet). How is this

handled in the function assignment1.load unicef data()?
For the rest of this question use the following data and splits for train/test and cross-validation.
- Target value: column 2 (Under-5 mortality rate (U5MR) 2011)1.
- Input features: columns 8-40.
- Training data: countries 1-100 (Afghanistan to Luxembourg).
- Testing data: countries 101-195 (Madagascar to Zimbabwe).
- Cross-validation: subdivide training data into folds with countries 1-10 (Afghanistan to Austria),
11-20 (Azerbaijan to Bhutan), ... . I.e. train on countries 11-100, validate on 1-10; train on
1-10 and 21-100, validate on 11-20, ...

### 2. Polynomial Regression

Implement linear basis function regression with polynomial basis functions. Use only monomials
of a single variable (x1; x2^2; x2^2) and no cross-terms (x1  x2).

Perform the following experiments:

1. Create a python script polynomial regression.py for the following.
Fit a polynomial basis function regression (unregularized) for degree 1 to degree 6 polynomials.
Include bias term. Plot training error and test error (in RMS error) versus polynomial
degree.

Put this plot in your report, along with a brief comment about what is “wrong” in your report.
Normalize the input features before using them (not the targets, just the inputs x). Use
assignment1.normalize data().

Run the code again, and put this new plot in your report.

2. Create a python script polynomial regression 1d.py for the following.

Perform regression using just a single input feature.

Try features 8-15 (Total population - Low birthweight). For each (un-normalized) feature fit
a degree 3 polynomial (unregularized). Try with and without a bias term.

Plot training error and test error (in RMS error) for each of the 8 features. This should be
as bar charts (e.g. use matplotlib.pyplot.bar())—one for models with bias term,
and another for models without bias term.

Put the two bar charts in your report.
The testing error for feature 11 (GNI per capita) is very high. To see what happened, produce
plots of the training data points, learned polynomial, and test data points. The code
visualize 1d.py may be useful.

In your report, include plots of the fits for degree 3 polynomials for features 11 (GNI), 12
(Life expectancy), 13 (literacy).

### 3. Sigmoid Basis Functions

1. Create a python script sigmoid regression.py for the following.
Implement regression using sigmoid basis functions for a single input feature. Use two
sigmoid basis functions, with mu = 100; 10000 and s = 2000:0. Include a bias term. Use
un-normalized features.

Fit this regression model using feature 11 (GNI per capita).
In your report, include a plot of the fit for feature 11 (GNI).
In your report, include the training and testing error for this regression model.

### 4. Regularized Polynomial Regression

1. Create a python script polynomial regression reg.py for the following.
Implement L2-regularized regression. Fit a degree 2 polynomial using lambda = f0; :01; :1; 1; 10; 102; 103; 104g.
Use normalized features as input. Include a bias term. Use 10-fold cross-validation to decide
on the best value for lambda. Produce a plot of average validation set error versus lambda. Use a
matplotlib.pyplot.semilogx plot, putting lambda on a log scale2.

Put this plot in your report, and note which lambda value you would choose from the crossvalidation.


