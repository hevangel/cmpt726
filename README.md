# SFU CMPT726 Machine Learning Assignments


## Lab 1 - Regression

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

## Lab 2-P5 Logistic Regression

In this question you will examine optimization for logistic regression.

1.  Download   the   assignment   2   code   and   data   from   the   website.Run   the   scriptlogisticregression.pyin  the  lr  directory.   This  code  performs  gradient  descentto findwwhich minimizes negative log-likelihood (i.e. maximizes likelihood).Include the final output of Figures 2 and 3 (plot of separator path in slope-intercept space;plot of neg. log likelihood over epochs) in your report.Why are these plots oscillating? Briefly explain why in your report.

2.  Create a Python scriptlogisticregressionmod.pyfor the following.Modifylogisticregression.pyto run gradient descent with the learning ratesη=0.5,0.3,0.1,0.05,0.01.Include in your report a single plot comparing negative log-likelihood versus epoch for thesedifferent learning rates.Compare these results. What are the relative advantages of the different rates? 

3.  Create a Python scriptlogisticregressionsgd.pyfor the following.Modify this code to do stochastic gradient descent. Use the parametersη= 0.5,0.3,0.1,0.05,0.01.Include in your report a new plot comparing negative log-likelihood versus iteration usingstochastic gradient descent.Is stochastic gradient descent faster than gradient descent?  Explain using your plots.

## Lab 2-P6 Fine-Tuning a Pre-Trained Network (ResNet20 on CIFA-100)

In this question you will experiment with fine-tuning a pre-trained network.  This is a standardworkflow in adapting existing deep networks to a new task.

We will utilize PyTorch (https://pytorch.org) a machine learning library for python. The provided code builds upon ResNet20, a state of the art deep network for image classification. The model has been trained for CIFAR-100 image classification with 100 output classes.The ResNet20 model has been adapted to solve a (simpler) different task, classifying an image asone of 10 classes on CIFAR10 dataset.

The code imagenetfinetune.py does the following:
- Constructs a deep network.  This network starts with ResNet20 up to its average poolinglayer. Then, a new linear classifier 
- Initializes the weights of the ResNet20 portion with the parameters from training on CIFAR-10.
- Performs training on only the new layers using CIFAR-10 dataset – all other weights arefixed to their values learned on ImageNet.

Start by running the code provided. It may be slow to train since the code runs on a CPU. You cantry figuring out how to change the code to train on a GPU if you have a good GPU and it couldaccelerate training. 

Try to do one of the following tasks:
- Write a Python function to be used at the end of training that generates HTML output show-ing each test image and its classification scores.  You could produce an HTML table outputfor example. (You can convert the HTML output to PDF or use screen shots.)
- Run validation of the model every few training epochs on validation or test set of the datasetand save the model with the best validation error.  Report the best validation error and thecorresponding training epoch in your report. (Do not submit saved models for this task.
- Try applying L2 regularization to the coefficients in the small networks we added.
- Try running this code on one of the datasets in torchvision.datasets (https://pytorch.org/docs/stable/torchvision/datasets.html) except CIFAR100.  You mayneed to change some layers in the network. Try creating a custom dataloader that loads datafrom your own dataset and run the code using your dataloader.  (Hints:  Your own datasetshould not come from torchvision.datasets.  A standard approach is to implement your owntorch.utils.data.Dataset and wrap it with torch.utils.data.DataLoader)
- Try modifying the structure of the new layers that were added on top of ResNet20.
- Try adding data augmentation for the training data using torchvision.transforms and then im-plementing your custom image transformation methods not available in torchvision.transforms,like gaussian blur.
- The current code is inefficient because it recomputes the output of ResNet20 every time atraining/validation example is seen, even though those layers aren’t being trained.  Change this by saving the output of ResNet20 and using these as input rather than the dataloader currently used.
- The current code does not train the other layers in ResNet20.  After training the new layersfor a while (until good values have been obtained), turn on training for the other ResNet20 layers to see if better performance can be achieved. Describe what task you do in details for this task in your report for this assignment.  If you haveany figures or tables to show, put them in the report as well.  We also request you to submit codefor this problem.
