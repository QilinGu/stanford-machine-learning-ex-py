#!/usr/local/bin/python
# -*- coding:utf-8 -*-

#  Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions
#  in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.

from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import mpl_toolkits.mplot3d as plt3d
from matplotlib import cm
from plotdata import plotData
from costfunction import costFunction


def ex2():

    #  Load Data
    #  The first two columns contains the exam scores and the third column
    #  contains the label.

    ex2data1 = open('ex2data1.txt')
    ex2data1_txt = ex2data1.read()
    data = np.loadtxt(StringIO(
        ex2data1_txt.replace(',', ' ')))

    X = data[:, [0, 1]]
    y = data[:, 2].reshape((len(data), 1))

    #  ==================== Part 1: Plotting ====================
    #  We start the exercise by first plotting the data to understand the
    #  the problem we are working with.

    print(
        "Plotting data with + indicating (y = 1) examples and"
        "o indicating (y = 0) examples.\n")

    plotData(X, y)

    # ============ Part 2: Compute Cost and Gradient ============
    #  In this part of the exercise, you will implement the cost and gradient
    #  for logistic regression. You neeed to complete the code in
    #  costFunction.m

    #  Setup the data matrix appropriately, and add ones for the intercept term
    m = X.shape[0]
    n = X.shape[1]

    # Add intercept term to x and X_test
    X_ = np.c_[np.ones(m), X]

    # Initialize fitting parameters
    initial_theta = np.zeros((n + 1, 1))

    # Compute and display initial cost and gradient
    cost, grad = costFunction(initial_theta, X_, y)

    print('Cost at initial theta (zeros): {}\n'.format(cost))
    print('Gradient at initial theta (zeros): \n')
    print('{} \n'.format(grad))



if __name__ == '__main__':
    ex2()
