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
import scipy.optimize as op
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import mpl_toolkits.mplot3d as plt3d
from matplotlib import cm
from plotdata import plotData
from costfunction import costFunction
from plotdecisionboundary import plotDecisionBoundary


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

    #  ============= Part 3: Optimizing using fminunc  =============
    #  In this exercise, you will use a built-in function (fminunc) to find the
    #  optimal parameters theta.

    #  Run fminunc to obtain the optimal theta
    #  This function will return theta and the cost

    def minimizedFunc(initial_theta, X, y):
        cost, grad = costFunction(initial_theta, X_, y)
        return cost  # must return only 1 arg

    # scipy.optimize.minimize()
    # If not arg [method] is given, chosen to be one of BFGS, L-BFGS-B, SLSQP,
    # depending if the problem has constraints or bounds.

    op_result = op.minimize(
        fun=minimizedFunc,
        x0=initial_theta,
        args=(X_, y),
        method='Nelder-Mead'
        )
    print(op_result)
    op_theta = op_result.x
    op_cost = op_result.fun

    # Print theta to screen
    print(
        'Cost at theta found by scipy.optimize.minimize: {}\n'.format(op_cost))
    print('theta: {}'.format(op_theta))

if __name__ == '__main__':
    ex2()
