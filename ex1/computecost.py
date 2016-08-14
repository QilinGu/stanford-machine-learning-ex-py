#!/usr/local/bin/python
# -*- coding:utf-8 -*-
#

# COMPUTECOST Compute cost for linear regression
#   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
#   parameter for linear regression to fit the data points in X and y


def computeCost(X, y, theta):
    """Compute cost for linear regression

    J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
    """
    # Initialize some useful values
    m = len(y)  # number of training examples

    # You need to return the following variables correctly
    J = 0
    predictions = X.dot(theta)
    sqr_errors = (predictions - y)**2
    J = 1/(2*m) * sum(sqr_errors)
    J = J[0]
    print('cost J is computed as {}'.format(J))
    return J
