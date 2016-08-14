#!/usr/local/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sigmoid import sigmoid


def costFunction(theta, X, y):
    """Compute cost and gradient for logistic regression

    J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    parameter for logistic regression and the gradient of the cost
    w.r.t. to the parameters.

    Instructions: Compute the cost of a particular choice of theta.
              You should set J to the cost.
              Compute the partial derivatives and set grad to the partial
              derivatives of the cost w.r.t. each parameter in theta

    Note: grad should have the same dimensions as theta
    """

    # Initialize some useful values
    m = len(y)  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    h_x = sigmoid(X.dot(theta))
    J = (1/m)*((-y.T).dot(np.log(h_x)) - ((1-y).T).dot(np.log(1-h_x)))
    grad = (1/m)*(X.T.dot(h_x-y))

    return J, grad
