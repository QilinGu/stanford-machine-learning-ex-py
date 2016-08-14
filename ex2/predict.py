#!/usr/local/bin/python
# -*- coding:utf-8 -*-

from sigmoid import sigmoid
import numpy as np


def predict(theta, X):
    """Predict whether the label is 0 or 1 using learned logistic

    regression parameters theta
    p = PREDICT(theta, X) computes the predictions for X using a
    threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

    Instructions: Complete the following code to make predictions using
                  your learned logistic regression parameters.
                  You should set p to a vector of 0's and 1's
    """

    # You need to return the following variables correctly
    h_x = sigmoid(X.dot(theta))
    return np.double(h_x >= 0.5)
