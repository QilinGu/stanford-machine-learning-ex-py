#!/usr/local/bin/python
# -*- coding:utf-8 -*-

import numpy as np


def mapFeature(X1, X2):
    """Feature mapping function to polynomial features

    MAPFEATURE(X1, X2) maps the two input features
    to quadratic features used in the regularization exercise.

    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

    Inputs X1, X2 must be the same size
    """

    DEGREE = 6
    out = np.ones(len(X1[:, 0]))
    end = 0
    for i in range(1, DEGREE+1):
        for j in range(0, i+1):
            end = end + 1
            out[:, end+1] = (np.power(X1, i-j)).dot(np.power(X2, j))
    return out
