#!/usr/local/bin/python
# -*- coding:utf-8 -*-

import numpy as np


def sigmoid(x):
    """Compute sigmoid functoon

    J = SIGMOID(z) computes the sigmoid of z.

    You need to return the following variables correctly
    g = zeros(size(z));

    Instructions: Compute the sigmoid of each value of z
    (z can be a matrix,vector or scalar).
    """

    return 1 / (1 + np.exp(-x))
