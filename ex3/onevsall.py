#!/usr/local/bin/python
# -*- coding:utf-8 -*-
#
# onevsall.py
#

import numpy as np
import scipy.optimize as op
from lrcostfunction import lrCostFunction


def oneVsAll(X, y, num_labels, lambda_):
    """oneVsAll trains multiple logistic regression classifiers and returns all
    the classifiers in a matrix all_theta, where the i-th row of all_theta
    corresponds to the classifier for label i.

    [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
    logisitc regression classifiers and returns each of these classifiers
    in a matrix all_theta, where the i-th row of all_theta corresponds
    to the classifier for label i


    Instructions: You should complete the following code to train num_labels
                  logistic regression classifiers with regularization
                  parameter lambda.

    Hint: theta(:) will return a column vector.

    Hint: You can use y == c to obtain a vector of 1's and 0's that tell use
          whether the ground truth is true/false for this class.

    Note: For this assignment, we recommend using fmincg to optimize the cost
          function. It is okay to use a for-loop (for c = 1:num_labels) to
          loop over the different classes.

          fmincg works similarly to fminunc, but is more efficient when we
          are dealing with large number of parameters.

    Example Code for fmincg:

        # Set Initial theta
        initial_theta = zeros(n + 1, 1);

        # Set options for fminunc
        options = optimset('GradObj', 'on', 'MaxIter', 50);

        # Run fmincg to obtain the optimal theta
        # This function will return theta and the cost
        [theta] = ...
            fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
                    initial_theta, options);

    """

    # Some useful variables
    m = X.shape[0]
    n = X.shape[1]

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n+1))

    # Add ones to the X data matrix
    X = np.c_[np.ones(m), X]

    initial_theta = np.zeros((X.shape[1], 1))

    def minimizedFunc(initial_theta, X, y, lambda_):
        cost, grad = lrCostFunction(initial_theta, X, y, lambda_)
        return cost  # must return only 1 arg

    for c in range(0, num_labels):
        print('------------------------------------------------------------------------label {}'.format(c))
        """
        op_result = op.minimize(
            fun=minimizedFunc,
            x0=initial_theta,
            args=(X, y, lambda_),
            method='Nelder-Mead'
            )
        """
        y = np.matrix([1 if y[j] == c else 0 for j in range(m)]).T
        op_result = op.fmin_cg(
            f=minimizedFunc,
            x0=initial_theta,
            maxiter=50,
            args=(X, y, lambda_)
            )
        """
        op_result = op.fmin_bfgs(
            f=minimizedFunc,
            x0=initial_theta,
            args=(X, y, lambda_),
            maxiter=50)
        """
        print(op_result)
        op_theta = op_result.x
        all_theta[c, :] = op_theta.T

    return all_theta
