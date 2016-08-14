#!/usr/local/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from computecost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    """Performs gradient descent to learn theta

    theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha

    Instructions: Perform a single gradient step on the parameter vector theta.

    Hint: While debugging, it can be useful to print out the values
    of the cost function (computeCost) and gradient here.
    """
    # Initialize some useful values
    m = len(y)  # number of training examples
    J_history = np.zeros((num_iters, 1))

    for i in range(0, num_iters):
        theta = theta - ((1/m) * alpha * ((X.dot(theta)-y).T).dot(X)).T

        print('theta({}) {} \n'.format(i, theta))

        # Save the cost J in every iteration
        J_history[i] = computeCost(X, y, theta)

        print('J_history {}'.format(J_history))

    return theta, J_history
