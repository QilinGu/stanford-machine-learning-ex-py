#!/usr/local/bin/python
# -*- coding:utf-8 -*-
#
# lrcostfunction
#

import numpy as np
from sigmoid import sigmoid


def lrCostFunction(theta, X, y, lambda_):
    """Compute cost and gradient for logistic regression with
    regularization.

    J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.

    Instructions: Compute the cost of a particular choice of theta.
                  You should set J to the cost.
                  Compute the partial derivatives and set grad to the partial
                  derivatives of the cost w.r.t. each parameter in theta

    Hint: The computation of the cost function and gradients can be
          efficiently vectorized. For example, consider the computation

              sigmoid(X * theta)

          Each row of the resulting matrix will contain the value of the
          prediction for that example. You can make use of this to vectorize
          the cost function and gradient computations.

    Hint: When computing the gradient of the regularized cost function,
          there're many possible vectorized solutions, but one solution
          looks like:
              grad = (unregularized gradient for logistic regression)
              temp = theta;
              temp(1) = 0;   # because we don't add anything for j = 0
              grad = grad + YOUR_CODE_HERE (using the temp variable)
    """

    # Initialize some useful values
    m = len(y)  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)
    ## print('theta shape {}'.format(theta.shape))

    # First calculate the unregularized cost (J)
    # and gradient (grad) for logistic regression
    # (Taken straight from Coding Exercise 2 (costFunction.m))

    h_x = sigmoid(X.dot(theta))
    J = (1/m)*((-y.T).dot(np.log(h_x)) - ((1-y).T).dot(np.log(1-h_x)))
    grad = (1/m)*(X.T.dot(h_x-y))

    # Next calculate the regularized cost (J)
    # and gradient (grad) for logistic regression
    # (Taken straight from Coding Exercise 2 (costFunctionReg.m))

    # this effectively ignores "theta zero" in the following calculations
    ## print(theta)
    theta_zeroed_first = np.array([np.append([0], theta[1:len(theta)])]).T
    J = J + lambda_/(2 * m) * sum(theta_zeroed_first**2)
    ## print('grad shape {}'.format(grad.shape))
    ## print('theta_zeroed_first shape {}'.format(theta_zeroed_first.shape))
    ## print('grad {}'.format(grad))
    grad = np.add(grad, ((lambda_/m) * theta_zeroed_first))
    ## print('theta_zeroed_first {}'.format(theta_zeroed_first))
    print('grad (added regularized cost) {}'.format(grad))

    return J, grad
