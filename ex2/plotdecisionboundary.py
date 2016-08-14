#!/usr/local/bin/python
# -*- coding:utf-8 -*-

from plotdata import plotData
import matplotlib.pyplot as plt
import numpy as np
from mapfeature import mapFeature


def plotDecisionBoundary(theta, X, y):
    """Plots the data points X and y into a new figure with
       the decision boundary defined by theta

    PLOTDECISIONBOUNDARY(theta, X, y) plots the data points with + for the
    positive examples and o for the negative examples. X is assumed to be
    a either
    1) Mx3 matrix, where the first column is an all-ones column for the
       intercept.
    2) MxN, N>3 matrix, where the first column is all-ones
    """

    plotData(X[:, 1:3], y, should_call_show=False)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = [min(X[:, 1])-2,  max(X[:, 1])+2]

        # Calculate the decision boundary line
        plot_y = np.multiply(
            (-1./theta[2]),
            (np.multiply(theta[1], plot_x) + theta[0])
            )

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y, label='Decision Boundary')

        # Legend, specific for the exercise
        plt.xlabel('Exam 1 score')
        plt.ylabel('Exam 2 score')
        plt.legend()
        plt.axis([30, 100, 30, 100])
        plt.show()
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros(len(u), len(v))
        # Evaluate z = theta*x over the grid
        for i in range(0, len(u)):
            for j in range(0, len(v)):
                z[i, j] = mapFeature(u[i], v[j]).dot(theta)

        z = z.T  # important to transpose z before calling contour

        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        meshgrid_x, meshgrid_y = np.meshgrid(u, v)
        CS = plt.contour(
            meshgrid_x,
            meshgrid_y,
            z,
            50,
            )
        plt.clabel(CS, inline=1, fontsize=10)
        plt.show()
    return
