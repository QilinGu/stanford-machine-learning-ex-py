#!/usr/local/bin/python
# -*- coding:utf-8 -*-
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     warmUpExercise.m
#     plotData.m
#     gradientDescent.m
#     computeCost.m
#     gradientDescentMulti.m
#     computeCostMulti.m
#     featureNormalize.m
#     normalEqn.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#

from io import StringIO
import numpy as np
from warmupexercise import warmUpExercise
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import mpl_toolkits.mplot3d as plt3d
from matplotlib import cm
from plotdata import plotData
from computecost import computeCost
from gradientdescent import gradientDescent


def ex1():

    # ==================== Part 1: Basic Function ====================
    print('Running warmUpExercise ... \n')
    print('5x5 Identity Matrix: \n')
    warmUpExercise()

    # ======================= Part 2: Plotting =======================
    print('Plotting Data ...\n')
    ex1data1 = open('ex1data1.txt')
    ex1data1_txt = ex1data1.read()
    data = np.loadtxt(StringIO(
        ex1data1_txt.replace(',', ' ')))
    X = data[:, 0]
    y = data[:, 1].reshape((len(data), 1))

    plotData(X, y)

    # =================== Part 3: Gradient descent ===================
    print('Running Gradient Descent ...\n')
    X_ = np.c_[
        np.ones(len(data[:, 1])),
        data[:, 0]
        ]
    theta = np.zeros((2, 1))

    # Some gradient descent settings
    ITERATIONS = 1500
    ALPHA = 0.01

    # compute and display initial cost
    computeCost(X_, y, theta)

    # run gradient descent
    theta, J_hitory = gradientDescent(X_, y, theta, ALPHA, ITERATIONS)
    print('Theta found by gradient descent: ')
    print('{} {} \n'.format(theta[0], theta[1]))

    # Plot the linear fit

    plt.plot(
        X, y, marker='x', markersize=6, linestyle='None', label='Training data')
    plt.plot(
        X_[:, 1], X_.dot(theta), label='Learning regression')
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.legend()
    plt.show()

    # Predict values for population sizes of 35,000 and 70,000

    predict1 = float(np.array([1, 3.5]).dot(theta))
    print('For population = 35,000, we predict a profit of {}\n'.format(predict1 * 10000))
    predict2 = float(np.array([1, 7]).dot(theta))
    print('For population = 70,000, we predict a profit of {}\n'.format(predict2 * 10000))
    print('Program paused. Press enter to continue.\n')


    # ============= Part 4: Visualizing J(theta_0, theta_1) =============
    print('Visualizing J(theta_0, theta_1) ...\n')

    # Grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    # initialize J_vals to a matrix of 0's
    J_vals = np.zeros(
        (len(theta0_vals), len(theta1_vals)))

    # Fill out J_vals
    for i in range(0, len(theta0_vals)):
        for j in range(0, len(theta1_vals)):
            t = np.array([theta0_vals[i], theta1_vals[j]]).reshape((2, 1))
            J_vals[i][j] = computeCost(X_, y, t)

    # Because of the way meshgrids work in the surf command, we need to
    # transpose J_vals before calling surf, or else the axes will be flipped
    J_vals = J_vals.T
    # Surface plot
    figure = plt.figure()
    axes3d = plt3d.Axes3D(figure)
    meshgrid_x, meshgrid_y = np.meshgrid(
        theta0_vals, theta1_vals)
    axes3d.plot_surface(
        meshgrid_x,  # X
        meshgrid_y,  # Y
        J_vals,  # Z
        rstride=1,  # Array row stride (step size), defaults to 10
        cstride=1,  # Array column stride (step size), defaults to 10
        cmap=cm.cool,  # A colormap for the surface patches.
        linewidth=0,
        antialiased=False)
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    plt.show()

    # Contour plot
    # Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    figure = plt.figure()
    CS = plt.contour(
        meshgrid_x,
        meshgrid_y,
        J_vals,
        50,
        )
    plt.clabel(CS, inline=1, fontsize=10)
    plt.plot(
        theta[0],
        theta[1],
        marker='x',
        markersize=6,
        linestyle='None'
    )
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    plt.show()

if __name__ == '__main__':
    ex1()
