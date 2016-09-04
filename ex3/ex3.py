#!/usr/local/bin/python
# -*- coding:utf-8 -*-

# Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     lrCostFunction.m (logistic regression cost function)
#     oneVsAll.m
#     predictOneVsAll.m
#     predict.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

import scipy.io
import numpy as np
from displaydata import displayData


def ex3():
    # Setup the parameters you will use
    # for this part of the exercise

    # 20x20 Input Images of Digits
    INPUT_LAYER_SIZE = 400
    # 10 labels, from 1 to 10
    NUM_LABELS = 10
    # (note that we have mapped "0" to label 10)

    # =========== Part 1: Loading and Visualizing Data =============
    #  We start the exercise by first loading and visualizing the dataset.
    #  You will be working with a dataset that contains handwritten digits.

    # Load Training Data
    print('Loading and Visualizing Data ...')

    # training data stored in arrays X, y
    data1_mat = scipy.io.loadmat('ex3data1.mat')
    X = data1_mat['X']  # numpy.ndarray
    y = data1_mat['y']  # numpy.ndarray
    m = X.shape[0]

    # Randomly select 100 data points to display
    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[0:100], :]

    displayData(sel)


if __name__ == '__main__':
    ex3()
