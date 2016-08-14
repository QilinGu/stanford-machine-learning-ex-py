#!/usr/local/bin/python
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt


def plotData(X, y, should_call_show=True):
    """Plots the data points X and y into a new figure

    PLOTDATA(x,y) plots the data points with + for the positive examples
    and o for the negative examples. X is assumed to be a Mx2 matrix.

    Instructions: Plot the positive and negative examples on a
                  2D plot, using the option 'k+' for the positive
                  examples and 'ko' for the negative examples.
    """

    pos_indexes = [
        index_pos for index_pos, is_pos in enumerate(y == 1) if is_pos]
    neg_indexes = [
        index_neg for index_neg, is_neg in enumerate(y == 0) if is_neg]

    plt.plot(
        X[pos_indexes, 0],  # X
        X[pos_indexes, 1],  # Y
        marker='+',
        markersize=7,
        linestyle='None',
        label='Admitted')

    plt.plot(
        X[neg_indexes, 0],  # X
        X[neg_indexes, 1],  # Y
        marker='o',
        markersize=7,
        linestyle='None',
        label='Not admitted')

    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()

    if should_call_show:
        plt.show()

    return
