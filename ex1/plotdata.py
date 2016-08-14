#!/usr/local/bin/python
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt


def plotData(x, y):
    """Plots the data points x and y into a new figure

    PLOTDATA(x,y) plots the data points and gives the figure axes labels of
    population and profit.

    Instructions: Plot the training data into a figure using the
               "figure" and "plot" commands. Set the axes labels using
               the "xlabel" and "ylabel" commands. Assume the
               population and revenue data have been passed in
               as the x and y arguments of this function.

    Hint: You can use the 'rx' option with plot to have the markers
       appear as red crosses. Furthermore, you can make the
       markers larger by using plot(..., 'rx', 'MarkerSize', 10);
    """
    plt.plot(
        x, y, marker='x', markersize=6, linestyle='None')
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.show()
    return
