#!/usr/local/bin/python
# -*- coding:utf-8 -*-
#
# displaydata.py
#

import numpy as np
from PIL import Image


def displayData(X, example_width=None):
    """Display 2D data in a nice grid.

    [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
    stored in X in a nice grid. It returns the figure handle h and the
    displayed array if requested.
    """
    # Set example_width automatically if not passed in
    # one sample image size = (example_width × example_height)
    if not example_width:
        example_width = round(np.sqrt(X.shape[1]))

    # -1 < y < 1 -> 0 < y < 1
    X = X + 1

    # Compute rows, cols
    m = X.shape[0]
    n = X.shape[1]
    example_height = n/example_width

    # Compute number of items to display
    # sample matrix = (display_rows × display_cols)
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    # Between images padding
    padding = 1

    # Setup blank display
    display_array = np.ones(
        (padding + (display_rows * (example_height + padding)),
            padding + (display_cols * (example_width + padding)))
        )

    # Copy each example into a patch on the display array
    current_ex_index = 0
    for j in range(0, display_rows):
        for i in range(0, display_cols):
            if current_ex_index > m:
                break
            # Copy the patch
            # Get the max value of the patch
            max_val = max(np.abs(X[current_ex_index, :]))

            y_offset = padding + j * (example_height + padding)
            x_offset = padding + i * (example_width + padding)
            display_array[
                y_offset:y_offset+example_height,
                x_offset:x_offset+example_width
                ] = np.reshape(
                X[current_ex_index, :],
                (example_height, example_width),
                order='F'
                ) * (250/max_val)
            current_ex_index = current_ex_index + 1
        if current_ex_index > m:
            break

    image = Image.fromarray(np.uint8(display_array))
    image.show()
