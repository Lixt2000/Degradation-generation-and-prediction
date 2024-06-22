"""

Note: Use original data to generate increment data with non-parameters(time-series)

"""

import numpy as np
import pandas as pd


def increment(data, indices):
    '''
    Calculate differences between specified columns based on given indices.

    Parameters:
        data (numpy.ndarray): (m, n)
        indices (list): 1d

    Returns:
          numpy.ndarray: Array containing differences between 0 and specified columns given by indices
    '''

    indices = [0] + indices

    diff = []
    for i in range(1, len(indices)):
        diff_column = data[:, indices[i]] - data[:, indices[i-1]]
        diff.append(diff_column)

    result = np.column_stack(diff)

    return result


def sample_extend(data, D, constant):
    '''
    Extend the increment data to generate data, and expand the size of data.
    '''
    m, n = data.shape
    extend_array = np.zeros((D, n + 1))

    extend_array[:, 0] = np.random.choice(constant, D)

    for i in range(1, n+1):
        sample_values = np.random.choice(data[:, i-1], D)
        extend_array[:, i] = sample_values

    #extend_array[:, 1:] = np.cumsum(extend_array[:, 1:], axis=1)
    extend_array = np.cumsum(extend_array, axis=1)

    return extend_array


def sample_extend0(data, constant):
    '''
    Extend the increment data to generate data in the same shape with original data.
    '''
    m, n = data.shape
    extend_array = np.zeros((m, n + 1))

    extend_array[:, 0] = constant
    extend_array[:, 1:] = data

    extend_array = np.cumsum(extend_array, axis=1)

    return extend_array


