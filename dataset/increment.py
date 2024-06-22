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

'''
data = np.array([[1.1, 2.4, 3, 4.3, 5],
                 [6.1, 7.2, 8.6, 9, 10]])
indices = [1, 2, 4]
incre_data = increment(data, indices)
'''
'''
data = pd.read_excel('fatigue.xlsx', header=0, index_col=0)
data = data.to_numpy()[1:, :11]
indices = [i for i in range(1, data.shape[1])]
incre_data = increment(data, indices)

print(data)
print(incre_data)

#Maximum of D with different extended data is data.shape[0]^indices.shape[1]
D = 10000

constant_value = data[:, 0]
extend_data = sample_extend(incre_data, D, constant_value)

print(extend_data)
print(np.max(extend_data[:, -1]))
'''