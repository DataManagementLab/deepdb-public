import numpy as np


def sum_by_group(values, groups):
    order = np.argsort(groups)
    groups = groups[order]
    values = values[order]
    values.cumsum(out=values, axis=0)
    index = np.ones(len(groups), 'bool')
    index[:-1] = groups[1:] != groups[:-1]
    values = values[index]
    groups = groups[index]
    values[1:] = values[1:] - values[:-1]
    return values, groups


def greater_close(a, b):
    return np.greater(a, b) | np.isclose(a, b)
