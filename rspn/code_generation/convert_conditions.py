import numpy as np
from spn.structure.StatisticalTypes import MetaType


def _convert_range(range, pos):
    if range[pos] == -np.inf or range[pos] == np.inf:
        minusInf = True
        condition = 0
    else:
        minusInf = False
        condition = range[pos]
    return minusInf, condition


def _convert_real(idx, condition, inverted_features):
    # method_params += [f'bool inverse{i}', f'bool leftMinusInf{i}', f'float leftCondition{i}',
    #                   f'bool rightMinusInf{i}', f'float rightCondition{i}', f'bool leftIncluded{i}',
    #                   f'bool rightIncluded{i}', f'float nullValue{i}']

    inverse = idx in inverted_features
    if condition is not None:
        leftMinusInf, leftCondition = _convert_range(condition.ranges[0], 0)
        rightMinusInf, rightCondition = _convert_range(condition.ranges[0], 1)
        return inverse, leftMinusInf, leftCondition, rightMinusInf, rightCondition, condition.inclusive_intervals[0][0], \
               condition.inclusive_intervals[0][1], condition.null_value

    return inverse, False, 0, False, 0, False, False, 0


def _convert_categorical(condition):
    # method_params += [f'vector <int> possibleValues{i}', f'int nullValueIdx{i}']

    if condition is not None:
        if condition.is_not_null_condition:
            return condition.possible_values, condition.null_value
        else:
            return condition.possible_values, -1

    # leaves will anyway not be evaluated
    return [0], 0


def convert_range(relevant_scope, featureScope, meta_types, conditions, inverted_features):
    """
    Translates conditions for an expectation method call into parameters that can be passed to generated SPN code.
    :param relevant_scope: relevant_scope from expectation method
    :param featureScope: feature_scope from expectation method
    :param meta_types: types of the columns of the SPN
    :param conditions: conditions to be translated
    :param inverted_features: list indicating which indexes are inverted features (1/x)
    :return: Boolean indicating whether inference is supported by generated SPN. Parameters that have to be passed.
    """
    parameters = (relevant_scope, featureScope)

    for idx, condition in enumerate(conditions):
        if meta_types[idx] == MetaType.DISCRETE:
            parameters += _convert_categorical(condition)
        elif meta_types[idx] == MetaType.REAL:
            # several conditions currently not supported
            if condition is not None and len(condition.ranges) > 1:
                return False, None
            # conditions on feature column currently not supported in C++
            if featureScope[idx] is None:
                return False, None
            parameters += _convert_real(idx, condition, inverted_features)

    return True, parameters