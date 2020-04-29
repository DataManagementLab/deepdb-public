import numpy as np
from spn.structure.Base import Leaf
from spn.structure.leaves.parametric.Parametric import Parametric


class Categorical(Parametric):
    """
    Implements a univariate categorical distribution with k parameters
    """

    from spn.structure.StatisticalTypes import Type
    from collections import namedtuple

    type = Type.CATEGORICAL
    property_type = namedtuple("Categorical", "p")

    def __init__(self, p, null_value, scope, cardinality=0):
        Parametric.__init__(self, type(self).type, scope=scope)

        # parameters
        assert np.isclose(np.sum(p), 1), "Probabilities p shall sum to 1"
        if not isinstance(p, np.ndarray):
            p = np.array(p)
        self.p = p
        self.cardinality = cardinality
        self.null_value = null_value

    def copy_node(self):
        return Categorical(np.copy(self.p), self.null_value, self.scope, cardinality=self.cardinality)

    @property
    def parameters(self):
        return __class__.property_type(p=self.p)

    @property
    def k(self):
        return len(self.p)


class IdentityNumericLeaf(Leaf):
    def __init__(self, unique_vals, probabilities, null_value, scope, cardinality=0):
        """
        Instead of histogram remember individual values.
        :param unique_vals: all possible values in leaf
        :param mean: mean of not null values
        :param inverted_mean: inverted mean of not null values
        :param square_mean: mean of squared not null values
        :param inverted_square_mean: mean of 1/squared not null values
        :param prob_sum: cumulative sum of probabilities
        :param null_value_prob: proportion of null values in the leaf
        :param scope:
        """
        Leaf.__init__(self, scope=scope)
        if not isinstance(unique_vals, np.ndarray):
            unique_vals = np.array(unique_vals)
        self.unique_vals = unique_vals
        self.cardinality = cardinality
        self.null_value = null_value
        self.unique_vals_idx = None
        self.update_unique_vals_idx()

        # will be updated later
        self.prob_sum = None
        self.null_value_prob = None
        self.mean = None
        self.inverted_mean = None
        self.square_mean = None
        self.inverted_square_mean = None

        if not isinstance(probabilities, np.ndarray):
            probabilities = np.array(probabilities)
        self.update_from_new_probabilities(probabilities)

    def copy_node(self):
        self_copy = IdentityNumericLeaf(np.copy(self.unique_vals), self.return_histogram(copy=True), self.null_value,
                                        self.scope, cardinality=self.cardinality)
        assert self_copy.mean == self.mean and self_copy.null_value_prob == self.null_value_prob
        assert self_copy.inverted_mean == self.inverted_mean and self_copy.square_mean == self.square_mean
        assert self_copy.inverted_square_mean == self.inverted_square_mean
        return self_copy

    def update_unique_vals_idx(self):
        self.unique_vals_idx = {self.unique_vals[idx]: idx for idx in range(self.unique_vals.shape[0])}

    def return_histogram(self, copy=True):
        if copy:
            return np.copy(self.prob_sum[1:] - self.prob_sum[:-1])
        else:
            return self.prob_sum[1:] - self.prob_sum[:-1]

    def update_from_new_probabilities(self, p):
        assert len(p) == len(self.unique_vals)
        # convert p back to cumulative prob sum
        self.prob_sum = np.concatenate([[0], np.cumsum(p)])
        # update null value prob
        not_null_indexes = np.where(self.unique_vals != self.null_value)[0]
        if self.null_value in self.unique_vals_idx.keys():
            self.null_value_prob = p[self.unique_vals_idx[self.null_value]]
        else:
            self.null_value_prob = 0
        # update not_null idxs
        zero_in_dataset = 0 in self.unique_vals_idx.keys()
        # all values NAN
        if len(not_null_indexes) == 0:
            self.mean = 0
            self.inverted_mean = np.nan

            # for variance computation
            self.square_mean = 0
            self.inverted_square_mean = np.nan
        # some values nan
        else:
            self.mean = np.dot(self.unique_vals[not_null_indexes], p[not_null_indexes]) / (1 - self.null_value_prob)
            self.square_mean = np.dot(np.square(self.unique_vals[not_null_indexes]), p[not_null_indexes]) / (
                    1 - self.null_value_prob)

            if zero_in_dataset:
                self.inverted_mean = np.nan
                self.inverted_square_mean = np.nan
            else:
                self.inverted_mean = np.dot(1 / self.unique_vals[not_null_indexes], p[not_null_indexes]) / (
                        1 - self.null_value_prob)
                self.inverted_square_mean = np.dot(1 / np.square(self.unique_vals[not_null_indexes]),
                                                   p[not_null_indexes]) / (1 - self.null_value_prob)


def _interval_probability(node, left, right, null_value, left_included, right_included):
    if left == -np.inf:
        lower_idx = 0
    else:
        lower_idx = np.searchsorted(node.unique_vals, left, side='left')
        if left == right == node.unique_vals[lower_idx - 1]:
            return node.prob_sum[lower_idx + 1] - node.prob_sum[lower_idx]

    if right == np.inf:
        higher_idx = len(node.unique_vals)
    else:
        higher_idx = np.searchsorted(node.unique_vals, right, side='right')

    if lower_idx == higher_idx:
        return 0

    p = node.prob_sum[higher_idx] - node.prob_sum[lower_idx]
    # null value included in interval
    if null_value is not None and \
            (left < null_value < right or
             null_value == left and left_included or
             null_value == right and right_included):
        p -= node.null_value_prob

    # left value should not be included in interval
    if not left_included and node.unique_vals[lower_idx] == left:
        # equivalent to p -= node.probs[lower_idx]
        p -= node.prob_sum[lower_idx + 1] - node.prob_sum[lower_idx]

    # same check for right value
    if not right_included and node.unique_vals[higher_idx - 1] == right and left != right:
        p -= node.prob_sum[higher_idx] - node.prob_sum[higher_idx - 1]

    return p


def _interval_expectation(power, node, left, right, null_value, left_included, right_included, inverted=False):
    lower_idx = np.searchsorted(node.unique_vals, left, side='left')
    higher_idx = np.searchsorted(node.unique_vals, right, side='right')
    exp = 0

    for j in np.arange(lower_idx, higher_idx):
        if node.unique_vals[j] == null_value:
            continue
        if node.unique_vals[j] == left and not left_included:
            continue
        if node.unique_vals[j] == right and not right_included:
            continue
        p_j = node.prob_sum[j + 1] - node.prob_sum[j]
        if power == 1:
            if not inverted:
                exp += p_j * node.unique_vals[j]
            else:
                exp += p_j * 1 / node.unique_vals[j]
        elif power == 2:
            if not inverted:
                exp += p_j * node.unique_vals[j] * node.unique_vals[j]
            else:
                exp += p_j * 1 / node.unique_vals[j] * 1 / node.unique_vals[j]

    return exp


def identity_expectation(node, data, inverted=False, power=1):
    exps = np.zeros((data.shape[0], 1))
    ranges = data[:, node.scope[0]]

    for i, rang in enumerate(ranges):

        if node.null_value_prob > 0:
            assert rang is not None, "Ensure that features of expectations are not null."

        if rang is None or rang.ranges == [[-np.inf, np.inf]]:
            if power == 1:
                if not inverted:
                    exps[i] = node.mean * (1 - node.null_value_prob)
                else:
                    exps[i] = node.inverted_mean * (1 - node.null_value_prob)
            elif power == 2:
                if not inverted:
                    exps[i] = node.square_mean * (1 - node.null_value_prob)
                else:
                    exps[i] = node.inverted_square_mean * (1 - node.null_value_prob)
            else:
                raise NotImplementedError
            continue

        for k, interval in enumerate(rang.get_ranges()):
            inclusive = rang.inclusive_intervals[k]

            exps[i] += _interval_expectation(power, node, interval[0], interval[1], rang.null_value, inclusive[0],
                                             inclusive[1], inverted=inverted)

    return exps


def _convert_to_single_tuple_set(scope, values):
    if values is None:
        return [scope], None

    return [scope], set(map(lambda x: (x,), values))


def identity_distinct_ranges(node, data, dtype=np.float64, **kwargs):
    """
    Returns distinct values.
    """
    ranges = data[:, node.scope[0]]

    assert len(ranges) == 1, "Only single range is supported"
    if ranges[0] is None:
        return _convert_to_single_tuple_set(node.scope[0], node.unique_vals)

    assert len(ranges[0].ranges) == 1, "Only single interval is supported"

    interval = ranges[0].ranges[0]
    inclusive = ranges[0].inclusive_intervals[0]

    lower_idx = np.searchsorted(node.unique_vals, interval[0], side='left')
    higher_idx = np.searchsorted(node.unique_vals, interval[1], side='right')

    if lower_idx == higher_idx:
        return _convert_to_single_tuple_set(node.scope[0], None)

    if node.unique_vals[lower_idx] == interval[0] and not inclusive[0]:
        lower_idx += 1

    if node.unique_vals[higher_idx - 1] == interval[1] and not inclusive[1]:
        higher_idx -= 1

    if lower_idx == higher_idx:
        return _convert_to_single_tuple_set(node.scope[0], None)

    vals = set(node.unique_vals[lower_idx:higher_idx])
    if ranges[0].null_value in vals:
        vals.remove(ranges[0].null_value)

    return _convert_to_single_tuple_set(node.scope[0], vals)


def identity_likelihood_wo_null(node, data, dtype=np.float64, **kwargs):
    assert len(node.scope) == 1, node.scope

    probs = np.empty((data.shape[0], 1), dtype=dtype)
    probs[:] = np.nan
    nd = data[:, node.scope[0]]

    for i, val in enumerate(nd):
        if not np.isnan(val):
            probs[i] = _interval_probability(node, val, val, None, True, True)

    return probs


def identity_likelihood_range(node, data, dtype=np.float64, overwrite_ranges=None, **kwargs):
    assert len(node.scope) == 1, node.scope

    probs = np.zeros((data.shape[0], 1), dtype=dtype)
    ranges = overwrite_ranges
    if overwrite_ranges is None:
        ranges = data[:, node.scope[0]]

    for i, rang in enumerate(ranges):

        # Skip if no range is specified aka use a log-probability of 0 for that instance
        if rang is None:
            probs[i] = 1
            continue

        if rang.is_not_null_condition:
            probs[i] = 1 - node.null_value_prob
            continue

        # Skip if no values for the range are provided
        if rang.is_impossible():
            continue

        for k, interval in enumerate(rang.get_ranges()):
            inclusive = rang.inclusive_intervals[k]

            probs[i] += _interval_probability(node, interval[0], interval[1], rang.null_value, inclusive[0],
                                              inclusive[1])

    return probs


def categorical_likelihood_wo_null(node, data, dtype=np.float64, **kwargs):
    """
    Returns the likelihood for the given values ignoring NULL values
    """
    probs = np.empty((data.shape[0], 1))
    probs[:] = np.nan
    for i in range(data.shape[0]):
        value = data[i, node.scope[0]]
        if not np.isnan(value):
            probs[i] = node.p[int(value)]
    # probs = np.reshape([node.p[val] for val in data[:, node.scope[0]]],
    #                    (data.shape[0], 1))

    return probs


def categorical_likelihood_range(node, data, dtype=np.float64, **kwargs):
    """
    Returns the probability for the given sets.
    """

    # Assert that the given node is only build on one instance
    assert len(node.scope) == 1, node.scope

    # Initialize the return variable log_probs with zeros
    probs = np.ones((data.shape[0], 1), dtype=dtype)

    # Only select the ranges for the specific feature
    ranges = data[:, node.scope[0]]

    # For each instance
    for i, rang in enumerate(ranges):

        # Skip if no range is specified aka use a log-probability of 0 for that instance
        if rang is None:
            continue

        if rang.is_not_null_condition:
            probs[i] = 1 - node.p[rang.null_value]
            continue

        # Skip if no values for the range are provided
        if len(rang.possible_values) == 0:
            probs[i] = 0

        # Compute the sum of the probability of all possible values
        probs[i] = sum([node.p[possible_val] for possible_val in rang.possible_values])

    return probs


def categorical_distinct_ranges(node, data, dtype=np.float64, **kwargs):
    """
    Returns distinct values.
    """

    ranges = data[:, node.scope[0]]
    assert len(ranges) == 1, "Only single range condition is supported"

    if ranges[0] is None:
        return _convert_to_single_tuple_set(node.scope[0], np.where(node.p > 0)[0])

    return _convert_to_single_tuple_set(node.scope[0],
                                        set(np.where(node.p > 0)[0]).intersection(ranges[0].possible_values))
