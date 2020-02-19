import numpy as np
from spn.structure.Base import Node, Leaf


class Parametric(Leaf):
    def __init__(self, type, scope=None):
        Leaf.__init__(self, scope=scope)
        self._type = type

    @property
    def type(self):
        return self._type


class Sum(Node):
    def __init__(self, weights=None, children=None, cluster_centers=None):

        Node.__init__(self)

        if cluster_centers is None:  # @@
            cluster_centers = []
        self.cluster_centers = cluster_centers  # @@

        if weights is None:
            weights = []
        self.weights = weights

        if children is None:
            children = []
        self.children = children

    @property
    def parameters(self):
        sorted_children = sorted(self.children, key=lambda c: c.id)
        params = [(n.id, self.weights[i]) for i, n in enumerate(sorted_children)]
        return tuple(params)


class Categorical(Parametric):
    """
    Implements a univariate categorical distribution with $k$ parameters
    {\pi_{k}}

    representing the probability of the k-th category

    The conjugate prior for these values would be a Dirichlet

    p(\{\pi_{k}\}) = Dir(\boldsymbol\alpha)
    """

    from spn.structure.StatisticalTypes import Type
    from collections import namedtuple

    type = Type.CATEGORICAL
    property_type = namedtuple("Categorical", "p")

    def __init__(self, p=None, scope=None):
        Parametric.__init__(self, type(self).type, scope=scope)

        # parameters
        if p is not None:
            assert np.isclose(np.sum(p), 1), "Probabilities p shall sum to 1"
        self.p = p

    def updateStatistics(self, dataset, metadata):
        """
        :param dataset: The full dataset which should be included in the statistics
        :param metadata: root node of aqp_spn
        :return:
        """

        idx = self.scope[0]

        value = dataset[idx]
        if metadata.null_values[idx] == value:
            self.cardinality += 1

            return None

        assert (len(self.p) >= value), "Unknown value " + value + ". Maximal value is " + str(len(self.node))
        cardinalities = np.array(self.p) * self.cardinality
        value = int(round(value))

        cardinalities[value] += 1
        self.cardinality += 1
        self.p = cardinalities / self.cardinality

        return True

    @property
    def parameters(self):
        return __class__.property_type(p=self.p)

    @property
    def k(self):
        return len(self.p)


class IdentityNumericLeaf(Leaf):
    def __init__(self, unique_vals, mean, inverted_mean, square_mean, inverted_square_mean, prob_sum, null_value_prob,
                 scope=None):
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
        self.unique_vals = unique_vals

        self.mean = mean
        self.inverted_mean = inverted_mean
        self.square_mean = square_mean
        self.inverted_square_mean = inverted_square_mean
        self.prob_sum = prob_sum
        # ok, err = is_valid_prob_sum(self, self.unique_vals, n.cardinality)
        # assert ok, err
        self.null_value_prob = null_value_prob

    def updateStatistics(self, dataset, metadata):
        """
        The method incrementually updates the mean, inverted_mean, and inverted_square_mean properties and the cardinality
        for the new value dataset[self.scope]

        :param dataset: The full dataset which should be included in the statistics
        :param metadata: root node of aqp_spn
        :return:
        """

        col = self.scope[0]
        number_null_values = round(self.null_value_prob * self.cardinality)
        self.cardinality += 1

        if (dataset[col] is None):
            self.null_value_prob = (self.null_value_prob * (self.cardinality - 1) + 1) / self.cardinality
        else:
            if self.null_value_prob != 0:
                self.null_value_prob = (self.null_value_prob * (self.cardinality - 1)) / self.cardinality

            self.mean = 1 / (self.cardinality - number_null_values) * (
                    dataset[col] + (self.cardinality - number_null_values - 1) * self.mean)

            self.square_mean = 1 / (self.cardinality - number_null_values) * (
                    dataset[col] ** 2 + (self.cardinality - number_null_values - 1) * self.square_mean)

            if dataset[col] != 0:
                self.inverted_mean = 1 / self.cardinality * (
                        (self.cardinality - 1) * self.inverted_mean + 1 / dataset[col])

            if dataset[col] != 0:
                self.inverted_square_mean = 1 / self.cardinality * (
                        (self.cardinality - 1) * self.inverted_square_mean + 1 / (dataset[col] * dataset[col]))

            # prob_sum
            #
            _calculate_new_probability_sum(self, dataset[col])
            _update_context_no_unique_values(metadata, col,
                                             self.unique_vals)

        return True


def _update_context_no_unique_values(metadata, column, unique_values):
    """
    Updates the fields no_unique_values and domains of the root node (ensemble)
    from the spn this node belongs to
    :param node: 
    :return: 
    """

    return None


def _calculate_new_probability_sum(node, value):
    """
    calculates the new probability-sum array.
    :param node:
    :param value_is_new: True, if the value has added to unique_vals
    :param value:
    :return:
    """

    card = node.cardinality - 1

    if not value in node.unique_vals:

        i = 0
        length = len(node.prob_sum)
        while length > i + 1 and node.unique_vals[i] < value:
            i += 1
        idx = i

        if idx > 0:
            node.prob_sum[1:idx + 1] *= card / (card + 1)

        if idx + 1 < length - 1:
            node.prob_sum[idx + 1:length - 1] = node.prob_sum[idx + 1:length - 1] * card / (card + 1) + 1 / (card + 1)

        node.prob_sum = np.insert(node.prob_sum, idx + 1, node.prob_sum[idx] + 1 / (card + 1))

        node.unique_vals = np.insert(node.unique_vals, idx, value)
        length += 1

    else:
        node.prob_sum *= card / (card + 1)

        idx_array = np.where(node.unique_vals == value)
        idx = int(idx_array[0])
        length = len(node.prob_sum)
        if idx + 1 < length - 1:
            node.prob_sum[idx + 1:length] += 1 / (card + 1)

    node.prob_sum[length - 1] = 1.0


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


def identity_likelihood(node, data, dtype=np.float64):
    assert len(node.scope) == 1, node.scope

    probs = np.zeros((data.shape[0], 1), dtype=dtype)
    nd = data[:, node.scope[0]]

    for i, val in enumerate(nd):
        if val is None:
            probs[i] = 1
        elif np.isnan(val):
            probs[i] = 1
        else:
            lower_idx = np.searchsorted(node.unique_vals, val, side='left')
            higher_idx = np.searchsorted(node.unique_vals, val, side='right')
            p = 0

            for j in np.arange(lower_idx, higher_idx):
                p += node.probs[j]

            probs[i] = p

    return probs


def identity_likelihood_range(node, data, dtype=np.float64, **kwargs):
    assert len(node.scope) == 1, node.scope

    probs = np.zeros((data.shape[0], 1), dtype=dtype)
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
