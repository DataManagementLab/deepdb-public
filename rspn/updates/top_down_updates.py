import logging

import numpy as np
from spn.structure.Base import Product

from scipy.spatial import distance

from rspn.structure.base import Sum
from rspn.structure.leaves import Categorical, IdentityNumericLeaf

logger = logging.getLogger(__name__)


def cluster_center_update_dataset(spn, dataset):
    """
    Updates the SPN when a new dataset arrives. The function recursively traverses the
    tree and inserts the different values of a dataset at the according places.

    At every sum node, the child node is selected, based on the minimal euclidian distance to the
    cluster_center of on of the child-nodes.
    :param spn:
    :param dataset:
    :param metadata: root of aqp_spn containing meta-information (ensemble-object)
    :return:
    """

    if isinstance(spn, Categorical):

        insert_into_categorical_leaf(spn, np.array([dataset]), np.array([1.0]))

        return spn
    elif isinstance(spn, IdentityNumericLeaf):

        insert_into_identity_numeric_leaf(spn, np.array([dataset]), np.array([1.0]))

        return spn
    elif isinstance(spn, Sum):
        cc = spn.cluster_centers

        node_idx = 0

        min_dist = np.inf
        min_idx = -1
        for n in spn.children:
            # distance calculation between the dataset and the different clusters
            # (there exist a much faster version on scipy)
            # this? https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
            #
            proj = projection(dataset, n.scope)
            dist = distance.euclidean(cc[node_idx], proj)
            if dist < min_dist:
                min_dist = dist
                min_idx = node_idx

            node_idx += 1
        assert min_idx > -1
        assert min_idx < len(spn.children)
        adapt_weights(spn, min_idx)
        cluster_center_update_dataset(spn.children[min_idx], dataset)
    elif isinstance(spn, Product):

        for n in spn.children:
            cluster_center_update_dataset(n, dataset)
    else:
        raise Exception("Invalid node type " + str(type(spn)))
    spn.cardinality += 1


def projection(dataset, scope):
    projection = []
    for idx in scope:
        assert len(dataset) > idx, "wrong scope " + str(scope) + " for dataset" + str(dataset)
        projection.append(dataset[idx])
    return projection


def adapt_weights(sum_node, selected_child_idx):
    assert isinstance(sum_node, Sum), "adapt_weights called on non Sum-node"

    card = sum_node.cardinality
    from math import isclose
    cardinalities = np.array(sum_node.weights) * card

    cardinalities[selected_child_idx] += 1

    sum_node.weights = cardinalities / (card + 1)
    sum_node.weights = sum_node.weights.tolist()

    weight_sum = np.sum(sum_node.weights)

    assert isclose(weight_sum, 1, abs_tol=0.05)


def insert_into_categorical_leaf(node, data, insert_weights, debug=False):
    """
    Updates categorical leaf according to data with associated insert_weights
    """

    relevant_updates, insert_weights = slice_relevant_updates(node.scope[0], data, insert_weights)
    node.cardinality, node.p = insert_into_histogram(node.p, node.cardinality, relevant_updates, insert_weights,
                                                     debug=debug)


def slice_relevant_updates(idx, data, insert_weights):
    relevant_updates = data[:, idx]
    relevant_idxs = np.where(insert_weights > 0)
    insert_weights = insert_weights[relevant_idxs]
    relevant_updates = relevant_updates[relevant_idxs]

    return relevant_updates, insert_weights


def insert_into_histogram(p, cardinality, relevant_updates_histogram, insert_weights, debug=False):
    p *= cardinality
    for i in range(relevant_updates_histogram.shape[0]):
        p[int(relevant_updates_histogram[i])] += insert_weights[i]

    new_cardinality = max(0, cardinality + np.sum(insert_weights))

    p = np.clip(p, 0, np.inf)

    sum_p = np.sum(p)
    if sum_p > 0:
        p /= sum_p

    if debug:
        assert not np.any(np.isnan(p))
        assert np.isclose(np.sum(p), 1)
        assert new_cardinality >= 0

    return new_cardinality, p


def insert_into_identity_numeric_leaf(node, data, insert_weights, debug=False):
    relevant_updates, insert_weights = slice_relevant_updates(node.scope[0], data, insert_weights)

    p = update_unique_vals(node, relevant_updates)

    # treat this as updating a histogram
    # however, we have plain unique values not idxs of histogram
    relevant_updates_histogram = np.array(
        [node.unique_vals_idx[relevant_updates[i]] for i in range(relevant_updates.shape[0])])

    node.cardinality, p = insert_into_histogram(p, node.cardinality, relevant_updates_histogram, insert_weights,
                                                debug=debug)

    node.update_from_new_probabilities(p)


def update_unique_vals(node, relevant_updates):
    # convert cumulative prob sum back to probabilities
    p = node.prob_sum[1:] - node.prob_sum[:-1]
    # extend the domain if necessary, i.e., all with null value
    update_domain = set([update for update in relevant_updates])
    additional_values = update_domain.difference(node.unique_vals_idx.keys())
    if len(additional_values) > 0:
        # update unique vals
        old_unique_vals = node.unique_vals
        node.unique_vals = np.array(sorted(update_domain.union((node.unique_vals_idx.keys()))))
        node.update_unique_vals_idx()

        # update p's according to new unique_vals
        new_p = np.zeros(len(node.unique_vals))
        for i in range(p.shape[0]):
            new_idx = node.unique_vals_idx[old_unique_vals[i]]
            new_p[new_idx] = p[i]
        p = new_p
    return p
