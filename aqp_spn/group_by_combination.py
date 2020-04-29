import copy
import logging

import numpy as np
from spn.algorithms.Inference import likelihood
from spn.structure.Base import get_nodes_by_type, Leaf, Product, eval_spn_bottom_up, assign_ids

from rspn.algorithms.transform_structure import Prune
from rspn.algorithms.validity.validity import is_valid
from rspn.structure.base import Sum

logger = logging.getLogger(__name__)


def prod_group_by(node, children, data=None, dtype=np.float64):
    contains_probs = False
    contains_values = False
    contains_none_values = False
    contains_zero_prob = False
    group_by_scopes = []
    # Check if only probabilities contained
    for child in children:
        # value
        if isinstance(child, tuple):
            contains_values = True

            scope, values = child
            group_by_scopes += scope
            if values is None:
                contains_none_values = True
        # probability
        else:
            contains_probs = True
            if (child == 0).any():
                contains_zero_prob = True

    # Probability of subtree zero or no matching tuples
    if contains_zero_prob or contains_none_values:
        return [None], None
    # Cartesian product
    elif contains_values:
        result_values = None
        group_by_scopes.sort()
        for group_by_scope in group_by_scopes:
            matching_values = None
            matching_idx = None
            for child in children:
                if isinstance(child, tuple):
                    scope, values = child
                    if group_by_scope in scope:
                        matching_values = values
                        matching_idx = scope.index(group_by_scope)
                        break
            assert matching_values is not None, "Matching values should not be None."
            if result_values is None:
                result_values = [(matching_value[matching_idx],) for matching_value in matching_values]
            else:
                result_values = [result_value + (matching_value[matching_idx],) for result_value in result_values for
                                 matching_value in matching_values]
                # assert len(result_values) <= len(group_by_scopes)
        return group_by_scopes, set(result_values)
    # Only probabilities, normal inference
    elif contains_probs:
        llchildren = np.concatenate(children, axis=1)
        return np.nanprod(llchildren, axis=1).reshape(-1, 1)


def sum_group_by(node, children, data=None, dtype=np.float64):
    """
    Propagate expectations in sum node.

    :param node: sum node
    :param children: nodes below
    :param data:
    :param dtype:
    :return:
    """

    # either all tuples or
    if isinstance(children[0], tuple):
        result_values = None
        group_by_scope = [None]
        for scope, values in children:
            if values is not None:
                group_by_scope = scope
                if result_values is None:
                    result_values = values
                else:
                    result_values = result_values.union(values)
        return group_by_scope, result_values

    # normal probability sum node code
    llchildren = np.concatenate(children, axis=1)
    relevant_children_idx = np.where(np.isnan(llchildren[0]) == False)[0]
    if len(relevant_children_idx) == 0:
        return np.array([np.nan])

    assert llchildren.dtype == dtype

    weights_normalizer = sum(node.weights[j] for j in relevant_children_idx)
    b = np.array(node.weights, dtype=dtype)[relevant_children_idx] / weights_normalizer

    return np.dot(llchildren[:, relevant_children_idx], b).reshape(-1, 1)


def group_by_combinations(spn, ds_context, feature_scope, ranges, node_distinct_vals=None, node_likelihoods=None):
    """
    Computes the distinct value combinations for features given the range conditions.
    """
    evidence_scope = set([i for i, r in enumerate(ranges[0]) if r is not None])
    evidence = ranges

    # make feature scope sorted
    feature_scope_unsorted = copy.copy(feature_scope)
    feature_scope.sort()
    # add range conditions to feature scope (makes checking with bloom filters easier)
    feature_scope = list(set(feature_scope)
                         .union(evidence_scope.intersection(np.where(ds_context.no_unique_values <= 1200)[0])))
    feature_scope.sort()
    inverted_order = [feature_scope.index(scope) for scope in feature_scope_unsorted]

    assert not (len(evidence_scope) > 0 and evidence is None)

    relevant_scope = set()
    relevant_scope.update(evidence_scope)
    relevant_scope.update(feature_scope)
    marg_spn = marginalize(spn, relevant_scope)

    def leaf_expectation(node, data, dtype=np.float64, **kwargs):

        if node.scope[0] in feature_scope:
            t_node = type(node)
            if t_node in node_distinct_vals:
                vals = node_distinct_vals[t_node](node, evidence)
                return vals
            else:
                raise Exception('Node type unknown: ' + str(t_node))

        return likelihood(node, evidence, node_likelihood=node_likelihoods)

    node_expectations = {type(leaf): leaf_expectation for leaf in get_nodes_by_type(marg_spn, Leaf)}
    node_expectations.update({Sum: sum_group_by, Product: prod_group_by})

    result = eval_spn_bottom_up(marg_spn, node_expectations, all_results={}, data=evidence, dtype=np.float64)
    if feature_scope_unsorted == feature_scope:
        return result
    scope, grouped_tuples = result
    return feature_scope_unsorted, set(
        [tuple(group_tuple[i] for i in inverted_order) for group_tuple in grouped_tuples])


def marginalize(node, keep, light=False):
    # keep must be a set of features that you want to keep
    # Loc.enter()
    keep = set(keep)

    # Loc.p('keep:', keep)

    def marg_recursive(node):
        # Loc.enter()
        new_node_scope = keep.intersection(set(node.scope))
        # Loc.p("new_node_scope:", new_node_scope)
        if len(new_node_scope) == 0:
            # we are summing out this node
            # Loc.leave(None)
            return None

        if isinstance(node, Leaf):
            if len(node.scope) > 1:
                raise Exception("Leaf Node with |scope| > 1")
            # Loc.leave('Leaf.deepcopy()')
            if light:
                return node
            return copy.deepcopy(node)

        newNode = node.__class__()
        newNode.cardinality = node.cardinality

        if isinstance(node, Sum):
            newNode.weights.extend(node.weights)
            if not light:
                newNode.cluster_centers.extend(node.cluster_centers)

        for c in node.children:
            new_c = marg_recursive(c)
            if new_c is None:
                continue
            newNode.children.append(new_c)

        newNode.scope.extend(new_node_scope)

        # Loc.leave()
        return newNode

    newNode = marg_recursive(node)

    if not light:
        assign_ids(newNode)
        newNode = Prune(newNode, check_cluster_centers=light)

        valid, err = is_valid(newNode, check_cluster_centers=light)
        assert valid, err
    # Loc.leave()
    return newNode
