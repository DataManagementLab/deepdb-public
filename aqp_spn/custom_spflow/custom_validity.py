"""
Created on March 20, 2018

@author: Alejandro Molina
"""
import logging

import numpy as np
from math import isclose
from spn.structure.Base import get_nodes_by_type, Product

from aqp_spn.aqp_leaves import Sum, IdentityNumericLeaf

logger = logging.getLogger(__name__)


def is_consistent(node):
    """
    all children of a product node have different scope
    """

    assert node is not None

    allchildscope = set()
    for prod_node in reversed(get_nodes_by_type(node, Product)):
        nscope = set(prod_node.scope)

        if len(prod_node.children) == 0:
            return False, "Product node %s has no children" % prod_node.id

        allchildscope.clear()
        sum_features = 0
        for child in prod_node.children:
            sum_features += len(child.scope)
            allchildscope.update(child.scope)

        if allchildscope != nscope or sum_features != len(allchildscope):
            return False, "children of (prod) node %s do not have exclusive scope" % prod_node.id

    return True, None


def is_complete(node):
    """
    all children of a sum node have same scope as the parent
    """

    assert node is not None

    for sum_node in reversed(get_nodes_by_type(node, Sum)):
        nscope = set(sum_node.scope)

        if len(sum_node.children) == 0:
            return False, "Sum node %s has no children" % sum_node.id

        for child in sum_node.children:
            if nscope != set(child.scope):
                return False, "children of (sum) node %s do not have the same scope as parent" % sum_node.id

    return True, None


def is_valid_prob_sum(prob_sum, unique_vals, card):
    # return True, Null
    length = len(prob_sum) - 1

    if len(prob_sum) != len(unique_vals) + 1:
        return False, "len(prob_sum)!= len(unique_vals)+1"
    last_prob_sum = 0
    cards = []

    sum_card = 0
    for i in range(0, len(prob_sum)):
        if prob_sum[i] > 1.0001:
            return False, "prob_sum[" + str(i) + "] must be =< 1.000, actual value at position " + str(i) + ":" + str(
                prob_sum[i]) + ", len:" + str(len(prob_sum))
        if last_prob_sum - 0.0000001 > prob_sum[i]:
            return False, "prob_sum value must be increase (last_prob_sum:" + str(last_prob_sum) + ", prob_sum[" + str(
                i) + "]:" + str(prob_sum[i])
        num = (prob_sum[i] - last_prob_sum) * card
        if False and not isclose(num, round(num), abs_tol=0.05):
            err_msg = "wrong probability value at idx " + str(i) + " (" + str(
                num) + ")- does not fit to an integer cardinality value for value " + str(unique_vals[i])

            return False, err_msg
        last_prob_sum = prob_sum[i]
        sum_card += round(num)
        cards.append(round(num))

    if not isclose(prob_sum[length], 1, abs_tol=0.05):
        return False, "Last value of prob_sum must be 1.0"
    if sum_card != card:
        return False, "Cardinality of the single values (" + str(
            sum_card) + ") does not match the overall cardinality (" + str(card) + ")"

    return True, None


def is_valid(node, check_ids=True, check_prob_sum=False, light=False):
    #
    if check_ids:
        val, err = has_valid_ids(node)
        if not val:
            return val, err

    for n in get_nodes_by_type(node):
        if len(n.scope) == 0:
            return False, "node %s has no scope" % n.id
        is_sum = isinstance(n, Sum)
        is_prod = isinstance(n, Product)
        is_float = isinstance(n, IdentityNumericLeaf)

        if is_sum:
            if len(n.children) != len(n.weights):
                return False, "node %s has different children/weights" % n.id

            if not light:
                if len(n.children) != len(n.cluster_centers):
                    return False, "node %s has different children/cluster_centers (#cluster_centers: %d, #childs: %d)" % (
                        n.id, len(n.cluster_centers), len(n.children))

            weight_sum = np.sum(n.weights)

            if not isclose(weight_sum, 1, abs_tol=0.05):
                return False, "Sum of weights is not equal 1.0 (instead:" + weight_sum + ")"

        if is_sum or is_prod:
            if len(n.children) == 0:
                return False, "node %s has no children" % n.id

        if is_float:
            ok, err = is_valid_prob_sum(n.prob_sum, n.unique_vals, n.cardinality)
            if not ok:
                return False, err
            if check_prob_sum:
                assert (hasattr(n, 'prob_num')), str(n) + " has no property prob_num"
                assert hasattr((n, 'unique_vals'))
                if len(n.prob_sum) - 1 != len(n.unique_vals):
                    #
                    return False, "size of prob_sum does not match unique_vals (required: prob_sum -1 == unique_vals) "

    a, err = is_consistent(node)
    if not a:
        return a, err

    b, err = is_complete(node)
    if not b:
        return b, err

    return True, None


def has_valid_ids(node):
    ids = set()
    all_nodes = get_nodes_by_type(node)
    for n in all_nodes:
        ids.add(n.id)

    if len(ids) != len(all_nodes):
        return False, "Nodes are missing ids or there are repeated ids"

    if min(ids) != 0:
        return False, "Node ids not starting at 0"

    if max(ids) != len(ids) - 1:
        return False, "Node ids not consecutive"

    return True, None
