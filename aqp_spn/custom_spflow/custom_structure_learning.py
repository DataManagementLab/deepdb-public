"""
Created on March 20, 2018

@author: Alejandro Molina
"""
import logging
import multiprocessing
import os
from collections import deque
from enum import Enum
from itertools import combinations

import numpy as np
from spn.structure.Base import assign_ids, Product

from aqp_spn.aqp_leaves import Sum
from aqp_spn.custom_spflow.custom_transform_structure import Prune
from aqp_spn.custom_spflow.custom_validity import is_valid
from aqp_spn.custom_spflow.utils import compute_cartesian_product_completeness, default_slicer
from aqp_spn.util.bloom_filter import BloomFilter

logger = logging.getLogger(__name__)

try:
    from time import perf_counter
except:
    from time import time

    perf_counter = time

parallel = True

if parallel:
    cpus = max(1, os.cpu_count() - 2)  # - int(os.getloadavg()[2])
else:
    cpus = 1
pool = multiprocessing.Pool(processes=cpus)


class Operation(Enum):
    CREATE_LEAF = 1
    SPLIT_COLUMNS = 2
    SPLIT_ROWS = 3
    NAIVE_FACTORIZATION = 4
    REMOVE_UNINFORMATIVE_FEATURES = 5
    CONDITIONING = 6


def get_next_operation(min_instances_slice=100, min_features_slice=1, multivariate_leaf=False):
    def next_operation(
            data,
            scope,
            create_leaf,
            no_clusters=False,
            no_independencies=False,
            is_first=False,
            cluster_first=True,
            cluster_univariate=False,
    ):

        minimalFeatures = len(scope) == min_features_slice
        minimalInstances = data.shape[0] <= min_instances_slice

        if minimalFeatures:
            if minimalInstances or no_clusters:
                return Operation.CREATE_LEAF, None
            else:
                if cluster_univariate:
                    return Operation.SPLIT_ROWS, None
                else:
                    return Operation.CREATE_LEAF, None

        uninformative_features_idx = np.var(data[:, 0: len(scope)], 0) == 0
        ncols_zero_variance = np.sum(uninformative_features_idx)
        if ncols_zero_variance > 0:
            if ncols_zero_variance == data.shape[1]:
                if multivariate_leaf:
                    return Operation.CREATE_LEAF, None
                else:
                    return Operation.NAIVE_FACTORIZATION, None
            else:
                return (
                    Operation.REMOVE_UNINFORMATIVE_FEATURES,
                    np.arange(len(scope))[uninformative_features_idx].tolist(),
                )

        if minimalInstances or (no_clusters and no_independencies):
            if multivariate_leaf:
                return Operation.CREATE_LEAF, None
            else:
                return Operation.NAIVE_FACTORIZATION, None

        if no_independencies:
            return Operation.SPLIT_ROWS, None

        if no_clusters:
            return Operation.SPLIT_COLUMNS, None

        if is_first:
            if cluster_first:
                return Operation.SPLIT_ROWS, None
            else:
                return Operation.SPLIT_COLUMNS, None

        return Operation.SPLIT_COLUMNS, None

    return next_operation


def learn_structure(
        bloom_filters,
        dataset,
        ds_context,
        split_rows,
        split_cols,
        create_leaf,
        next_operation=get_next_operation(),
        initial_scope=None,
        data_slicer=default_slicer,
):
    assert dataset is not None
    assert ds_context is not None
    assert split_rows is not None
    assert split_cols is not None
    assert create_leaf is not None
    assert next_operation is not None

    root = Product()
    root.children.append(None)
    root.cardinality = dataset.shape[0]

    if initial_scope is None:
        initial_scope = list(range(dataset.shape[1]))
        num_conditional_cols = None
    elif len(initial_scope) < dataset.shape[1]:
        num_conditional_cols = dataset.shape[1] - len(initial_scope)
    else:
        num_conditional_cols = None
        assert len(initial_scope) > dataset.shape[1], "check initial scope: %s" % initial_scope

    tasks = deque()
    tasks.append((dataset, root, 0, initial_scope, False, False))

    while tasks:

        local_data, parent, children_pos, scope, no_clusters, no_independencies = tasks.popleft()

        operation, op_params = next_operation(
            local_data,
            scope,
            create_leaf,
            no_clusters=no_clusters,
            no_independencies=no_independencies,
            is_first=(parent is root),
        )

        logging.debug("OP: {} on slice {} (remaining tasks {})".format(operation, local_data.shape, len(tasks)))

        if operation == Operation.REMOVE_UNINFORMATIVE_FEATURES:

            node = Product()
            node.cardinality = local_data.shape[0]

            # In this case, no bloom filters are required since the variables which are split are constant. As a
            # consequence, no illegal group by combinations can occur.
            if bloom_filters:
                node.binary_bloom_filters = dict()
            node.scope.extend(scope)
            parent.children[children_pos] = node

            rest_scope = set(range(len(scope)))
            scope_slices = []
            for col in op_params:
                rest_scope.remove(col)
                scope_slices.append([scope[col]])
                node.children.append(None)
                tasks.append(
                    (
                        data_slicer(local_data, [col], num_conditional_cols),
                        node,
                        len(node.children) - 1,
                        [scope[col]],
                        True,
                        True,
                    )
                )
            if len(rest_scope) > 0:
                scope_slices.append([scope[col] for col in rest_scope])

            next_final = False

            if len(rest_scope) == 0:
                continue
            elif len(rest_scope) == 1:
                next_final = True

            node.children.append(None)
            c_pos = len(node.children) - 1

            rest_cols = list(rest_scope)
            rest_scope = [scope[col] for col in rest_scope]
            tasks.append(
                (
                    data_slicer(local_data, rest_cols, num_conditional_cols),
                    node,
                    c_pos,
                    rest_scope,
                    next_final,
                    next_final,
                )
            )

            continue

        elif operation == Operation.SPLIT_ROWS:

            split_start_t = perf_counter()
            data_slices, cluster_centers = split_rows(local_data, ds_context, scope)
            split_end_t = perf_counter()
            logging.debug(
                "\t\tfound {} row clusters (in {:.5f} secs)".format(len(data_slices), split_end_t - split_start_t)
            )

            if len(data_slices) == 1:
                tasks.append((local_data, parent, children_pos, scope, True, False))
                continue

            create_sum_node(children_pos, data_slices, parent, scope, tasks, cluster_centers)

            continue

        elif operation == Operation.SPLIT_COLUMNS:
            split_start_t = perf_counter()
            data_slices = split_cols(local_data, ds_context, scope)
            split_end_t = perf_counter()
            logging.debug(
                "\t\tfound {} col clusters (in {:.5f} secs)".format(len(data_slices), split_end_t - split_start_t)
            )

            if len(data_slices) == 1:
                tasks.append((local_data, parent, children_pos, scope, False, True))
                assert np.shape(data_slices[0][0]) == np.shape(local_data)
                assert data_slices[0][1] == scope
                continue

            node = Product()
            node.cardinality = local_data.shape[0]

            node.scope.extend(scope)
            parent.children[children_pos] = node

            # Create bloom filter for every combination
            scope_slices = [scope_slice for data_slice, scope_slice, _ in data_slices]
            if bloom_filters:
                node.binary_bloom_filters = create_bloom_filters(ds_context, local_data, scope,
                                                                 scope_slices=scope_slices)

            for data_slice, scope_slice, _ in data_slices:
                assert isinstance(scope_slice, list), "slice must be a list"

                node.children.append(None)
                tasks.append((data_slice, node, len(node.children) - 1, scope_slice, False, False))

            continue

        elif operation == Operation.NAIVE_FACTORIZATION:

            node = Product()
            node.cardinality = local_data.shape[0]

            node.scope.extend(scope)
            parent.children[children_pos] = node
            # again check for appearing value combinations
            if bloom_filters:
                node.binary_bloom_filters = create_bloom_filters(ds_context, local_data, scope)

            local_tasks = []
            local_children_params = []
            split_start_t = perf_counter()
            for col in range(len(scope)):
                node.children.append(None)
                local_tasks.append(len(node.children) - 1)
                child_data_slice = data_slicer(local_data, [col], num_conditional_cols)
                local_children_params.append((child_data_slice, ds_context, [scope[col]]))

            result_nodes = pool.starmap(create_leaf, local_children_params)

            for child_pos, child in zip(local_tasks, result_nodes):
                node.children[child_pos] = child

            split_end_t = perf_counter()

            logging.debug(
                "\t\tnaive factorization {} columns (in {:.5f} secs)".format(len(scope), split_end_t - split_start_t)
            )

            continue

        elif operation == Operation.CREATE_LEAF:
            leaf_start_t = perf_counter()
            node = create_leaf(local_data, ds_context, scope)
            parent.children[children_pos] = node
            leaf_end_t = perf_counter()

            logging.debug(
                "\t\t created leaf {} for scope={} (in {:.5f} secs)".format(
                    node.__class__.__name__, scope, leaf_end_t - leaf_start_t
                )
            )

        else:
            raise Exception("Invalid operation: " + operation)

    node = root.children[0]
    assign_ids(node)
    valid, err = is_valid(node)
    assert valid, "invalid spn: " + err
    node = Prune(node)
    valid, err = is_valid(node)
    assert valid, "invalid spn: " + err

    return node


def create_sum_node(children_pos, data_slices, parent, scope, tasks, cluster_centers=None):
    node = Sum()
    node.scope.extend(scope)

    if cluster_centers is not None:
        node.cluster_centers = cluster_centers

    parent.children[children_pos] = node
    # assert parent.scope == node.scope
    cardinality = 0
    for data_slice, scope_slice, proportion in data_slices:
        assert isinstance(scope_slice, list), "slice must be a list"
        cardinality += len(data_slice)
        node.children.append(None)
        node.weights.append(proportion)
        tasks.append((data_slice, node, len(node.children) - 1, scope, False, False))
    node.cardinality = cardinality


def create_bloom_filters(ds_context, local_data, scope, scope_slices=None, target_cart_product_completeness=0.95,
                         min_sample_size=1000, max_sample_size=50000, debug=False):
    """
    Creates required bloom filters for splitted scope of a product node.
    :param ds_context:
    :param local_data:
    :param scope:
    :param scope_slices:
    :param target_cart_product_completeness:
    :param min_sample_size:
    :param max_sample_size:
    :param debug:
    :return:
    """

    def matching_scope(single_scope):
        for slide_ix, scope_slice in enumerate(scope_slices):
            if single_scope in scope_slice:
                return slide_ix
        raise ValueError("No matching slice found.")

    bloom_start_t = perf_counter()
    binary_bloom_filters = dict()
    group_by_scopes = set(scope).intersection(ds_context.group_by_attributes)
    binary_scopes = list(combinations(group_by_scopes, 2))

    for binary_scope in binary_scopes:

        matching_scope_left = 0
        matching_scope_right = 1

        if scope_slices is not None:
            # has to belong to different subtrees
            matching_scope_left = matching_scope(binary_scope[0])
            matching_scope_right = matching_scope(binary_scope[1])

        if matching_scope_left != matching_scope_right:

            col1 = scope.index(binary_scope[0])
            col2 = scope.index(binary_scope[1])

            # first try with smaller sample size
            cartesian_product_completeness, value_combinations_sample, len_cartesian_product = \
                compute_cartesian_product_completeness(col1, col2, ds_context, local_data, min_sample_size,
                                                       max_sample_size, oversampling_cart_product=10, debug=debug)

            if cartesian_product_completeness > target_cart_product_completeness:
                if debug:
                    logging.debug(
                        f"Skipped Bloom filter for binary_scope {binary_scope} because probably full "
                        f"cartesian product appears. (Even in sample {(cartesian_product_completeness * 100):.2f}% were"
                        f" reached")
            else:
                # try again with larger sample
                cartesian_product_completeness, value_combinations, _ = compute_cartesian_product_completeness(
                    col1, col2, ds_context, local_data, min_sample_size, max_sample_size, oversampling_cart_product=100,
                    debug=debug)

                if len(value_combinations) < len_cartesian_product:
                    # bloom = BloomFilter(max_elements=len_cartesian_product, error_rate=0.1)
                    single_bloom_start_t = perf_counter()
                    bloom = BloomFilter()
                    for value_combination in value_combinations:
                        bloom.add(value_combination)
                    binary_bloom_filters[(binary_scope[0], binary_scope[1],)] = bloom
                    single_bloom_end_t = perf_counter()
                    if debug:
                        logging.debug(
                            f"Computed bloom filter for {binary_scope} (combinations: {len(value_combinations)}) "
                            f"in {single_bloom_end_t - single_bloom_start_t} sec.")

                else:
                    if debug:
                        logging.debug(
                            f"Skipped Bloom filter for binary_scope {binary_scope} because full cartesian product "
                            f"appears in larger sample.")

    bloom_end_t = perf_counter()
    logging.debug(f"Created {len(binary_bloom_filters.keys())} bloom filters in {bloom_end_t - bloom_start_t} sec.")

    return binary_bloom_filters
