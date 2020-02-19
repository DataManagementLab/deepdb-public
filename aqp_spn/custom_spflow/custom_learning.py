import logging

import numpy as np
from sklearn.cluster import KMeans
from spn.algorithms.splitting.Base import preproc, split_data_by_clusters
from spn.algorithms.splitting.RDC import getIndependentRDCGroups_py
from spn.structure.StatisticalTypes import MetaType

from aqp_spn.aqp_leaves import Categorical
from aqp_spn.aqp_leaves import IdentityNumericLeaf

logger = logging.getLogger(__name__)
MAX_UNIQUE_LEAF_VALUES = 10000


def learn_mspn(
        data,
        ds_context,
        cols="rdc",
        rows="kmeans",
        min_instances_slice=200,
        threshold=0.3,
        max_sampling_threshold_cols=10000,
        max_sampling_threshold_rows=100000,
        bloom_filters=False,
        ohe=False,
        leaves=None,
        memory=None,
        rand_gen=None,
        cpus=-1,
):
    """
    Adapts normal learn_mspn to use custom identity leafs and use sampling for structure learning.
    :param bloom_filters:
    :param max_sampling_threshold_rows:
    :param max_sampling_threshold_cols:
    :param data:
    :param ds_context:
    :param cols:
    :param rows:
    :param min_instances_slice:
    :param threshold:
    :param ohe:
    :param leaves:
    :param memory:
    :param rand_gen:
    :param cpus:
    :return:
    """
    if leaves is None:
        leaves = create_custom_leaf

    if rand_gen is None:
        rand_gen = np.random.RandomState(17)

    from aqp_spn.custom_spflow.custom_structure_learning import get_next_operation, learn_structure

    def l_mspn(data, ds_context, cols, rows, min_instances_slice, threshold, ohe):
        split_cols, split_rows = get_splitting_functions(max_sampling_threshold_rows, max_sampling_threshold_cols, cols,
                                                         rows, ohe, threshold, rand_gen, cpus)

        nextop = get_next_operation(min_instances_slice)

        node = learn_structure(bloom_filters, data, ds_context, split_rows, split_cols, leaves, next_operation=nextop)
        return node

    if memory:
        l_mspn = memory.cache(l_mspn)

    spn = l_mspn(data, ds_context, cols, rows, min_instances_slice, threshold, ohe)
    return spn


def create_custom_leaf(data, ds_context, scope):
    """
    Adapted leafs for cardinality SPN. Either categorical or identityNumeric leafs.
    """

    idx = scope[0]
    meta_type = ds_context.meta_types[idx]

    if meta_type == MetaType.REAL:
        assert len(scope) == 1, "scope for more than one variable?"

        unique_vals, counts = np.unique(data[:, 0], return_counts=True)

        if hasattr(ds_context, 'no_compression_scopes') and idx not in ds_context.no_compression_scopes and \
                len(unique_vals) > MAX_UNIQUE_LEAF_VALUES:
            # if there are too many unique values build identity leaf with histogram representatives
            hist, bin_edges = np.histogram(data[:, 0], bins=MAX_UNIQUE_LEAF_VALUES, density=False)
            logger.debug(f"\t\tDue to histograms leaf size was reduced "
                         f"by {(1 - float(MAX_UNIQUE_LEAF_VALUES) / len(unique_vals)) * 100:.2f}%")
            unique_vals = bin_edges[:-1]
            probs = hist / data.shape[0]
            lidx = len(probs) - 1

            assert len(probs) == len(unique_vals)

        else:
            probs = np.array(counts, np.float64) / len(data[:, 0])
            lidx = len(probs) - 1

        # cumulative sum to make inference faster
        prob_sum = np.concatenate([[0], np.cumsum(probs)])

        null_value = ds_context.null_values[idx]
        zero_in_dataset = data.shape[0] != np.count_nonzero(data[:, 0])

        not_null_indexes = np.where(data[:, 0] != null_value)[0]

        # This version also removes 0 (for inverted (square) mean)
        # not_null_indexes = np.where((data[:, 0] != null_value) & (data[:, 0] != 0.0))[0]

        null_value_prob = 1 - len(not_null_indexes) / len(data[:, 0])

        # all values NAN
        if len(not_null_indexes) == 0:
            mean = 0
            inverted_mean = np.nan

            # for variance computation
            square_mean = 0
            inverted_square_mean = np.nan
        # some values nan
        else:
            mean = np.mean(data[not_null_indexes, 0])
            if zero_in_dataset:
                inverted_mean = np.nan
            else:
                inverted_mean = np.mean(1 / data[not_null_indexes, 0])

            # for variance computation
            square_mean = np.mean(np.square(data[not_null_indexes, 0]))
            if zero_in_dataset:
                inverted_square_mean = np.nan
            else:
                inverted_square_mean = np.mean(1 / np.square(data[not_null_indexes, 0]))

        leaf = IdentityNumericLeaf(unique_vals, mean, inverted_mean, square_mean, inverted_square_mean, prob_sum,
                                   null_value_prob, scope=scope)
        from aqp_spn.custom_spflow.custom_validity import is_valid_prob_sum
        leaf.cardinality = data.shape[0]
        ok, err = is_valid_prob_sum(prob_sum, unique_vals, leaf.cardinality)
        assert ok, err

        return leaf

    elif meta_type == MetaType.DISCRETE:

        unique, counts = np.unique(data[:, 0], return_counts=True)
        # +1 because of potential 0 value that might not occur
        sorted_counts = np.zeros(len(ds_context.domains[idx]) + 1, dtype=np.float64)
        for i, x in enumerate(unique):
            sorted_counts[int(x)] = counts[i]
        p = sorted_counts / data.shape[0]
        node = Categorical(p, scope)
        node.cardinality = data.shape[0]

        return node


def get_splitting_functions(max_sampling_threshold_rows, max_sampling_threshold_cols, cols, rows, ohe, threshold,
                            rand_gen, n_jobs):
    from spn.algorithms.splitting.Clustering import get_split_rows_TSNE, get_split_rows_GMM
    from spn.algorithms.splitting.PoissonStabilityTest import get_split_cols_poisson_py
    from spn.algorithms.splitting.RDC import get_split_rows_RDC_py

    if isinstance(cols, str):

        if cols == "rdc":
            split_cols = get_split_cols_RDC_py(max_sampling_threshold_cols=max_sampling_threshold_cols,
                                               threshold=threshold,
                                               rand_gen=rand_gen, ohe=ohe, n_jobs=n_jobs)
        elif cols == "poisson":
            split_cols = get_split_cols_poisson_py(threshold, n_jobs=n_jobs)
        else:
            raise AssertionError("unknown columns splitting strategy type %s" % str(cols))
    else:
        split_cols = cols

    if isinstance(rows, str):

        if rows == "rdc":
            split_rows = get_split_rows_RDC_py(rand_gen=rand_gen, ohe=ohe, n_jobs=n_jobs)
        elif rows == "kmeans":
            split_rows = get_split_rows_KMeans(max_sampling_threshold_rows=max_sampling_threshold_rows)
        elif rows == "tsne":
            split_rows = get_split_rows_TSNE()
        elif rows == "gmm":
            split_rows = get_split_rows_GMM()
        else:
            raise AssertionError("unknown rows splitting strategy type %s" % str(rows))
    else:
        split_rows = rows
    return split_cols, split_rows


# noinspection PyPep8Naming
def get_split_rows_KMeans(max_sampling_threshold_rows, n_clusters=2, pre_proc=None, ohe=False, seed=17):
    # noinspection PyPep8Naming
    def split_rows_KMeans(local_data, ds_context, scope):
        data = preproc(local_data, ds_context, pre_proc, ohe)

        if data.shape[0] > max_sampling_threshold_rows:
            data_sample = data[np.random.randint(data.shape[0], size=max_sampling_threshold_rows), :]

            kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
            clusters = kmeans.fit(data_sample).predict(data)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
            clusters = kmeans.fit_predict(data)

        cluster_centers = kmeans.cluster_centers_
        result = split_data_by_clusters(local_data, clusters, scope, rows=True)

        return result, cluster_centers.tolist()

    return split_rows_KMeans


# noinspection PyPep8Naming
def get_split_cols_RDC_py(max_sampling_threshold_cols=10000, threshold=0.3, ohe=True, k=10, s=1 / 6,
                          non_linearity=np.sin,
                          n_jobs=-2, rand_gen=None):
    from spn.algorithms.splitting.RDC import split_data_by_clusters

    def split_cols_RDC_py(local_data, ds_context, scope):
        meta_types = ds_context.get_meta_types_by_scope(scope)
        domains = ds_context.get_domains_by_scope(scope)

        if local_data.shape[0] > max_sampling_threshold_cols:
            local_data_sample = local_data[np.random.randint(local_data.shape[0], size=max_sampling_threshold_cols), :]
            clusters = getIndependentRDCGroups_py(
                local_data_sample,
                threshold,
                meta_types,
                domains,
                k=k,
                s=s,
                # ohe=True,
                non_linearity=non_linearity,
                n_jobs=n_jobs,
                rand_gen=rand_gen,
            )
            return split_data_by_clusters(local_data, clusters, scope, rows=False)
        else:
            clusters = getIndependentRDCGroups_py(
                local_data,
                threshold,
                meta_types,
                domains,
                k=k,
                s=s,
                # ohe=True,
                non_linearity=non_linearity,
                n_jobs=n_jobs,
                rand_gen=rand_gen,
            )
            return split_data_by_clusters(local_data, clusters, scope, rows=False)

    return split_cols_RDC_py
