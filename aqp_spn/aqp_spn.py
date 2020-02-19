import copy
import logging
import time
from functools import partial
from time import perf_counter

import numpy as np
from spn.structure.Base import Context, Product
from spn.structure.StatisticalTypes import MetaType

from aqp_spn.aqp_leaves import Categorical
from aqp_spn.aqp_leaves import IdentityNumericLeaf, identity_likelihood_range, identity_expectation, \
    categorical_likelihood_range, identity_distinct_ranges, categorical_distinct_ranges
from aqp_spn.aqp_leaves import Sum
from aqp_spn.custom_spflow.custom_learning import learn_mspn
from aqp_spn.expectations import expectation
from aqp_spn.group_by_combination import group_by_combinations
from aqp_spn.ranges import NominalRange, NumericRange
from ensemble_compilation.spn_ensemble import CombineSPN

logger = logging.getLogger(__name__)


def build_ds_context(column_names, meta_types, null_values, table_meta_data, train_data, group_by_threshold=1200):
    """
    Builds context according to training data.
    :param column_names:
    :param meta_types:
    :param null_values:
    :param table_meta_data:
    :param train_data:
    :return:
    """
    ds_context = Context(meta_types=meta_types)
    ds_context.null_values = null_values
    assert ds_context.null_values is not None, "Null-Values have to be specified"
    domain = []
    no_unique_values = []
    # If metadata is given use this to build domains for categorical values
    unified_column_dictionary = None
    if table_meta_data is not None:
        unified_column_dictionary = {k: v for table, table_md in table_meta_data.items() if
                                     table != 'inverted_columns_dict' and table != 'inverted_fd_dict'
                                     for k, v in table_md['categorical_columns_dict'].items()}

    # domain values
    group_by_attributes = []
    for col in range(train_data.shape[1]):

        feature_meta_type = meta_types[col]
        min_val = np.nanmin(train_data[:, col])
        max_val = np.nanmax(train_data[:, col])

        unique_vals = len(np.unique(train_data[:, col]))
        no_unique_values.append(unique_vals)
        if column_names is not None:
            if unique_vals <= group_by_threshold and 'mul_' not in column_names[col] and '_nn' not in column_names[col]:
                group_by_attributes.append(col)

        domain_values = [min_val, max_val]

        if feature_meta_type == MetaType.REAL:
            min_val = np.nanmin(train_data[:, col])
            max_val = np.nanmax(train_data[:, col])
            domain.append([min_val, max_val])
        elif feature_meta_type == MetaType.DISCRETE:
            # if no metadata is given, infer domains from data
            if column_names is not None \
                    and unified_column_dictionary.get(column_names[col]) is not None:
                no_diff_values = len(unified_column_dictionary[column_names[col]].keys())
                domain.append(np.arange(0, no_diff_values + 1, 1))
            else:
                domain.append(np.arange(domain_values[0], domain_values[1] + 1, 1))
        else:
            raise Exception("Unkown MetaType " + str(meta_types[col]))

    ds_context.domains = np.asanyarray(domain)
    ds_context.no_unique_values = np.asanyarray(no_unique_values)
    ds_context.group_by_attributes = group_by_attributes

    return ds_context


def insert_dataset(spn, dataset, metadata):
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

        spn.updateStatistics(dataset, metadata)

        return spn
    elif isinstance(spn, IdentityNumericLeaf):

        spn.updateStatistics(dataset, metadata)

        return spn
    elif isinstance(spn, Sum):
        cc = spn.cluster_centers

        node_idx = 0

        from scipy.spatial import distance
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
        insert_dataset(spn.children[min_idx], dataset, metadata)
    elif isinstance(spn, Product):

        for n in spn.children:
            proj = projection(dataset, n.scope)
            insert_dataset(n, dataset, metadata)
    else:
        raise Exception("Invalid node type " + str(type(spn)))
    spn.cardinality += 1


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


def projection(dataset, scope):
    projection = []
    for idx in scope:
        assert len(dataset) > idx, "wrong scope " + str(scope) + " for dataset" + str(dataset)
        projection.append(dataset[idx])
    return projection


class AQPSPN(CombineSPN):

    def __init__(self, meta_types, null_values, full_join_size, schema_graph, relationship_list, full_sample_size=None,
                 table_set=None, column_names=None, table_meta_data=None):

        CombineSPN.__init__(self, full_join_size, schema_graph, relationship_list, table_set=table_set)

        self.meta_types = meta_types
        self.null_values = null_values
        self.full_sample_size = full_sample_size
        if full_sample_size is None:
            self.full_sample_size = full_join_size
        self.column_names = column_names
        self.table_meta_data = table_meta_data
        self.mspn = None
        self.ds_context = None

        self.use_generated_code = False

        # training stats
        self.learn_time = None
        self.rdc_threshold = None
        self.min_instances_slice = None

    def add_dataset(self, dataset):
        """
        modifies the SPN, based on new dataset.
        What has to be done?
            - Traverse the tree to find the leave nodes where the values have to be added.
            - Depending on the leaf-nodes, the data in the nodes (mean, unique_vale, p, ...) have to be changed
              + IdentityNumericleaf:
                    unique_vals: add value if not already in there
                    mean:
                    inverted_mean:
                    square_mean:
                    inverted_square_mean:
                    prob_sum:
                    null_value_prob:
               + Categorical:
                    p:

            - The weights of the sum-nodes need to be adjusted
        :param self:
        :param dataset: The new dataset
        :return:
        """

        logging.debug(f"add dataset (incremental)")
        assert len(self.mspn.scope) == len(
            dataset), "dataset has a differnt number of columns as spn. spn expects " + str(
            str(len(self.mspn.scope))) + " columns, but dataset has " + str(len(dataset)) + " columns"

        # from var_dump import var_dump
        # print(var_dump(self.mspn))

        insert_dataset(self.mspn, dataset, self)
        self.full_sample_size += 1
        self.full_join_size += 1

    def learn(self, train_data, rdc_threshold=0.3, min_instances_slice=1, max_sampling_threshold_cols=10000,
              max_sampling_threshold_rows=100000, bloom_filters=False):

        # build domains (including the dependence analysis)
        domain_start_t = time.perf_counter()
        ds_context = build_ds_context(self.column_names, self.meta_types, self.null_values, self.table_meta_data,
                                      train_data)
        self.ds_context = ds_context
        domain_end_t = time.perf_counter()
        logging.debug(f"Built domains in {domain_end_t - domain_start_t} sec")

        # learn mspn
        learn_start_t = time.perf_counter()

        # find scopes for variables which indicate not null column
        if self.column_names is not None:
            ds_context.no_compression_scopes = []
            for table in self.table_set:
                table_obj = self.schema_graph.table_dictionary[table]
                for attribute in table_obj.no_compression:
                    column_name = table + '.' + attribute
                    if column_name in self.column_names:
                        ds_context.no_compression_scopes.append(self.column_names.index(column_name))

        self.mspn = learn_mspn(train_data, ds_context,
                               min_instances_slice=min_instances_slice, threshold=rdc_threshold,
                               max_sampling_threshold_cols=max_sampling_threshold_cols,
                               max_sampling_threshold_rows=max_sampling_threshold_rows,
                               bloom_filters=bloom_filters)
        learn_end_t = time.perf_counter()
        self.learn_time = learn_end_t - learn_start_t
        logging.debug(f"Built SPN in {learn_end_t - learn_start_t} sec")

        # statistics
        self.rdc_threshold = rdc_threshold
        self.min_instances_slice = min_instances_slice

    def learn_incremental(self, data):
        """

        :param data:
        :return:
        """

        logging.info(f"Incremental adding {len(data)} datasets to SPN ...")
        add_ds_start_t = perf_counter()
        i = 0
        for ds in data:
            self.add_dataset(ds)
            if i % 10000 == 0:
                logging.debug(f"\t{i}/{len(data)} ")
            i += 1
        logging.debug(f"{i}/{len(data)} ")
        add_ds_end_t = perf_counter()
        logging.info(
            "f{len(data)} datasets inserted in {(add_ds_end_t - add_ds_start_t)} secs. ({(add_ds_end_t - add_ds_start_t)/len(data)} sec./dataset")

        return self

    # def predict(self, ranges, feature):
    #     assert feature in self.column_names, "Feature not in column names"
    #     regressor_idx = self.column_names.index(feature)
    #
    #     prediction = predict(self.mspn, ranges, regressor_idx)
    #     return prediction

    def evaluate_expectation(self, expectation, standard_deviations=False, gen_code_stats=None):
        """
        Evaluates expectation. Does transformations to map to SPN query compilation.
        :param expectation:
        :return:
        """
        return self.evaluate_expectation_batch(expectation, None, None, standard_deviations=standard_deviations,
                                               gen_code_stats=gen_code_stats)

    def evaluate_indicator_expectation(self, indicator_expectation, standard_deviations=False,
                                       gen_code_stats=None):
        """
        Evaluates indicator expectation.
        :param indicator_expectation:
        :return:
        """
        return self.evaluate_indicator_expectation_batch(indicator_expectation, None, None,
                                                         standard_deviations=standard_deviations,
                                                         gen_code_stats=gen_code_stats)

    def evaluate_expectation_batch(self, expectation, group_bys, group_by_tuples, standard_deviations=False,
                                   impute_p=False, gen_code_stats=None):
        """
        Evaluates a batch of expectations according to different groupings.
        :param expectation:
        :return:
        """

        def postprocess_exps(expectation, exp_values):
            exp_values = np.clip(exp_values, expectation.min_val, np.inf)
            if group_bys is None or group_bys == []:
                return exp_values.item()
            else:
                return exp_values.reshape(len(group_by_tuples), 1)

        features = []
        inverted_features = []
        normalizing_scope = []
        range_conditions = self._parse_conditions(expectation.conditions,
                                                  group_by_columns=group_bys,
                                                  group_by_tuples=group_by_tuples)

        for (table, multiplier) in expectation.features:
            # title.mul_movie_info_idx.movie_id_nn
            features.append(self.column_names.index(table + '.' + multiplier))
            inverted_features.append(False)

        for (table, multiplier) in expectation.normalizing_multipliers:
            # title.mul_movie_info_idx.movie_id_nn
            index = self.column_names.index(table + '.' + multiplier)
            features.append(index)
            normalizing_scope.append(index)
            inverted_features.append(True)

        std_values, exp_values = \
            self._normalized_conditional_expectation(features, inverted_features=inverted_features,
                                                     normalizing_scope=normalizing_scope,
                                                     range_conditions=range_conditions,
                                                     standard_deviations=standard_deviations,
                                                     impute_p=impute_p,
                                                     gen_code_stats=gen_code_stats)
        if standard_deviations:
            if group_bys is None or group_bys == []:
                std_values = std_values.item()
            else:
                std_values = std_values.reshape(len(group_by_tuples), 1)

        return std_values, postprocess_exps(expectation, exp_values)

    def evaluate_indicator_expectation_batch(self, indicator_expectation, group_bys, group_by_tuples,
                                             standard_deviations=False, gen_code_stats=None):
        """
        Evaluates a batch of indicator expectations according to different groupings.
        :param indicator_expectation:
        :param group_bys:
        :param result_tuples:
        :param result_tuples_translated:
        :return:
        """

        def isclosetozero(exp_values):
            isclose = np.isclose(exp_values, 0)
            if isinstance(isclose, bool):
                return isclose
            return all(isclose)

        def postprocess_exps(indicator_expectation, features, exp_values, std_values):
            # if indicator expectation has zero probability, split up
            if isclosetozero(exp_values) and \
                    indicator_expectation.nominator_multipliers is not None \
                    and len(indicator_expectation.nominator_multipliers) > 0:
                # average expectation of multipliers, ignore denominator multipliers here since they belong to
                # the probability part
                exp_values = np.ones(exp_values.shape)
                if standard_deviations:
                    exp_values *= self._unnormalized_conditional_expectation(features)
                else:
                    std, exp = self._unnormalized_conditional_expectation_with_std(features)
                    std_values = np.ones(exp_values.shape) * std
                    exp_values *= exp

                # min probability
                if std_values is not None:
                    std_values *= indicator_expectation.min_val
                exp_values *= indicator_expectation.min_val

            if std_values is not None:
                if group_bys is None:
                    std_values = std_values.item()
                else:
                    std_values = std_values.reshape(len(group_by_tuples), 1)

            exp_values = np.clip(exp_values, indicator_expectation.min_val, np.inf)
            if indicator_expectation.inverse:
                exp_values = 1 / exp_values
            if group_bys is None:
                return std_values, exp_values.item()
            else:
                return std_values, exp_values.reshape(len(group_by_tuples), 1)

        # multipliers present, use indicator expectation
        features = []
        inverted_features = []
        range_conditions = self._parse_conditions(indicator_expectation.conditions, group_by_columns=group_bys,
                                                  group_by_tuples=group_by_tuples)
        for (table, multiplier) in indicator_expectation.nominator_multipliers:
            # title.mul_movie_info_idx.movie_id_nn
            features.append(self.column_names.index(table + '.' + multiplier))
            inverted_features.append(False)
        if indicator_expectation.denominator_multipliers is not None:
            for (table, multiplier) in indicator_expectation.denominator_multipliers:
                features.append(self.column_names.index(table + '.' + multiplier))
                inverted_features.append(True)
        if standard_deviations:
            std_values, exp_values = self._indicator_expectation_with_std(features, inverted_features=inverted_features,
                                                                          range_conditions=range_conditions)

            return postprocess_exps(indicator_expectation, features, exp_values, std_values)

        exp_values = self._indicator_expectation(features, inverted_features=inverted_features,
                                                 range_conditions=range_conditions, gen_code_stats=gen_code_stats)
        return postprocess_exps(indicator_expectation, features, exp_values, None)

    def evaluate_group_by_combinations(self, features, range_conditions=None):
        if range_conditions is not None:
            range_conditions = self._parse_conditions(range_conditions)
        feature_scope = []
        replaced_features = []
        # check if group by attribute is in relevant attributes, could also be omitted because of FD redundancy
        for feature in features:
            if feature in self.column_names:
                feature_scope.append(self.column_names.index(feature))
                replaced_features.append((feature,))
            elif any([feature in self.table_meta_data[table]['fd_dict'].keys() for table in self.table_set]):
                def find_ancestor(grouping_feature, lineage=None):
                    if lineage is None:
                        lineage = tuple()
                    lineage = (grouping_feature,) + lineage

                    if grouping_feature in self.column_names:
                        return grouping_feature, lineage

                    table = grouping_feature.split('.', 1)[0]
                    source_attributes = self.table_meta_data[table]['fd_dict'][grouping_feature].keys()
                    if len(source_attributes) > 1:
                        logger.warning(f"Current functional dependency handling is not designed for attributes with "
                                       f"more than one ancestor such as {grouping_feature}. This can lead to error in "
                                       f"further processing.")
                    grouping_source_attribute = list(source_attributes)[0]

                    return find_ancestor(grouping_source_attribute, lineage=lineage)

                # another attribute that is FD ancestor of group by attribute we are interested in
                source_attribute, lineage = find_ancestor(feature)
                feature_scope.append(self.column_names.index(source_attribute))
                replaced_features.append(lineage)

        scope, group_bys = self._group_by_combinations(copy.copy(feature_scope), range_conditions=range_conditions)
        group_bys = list(group_bys)
        group_bys_translated = group_bys

        # replace by alternative, i.e. replace by real categorical values or replace by categorical value of top
        # fd attribute
        if any([feature in self.table_meta_data['inverted_columns_dict'].keys() for feature in features]) or \
                any([any([feature in self.table_meta_data[table]['fd_dict'].keys() for feature in features])
                     for table in self.table_meta_data.keys() if
                     table != 'inverted_columns_dict' and table != 'inverted_fd_dict']):
            def replace_all_columns(result_tuple):
                new_result_tuple = tuple()
                for idx, feature in enumerate(features):
                    lineage = replaced_features[idx]
                    # categorical column
                    if feature in self.table_meta_data['inverted_columns_dict'].keys():
                        replaced_value = self.table_meta_data['inverted_columns_dict'][feature][result_tuple[idx]]
                        new_result_tuple += (replaced_value,)
                    # categorical column with top fd attribute
                    elif feature != lineage[0]:
                        # e.g. #MFRG_1111
                        replaced_value = self.table_meta_data['inverted_columns_dict'][lineage[0]][
                            result_tuple[idx]]
                        for idx, attribute in enumerate(lineage):
                            if idx == 0:
                                continue
                            # e.g. #MFRG_1111 > #MFRG_11 > #MFRG_1
                            replaced_value = self.table_meta_data['inverted_fd_dict'][lineage[idx - 1]][lineage[idx]][
                                replaced_value]
                        new_result_tuple += (replaced_value,)
                    else:
                        new_result_tuple += (result_tuple[idx],)
                return new_result_tuple

            group_bys_translated = list(map(replace_all_columns, group_bys))

        # unique group bys
        unique_group_bys = {k: None for k in list(set(group_bys_translated))}

        # combine group bys with equal group_by_translated columns
        for i, group_by in enumerate(group_bys):
            group_by_translated = group_bys_translated[i]
            if unique_group_bys[group_by_translated] is None:
                unique_group_bys[group_by_translated] = group_by
            else:
                # already existing, merge
                current_tuple = list(unique_group_bys[group_by_translated])
                for j, feature in enumerate(group_by):
                    if isinstance(current_tuple[j], list):
                        if feature not in current_tuple[j]:
                            current_tuple[j].append(feature)
                    else:
                        if feature != current_tuple[j]:
                            current_tuple[j] = [current_tuple[j]]
                            current_tuple[j].append(feature)
                unique_group_bys[group_by_translated] = tuple(current_tuple)

        group_bys_translated = list(unique_group_bys.keys())
        group_bys = [unique_group_bys[k] for k in group_bys_translated]

        return [self.column_names[feature] for feature in feature_scope], group_bys, group_bys_translated

    def _group_by_combinations(self, feature_scope, range_conditions=None):
        """
        Computes all value combinations of features given the range_conditions
        :param feature_scope: array of features
        :param range_conditions:  e.g. np.array([NominalRange([0]), NumericRange([[0,0.3]]), None])
        """

        if range_conditions is None:
            range_conditions = np.array([None] * len(self.mspn.scope)).reshape(1, len(self.mspn.scope))
        else:
            range_conditions = self._add_null_values_to_ranges(range_conditions)

        range_conditions = self._augment_not_null_conditions(feature_scope, range_conditions)
        range_conditions = self._add_null_values_to_ranges(range_conditions)

        _node_distinct_values = {IdentityNumericLeaf: identity_distinct_ranges,
                                 Categorical: categorical_distinct_ranges}
        _node_likelihoods_range = {IdentityNumericLeaf: identity_likelihood_range,
                                   Categorical: categorical_likelihood_range}

        return group_by_combinations(self.mspn, self.ds_context, feature_scope, range_conditions,
                                     node_distinct_vals=_node_distinct_values, node_likelihoods=_node_likelihoods_range)

    def _add_null_values_to_ranges(self, range_conditions):
        for col in range(range_conditions.shape[1]):
            if range_conditions[0][col] is None:
                continue
            for idx in range(range_conditions.shape[0]):
                range_conditions[idx, col].null_value = self.null_values[col]

        return range_conditions

    def _probability(self, range_conditions):
        """
        Compute probability of range conditions.

        e.g. np.array([NominalRange([0]), NumericRange([[0,0.3]]), None])
        """

        return self._indicator_expectation([], range_conditions=range_conditions)

    def _indicator_expectation(self, feature_scope, identity_leaf_expectation=None, inverted_features=None,
                               range_conditions=None, force_no_generated=False, gen_code_stats=None):
        """
        Compute E[1_{conditions} * X_feature_scope]. Can also compute products (specify multiple feature scopes).
        For inverted features 1/val is used.

        Is basis for both unnormalized and normalized expectation computation.

        Uses safe evaluation for products, i.e. compute extra expectation for every multiplier. If results deviate too
        largely (>max_deviation), replace by mean. We also experimented with splitting the multipliers into 10 random
        groups. However, this works equally well and is faster.
        """

        if inverted_features is None:
            inverted_features = [False] * len(feature_scope)

        if range_conditions is None:
            range_conditions = np.array([None] * len(self.mspn.scope)).reshape(1, len(self.mspn.scope))
        else:
            range_conditions = self._add_null_values_to_ranges(range_conditions)

        if identity_leaf_expectation is None:
            _node_expectation = {IdentityNumericLeaf: identity_expectation}
        else:
            _node_expectation = {IdentityNumericLeaf: identity_leaf_expectation}

        _node_likelihoods_range = {IdentityNumericLeaf: identity_likelihood_range,
                                   Categorical: categorical_likelihood_range}

        if hasattr(self, 'use_generated_code') and self.use_generated_code and not force_no_generated:
            full_result = expectation(self.mspn, feature_scope, inverted_features, range_conditions,
                                      node_expectation=_node_expectation, node_likelihoods=_node_likelihoods_range,
                                      use_generated_code=True, spn_id=self.id, meta_types=self.meta_types,
                                      gen_code_stats=gen_code_stats)
        else:
            full_result = expectation(self.mspn, feature_scope, inverted_features, range_conditions,
                                      node_expectation=_node_expectation, node_likelihoods=_node_likelihoods_range)

        return full_result

    def _augment_not_null_conditions(self, feature_scope, range_conditions):
        if range_conditions is None:
            range_conditions = np.array([None] * len(self.mspn.scope)).reshape(1, len(self.mspn.scope))

        # for second computation make sure that features that are not normalized are not NULL
        for not_null_scope in feature_scope:
            if self.null_values[not_null_scope] is None:
                continue

            if range_conditions[0, not_null_scope] is None:
                if self.meta_types[not_null_scope] == MetaType.REAL:
                    range_conditions[:, not_null_scope] = NumericRange([[-np.inf, np.inf]], is_not_null_condition=True)
                elif self.meta_types[not_null_scope] == MetaType.DISCRETE:
                    NumericRange([[-np.inf, np.inf]], is_not_null_condition=True)
                    categorical_feature_name = self.column_names[not_null_scope]

                    for table in self.table_meta_data.keys():
                        categorical_values = self.table_meta_data[table]['categorical_columns_dict'] \
                            .get(categorical_feature_name)
                        if categorical_values is not None:
                            possible_values = list(categorical_values.values())
                            possible_values.remove(self.null_values[not_null_scope])
                            range_conditions[:, not_null_scope] = NominalRange(possible_values,
                                                                               is_not_null_condition=True)
                            break

        return range_conditions

    def _indicator_expectation_with_std(self, feature_scope, inverted_features=None,
                                        range_conditions=None):
        """
        Computes standard deviation of the estimator for 1_{conditions}*X_feature_scope. Uses the identity
        V(X)=E(X^2)-E(X)^2.
        :return:
        """
        e_x = self._indicator_expectation(feature_scope, identity_leaf_expectation=identity_expectation,
                                          inverted_features=inverted_features,
                                          range_conditions=range_conditions)

        not_null_conditions = self._augment_not_null_conditions(feature_scope, None)
        n = self._probability(not_null_conditions) * self.full_sample_size

        # shortcut: use binomial std if it is just a probability
        if len(feature_scope) == 0:
            std = np.sqrt(e_x * (1 - e_x) * 1 / n)
            return std, e_x

        e_x_sq = self._indicator_expectation(feature_scope,
                                             identity_leaf_expectation=partial(identity_expectation, power=2),
                                             inverted_features=inverted_features,
                                             range_conditions=range_conditions,
                                             force_no_generated=True)

        v_x = e_x_sq - e_x * e_x

        # Indeed divide by sample size of SPN not only qualifying tuples. Because this is not a conditional expectation.
        std = np.sqrt(v_x / n)

        return std, e_x

    def _unnormalized_conditional_expectation(self, feature_scope, inverted_features=None, range_conditions=None,
                                              impute_p=False, gen_code_stats=None):
        """
        Compute conditional expectation. Can also compute products (specify multiple feature scopes).
        For inverted features 1/val is used. Normalization is not possible here.
        """

        range_conditions = self._augment_not_null_conditions(feature_scope, range_conditions)
        unnormalized_exp = self._indicator_expectation(feature_scope, inverted_features=inverted_features,
                                                       range_conditions=range_conditions, gen_code_stats=gen_code_stats)

        p = self._probability(range_conditions)
        if any(p == 0):
            if impute_p:
                impute_val = np.mean(
                    unnormalized_exp[np.where(p != 0)[0]] / p[np.where(p != 0)[0]])
                result = unnormalized_exp / p
                result[np.where(p == 0)[0]] = impute_val
                return result

            return self._indicator_expectation(feature_scope, inverted_features=inverted_features,
                                               gen_code_stats=gen_code_stats)

        return unnormalized_exp / p

    def _unnormalized_conditional_expectation_with_std(self, feature_scope, inverted_features=None,
                                                       range_conditions=None, gen_code_stats=None):
        """
        Compute conditional expectation. Can also compute products (specify multiple feature scopes).
        For inverted features 1/val is used. Normalization is not possible here.
        """
        range_conditions = self._augment_not_null_conditions(feature_scope, range_conditions)
        p = self._probability(range_conditions)

        e_x_sq = self._indicator_expectation(feature_scope,
                                             identity_leaf_expectation=partial(identity_expectation, power=2),
                                             inverted_features=inverted_features,
                                             range_conditions=range_conditions,
                                             force_no_generated=True,
                                             gen_code_stats=gen_code_stats) / p

        e_x = self._indicator_expectation(feature_scope, inverted_features=inverted_features,
                                          range_conditions=range_conditions,
                                          gen_code_stats=gen_code_stats) / p

        v_x = e_x_sq - e_x * e_x

        n = p * self.full_sample_size
        std = np.sqrt(v_x / n)

        return std, e_x

    def _normalized_conditional_expectation(self, feature_scope, inverted_features=None, normalizing_scope=None,
                                            range_conditions=None, standard_deviations=False, impute_p=False,
                                            gen_code_stats=None):
        """
        Computes unbiased estimate for conditional expectation E(feature_scope| range_conditions).
        To this end, normalization might be required (will always be certain multipliers.)
        E[1_{conditions} * X_feature_scope] / E[1_{conditions} * X_normalizing_scope]

        :param feature_scope:
        :param inverted_features:
        :param normalizing_scope:
        :param range_conditions:
        :return:
        """
        if range_conditions is None:
            range_conditions = np.array([None] * len(self.mspn.scope)).reshape(1, len(self.mspn.scope))

        # If normalization is not required, simply return unnormalized conditional expectation
        if normalizing_scope is None or len(normalizing_scope) == 0:
            if standard_deviations:
                return self._unnormalized_conditional_expectation_with_std(feature_scope,
                                                                           inverted_features=inverted_features,
                                                                           range_conditions=range_conditions,
                                                                           gen_code_stats=gen_code_stats)
            else:
                return None, self._unnormalized_conditional_expectation(feature_scope,
                                                                        inverted_features=inverted_features,
                                                                        range_conditions=range_conditions,
                                                                        impute_p=impute_p,
                                                                        gen_code_stats=gen_code_stats)

        assert set(normalizing_scope).issubset(feature_scope), "Normalizing scope must be subset of feature scope"

        # for computation make sure that features that are not normalized are not NULL
        range_conditions = self._augment_not_null_conditions(set(feature_scope).difference(normalizing_scope),
                                                             range_conditions)

        # E[1_{conditions} * X_feature_scope]
        std = None
        if standard_deviations:
            std, _ = self._unnormalized_conditional_expectation_with_std(feature_scope,
                                                                         inverted_features=inverted_features,
                                                                         range_conditions=range_conditions,
                                                                         gen_code_stats=gen_code_stats)

        nominator = self._indicator_expectation(feature_scope,
                                                inverted_features=inverted_features,
                                                range_conditions=range_conditions,
                                                gen_code_stats=gen_code_stats)

        # E[1_{conditions} * X_normalizing_scope]
        inverted_features_of_norm = \
            [inverted_features[feature_scope.index(variable_scope)] for variable_scope in normalizing_scope]
        assert all(inverted_features_of_norm), "Normalizing factors should be inverted"

        denominator = self._indicator_expectation(normalizing_scope,
                                                  inverted_features=inverted_features_of_norm,
                                                  range_conditions=range_conditions)
        return std, nominator / denominator

    def _parse_conditions(self, conditions, group_by_columns=None, group_by_tuples=None):
        """
        Translates string conditions to NumericRange and NominalRanges the SPN understands.
        """
        assert self.column_names is not None, "For probability evaluation column names have to be provided."
        group_by_columns_merged = None
        if group_by_columns is None or group_by_columns == []:
            ranges = np.array([None] * len(self.column_names)).reshape(1, len(self.column_names))
        else:
            ranges = np.array([[None] * len(self.column_names)] * len(group_by_tuples))
            group_by_columns_merged = [table + '.' + attribute for table, attribute in group_by_columns]

        for (table, condition) in conditions:

            table_obj = self.schema_graph.table_dictionary[table]

            # is an nn attribute condition
            if table_obj.table_nn_attribute in condition:
                full_nn_attribute_name = table + '.' + table_obj.table_nn_attribute
                # unnecessary because column is never NULL
                if full_nn_attribute_name not in self.column_names:
                    continue
                # column can become NULL
                elif condition == table_obj.table_nn_attribute + ' IS NOT NULL':
                    attribute_index = self.column_names.index(full_nn_attribute_name)
                    ranges[:, attribute_index] = NominalRange([1])
                    continue
                elif condition == table_obj.table_nn_attribute + ' IS NULL':
                    attribute_index = self.column_names.index(full_nn_attribute_name)
                    ranges[:, attribute_index] = NominalRange([0])
                    continue
                else:
                    raise NotImplementedError

            # for other attributes parse. Find matching attr.
            matching_fd_cols = [column for column in list(self.table_meta_data[table]['fd_dict'].keys())
                                if column + '<' in table + '.' + condition or column + '=' in table + '.' + condition
                                or column + '>' in table + '.' + condition or column + ' ' in table + '.' + condition]
            matching_cols = [column for column in self.column_names if column + '<' in table + '.' + condition or
                             column + '=' in table + '.' + condition or column + '>' in table + '.' + condition
                             or column + ' ' in table + '.' + condition]
            assert len(matching_cols) == 1 or len(matching_fd_cols) == 1, "Found multiple or no matching columns"
            if len(matching_cols) == 1:
                matching_column = matching_cols[0]

            elif len(matching_fd_cols) == 1:
                matching_fd_column = matching_fd_cols[0]

                def find_recursive_values(column, dest_values):
                    source_attribute, dictionary = list(self.table_meta_data[table]['fd_dict'][column].items())[0]
                    if len(self.table_meta_data[table]['fd_dict'][column].keys()) > 1:
                        logger.warning(f"Current functional dependency handling is not designed for attributes with "
                                       f"more than one ancestor such as {column}. This can lead to error in further "
                                       f"processing.")
                    source_values = []
                    for dest_value in dest_values:
                        if not isinstance(list(dictionary.keys())[0], str):
                            dest_value = float(dest_value)
                        source_values += dictionary[dest_value]

                    if source_attribute in self.column_names:
                        return source_attribute, source_values
                    return find_recursive_values(source_attribute, source_values)

                if '=' in condition:
                    _, literal = condition.split('=', 1)
                    literal_list = [literal.strip(' "\'')]
                elif 'NOT IN' in condition:
                    literal_list = _literal_list(condition)
                elif 'IN' in condition:
                    literal_list = _literal_list(condition)

                matching_column, values = find_recursive_values(matching_fd_column, literal_list)
                attribute_index = self.column_names.index(matching_column)

                if self.meta_types[attribute_index] == MetaType.DISCRETE:
                    condition = matching_column + 'IN ('
                    for i, value in enumerate(values):
                        condition += '"' + value + '"'
                        if i < len(values) - 1:
                            condition += ','
                    condition += ')'
                else:
                    min_value = min(values)
                    max_value = max(values)
                    if values == list(range(min_value, max_value + 1)):
                        ranges = _adapt_ranges(attribute_index, max_value, ranges, inclusive=True, lower_than=True)
                        ranges = _adapt_ranges(attribute_index, min_value, ranges, inclusive=True, lower_than=False)
                        continue
                    else:
                        raise NotImplementedError

            attribute_index = self.column_names.index(matching_column)

            if self.meta_types[attribute_index] == MetaType.DISCRETE:

                val_dict = self.table_meta_data[table]['categorical_columns_dict'][matching_column]

                if '=' in condition:
                    column, literal = condition.split('=', 1)
                    literal = literal.strip(' "\'')

                    if group_by_columns_merged is None or matching_column not in group_by_columns_merged:
                        ranges[:, attribute_index] = NominalRange([val_dict[literal]])
                    else:
                        matching_group_by_idx = group_by_columns_merged.index(matching_column)
                        # due to functional dependencies this check does not make sense any more
                        # assert val_dict[literal] == group_by_tuples[0][matching_group_by_idx]
                        for idx in range(len(ranges)):
                            literal = group_by_tuples[idx][matching_group_by_idx]
                            ranges[idx, attribute_index] = NominalRange([literal])

                elif 'NOT IN' in condition:
                    literal_list = _literal_list(condition)
                    single_range = NominalRange(
                        [val_dict[literal] for literal in val_dict.keys() if not literal in literal_list])
                    if self.null_values[attribute_index] in single_range.possible_values:
                        single_range.possible_values.remove(self.null_values[attribute_index])
                    if all([single_range is None for single_range in ranges[:, attribute_index]]):
                        ranges[:, attribute_index] = single_range
                    else:
                        for i, nominal_range in enumerate(ranges[:, attribute_index]):
                            ranges[i, attribute_index] = NominalRange(
                                list(set(nominal_range.possible_values).intersection(single_range.possible_values)))

                elif 'IN' in condition:
                    literal_list = _literal_list(condition)
                    single_range = NominalRange([val_dict[literal] for literal in literal_list])
                    if all([single_range is None for single_range in ranges[:, attribute_index]]):
                        ranges[:, attribute_index] = single_range
                    else:
                        for i, nominal_range in enumerate(ranges[:, attribute_index]):
                            ranges[i, attribute_index] = NominalRange(list(
                                set(nominal_range.possible_values).intersection(single_range.possible_values)))

            elif self.meta_types[attribute_index] == MetaType.REAL:
                if '<=' in condition:
                    _, literal = condition.split('<=', 1)
                    literal = float(literal.strip())
                    ranges = _adapt_ranges(attribute_index, literal, ranges, inclusive=True, lower_than=True)

                elif '>=' in condition:
                    _, literal = condition.split('>=', 1)
                    literal = float(literal.strip())
                    ranges = _adapt_ranges(attribute_index, literal, ranges, inclusive=True, lower_than=False)
                elif '=' in condition:
                    _, literal = condition.split('=', 1)
                    literal = float(literal.strip())

                    def non_conflicting(single_numeric_range):
                        assert single_numeric_range[attribute_index] is None or \
                               (single_numeric_range[attribute_index][0][0] > literal or
                                single_numeric_range[attribute_index][0][1] < literal), "Value range does not " \
                                                                                        "contain any values"

                    map(non_conflicting, ranges)
                    if group_by_columns_merged is None or matching_column not in group_by_columns_merged:
                        ranges[:, attribute_index] = NumericRange([[literal, literal]])
                    else:
                        matching_group_by_idx = group_by_columns_merged.index(matching_column)
                        assert literal == group_by_tuples[0][matching_group_by_idx]
                        for idx in range(len(ranges)):
                            literal = group_by_tuples[idx][matching_group_by_idx]
                            ranges[idx, attribute_index] = NumericRange([[literal, literal]])

                elif '<' in condition:
                    _, literal = condition.split('<', 1)
                    literal = float(literal.strip())
                    ranges = _adapt_ranges(attribute_index, literal, ranges, inclusive=False, lower_than=True)
                elif '>' in condition:
                    _, literal = condition.split('>', 1)
                    literal = float(literal.strip())
                    ranges = _adapt_ranges(attribute_index, literal, ranges, inclusive=False, lower_than=False)
                else:
                    raise ValueError("Unknown operator")

                def is_invalid_interval(single_numeric_range):
                    assert single_numeric_range[attribute_index].ranges[0][1] >= \
                           single_numeric_range[attribute_index].ranges[0][0], \
                        "Value range does not contain any values"

                map(is_invalid_interval, ranges)

            else:
                raise ValueError("Unknown Metatype")

        if group_by_columns_merged is not None:
            for matching_group_by_idx, column in enumerate(group_by_columns_merged):
                if column not in self.column_names:
                    continue
                attribute_index = self.column_names.index(column)
                if self.meta_types[attribute_index] == MetaType.DISCRETE:
                    for idx in range(len(ranges)):
                        literal = group_by_tuples[idx][matching_group_by_idx]
                        if not isinstance(literal, list):
                            literal = [literal]

                        if ranges[idx, attribute_index] is None:
                            ranges[idx, attribute_index] = NominalRange(literal)
                        else:
                            updated_possible_values = set(ranges[idx, attribute_index].possible_values).intersection(
                                literal)
                            ranges[idx, attribute_index] = NominalRange(list(updated_possible_values))

                elif self.meta_types[attribute_index] == MetaType.REAL:
                    for idx in range(len(ranges)):
                        literal = group_by_tuples[idx][matching_group_by_idx]
                        assert not isinstance(literal, list)
                        ranges[idx, attribute_index] = NumericRange([[literal, literal]])
                else:
                    raise ValueError("Unknown Metatype")

        return ranges


def _literal_list(condition):
    _, literals = condition.split('(', 1)
    return [value.strip(' "\'') for value in literals[:-1].split(',')]


def _adapt_ranges(attribute_index, literal, ranges, inclusive=True, lower_than=True):
    matching_none_intervals = [idx for idx, single_range in enumerate(ranges[:, attribute_index]) if
                               single_range is None]
    if lower_than:
        for idx, single_range in enumerate(ranges):
            if single_range[attribute_index] is None or single_range[attribute_index].ranges[0][1] <= literal:
                continue
            ranges[idx, attribute_index].ranges[0][1] = literal
            ranges[idx, attribute_index].inclusive_intervals[0][1] = inclusive

        ranges[matching_none_intervals, attribute_index] = NumericRange([[-np.inf, literal]],
                                                                        inclusive_intervals=[[False, inclusive]])

    else:
        for idx, single_range in enumerate(ranges):
            if single_range[attribute_index] is None or single_range[attribute_index].ranges[0][0] >= literal:
                continue
            ranges[idx, attribute_index].ranges[0][0] = literal
            ranges[idx, attribute_index].inclusive_intervals[0][0] = inclusive
        ranges[matching_none_intervals, attribute_index] = NumericRange([[literal, np.inf]],
                                                                        inclusive_intervals=[[inclusive, False]])

    return ranges
