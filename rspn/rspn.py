import logging
import time
from functools import partial

import numpy as np
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType

from rspn.algorithms.expectations import expectation
from rspn.algorithms.ranges import NominalRange, NumericRange
from rspn.algorithms.validity.validity import is_valid
from rspn.learning.rspn_learning import learn_mspn
from rspn.structure.leaves import IdentityNumericLeaf, identity_expectation, Categorical, categorical_likelihood_range, \
    identity_likelihood_range

logger = logging.getLogger(__name__)


def build_ds_context(column_names, meta_types, null_values, table_meta_data, no_compression_scopes, train_data,
                     group_by_threshold=1200):
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

    if no_compression_scopes is None:
        no_compression_scopes = []
    ds_context.no_compression_scopes = no_compression_scopes

    return ds_context


class RSPN:

    def __init__(self, meta_types, null_values, full_sample_size, column_names=None, table_meta_data=None):

        self.meta_types = meta_types
        self.null_values = null_values
        self.full_sample_size = full_sample_size
        self.column_names = column_names
        self.table_meta_data = table_meta_data
        self.mspn = None
        self.ds_context = None

        self.use_generated_code = False

        # training stats
        self.learn_time = None
        self.rdc_threshold = None
        self.min_instances_slice = None

    def learn(self, train_data, rdc_threshold=0.3, min_instances_slice=1, max_sampling_threshold_cols=10000,
              max_sampling_threshold_rows=100000, no_compression_scopes=None):

        # build domains (including the dependence analysis)
        domain_start_t = time.perf_counter()
        ds_context = build_ds_context(self.column_names, self.meta_types, self.null_values, self.table_meta_data,
                                      no_compression_scopes, train_data)
        self.ds_context = ds_context
        domain_end_t = time.perf_counter()
        logging.debug(f"Built domains in {domain_end_t - domain_start_t} sec")

        # learn mspn
        learn_start_t = time.perf_counter()

        self.mspn = learn_mspn(train_data, ds_context,
                               min_instances_slice=min_instances_slice, threshold=rdc_threshold,
                               max_sampling_threshold_cols=max_sampling_threshold_cols,
                               max_sampling_threshold_rows=max_sampling_threshold_rows)
        assert is_valid(self.mspn, check_ids=True)
        learn_end_t = time.perf_counter()
        self.learn_time = learn_end_t - learn_start_t
        logging.debug(f"Built SPN in {learn_end_t - learn_start_t} sec")

        # statistics
        self.rdc_threshold = rdc_threshold
        self.min_instances_slice = min_instances_slice

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
