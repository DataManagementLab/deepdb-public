import copy
import logging
from time import perf_counter

import numpy as np
from spn.structure.StatisticalTypes import MetaType

from aqp_spn.group_by_combination import group_by_combinations
from ensemble_compilation.spn_ensemble import CombineSPN
from rspn.algorithms.ranges import NominalRange, NumericRange
from rspn.rspn import RSPN
from rspn.structure.leaves import IdentityNumericLeaf, identity_distinct_ranges, categorical_distinct_ranges, \
    Categorical, identity_likelihood_range, categorical_likelihood_range
from rspn.updates.top_down_updates import cluster_center_update_dataset

logger = logging.getLogger(__name__)


class AQPSPN(CombineSPN, RSPN):

    def __init__(self, meta_types, null_values, full_join_size, schema_graph, relationship_list, full_sample_size=None,
                 table_set=None, column_names=None, table_meta_data=None):

        full_sample_size = full_sample_size
        if full_sample_size is None:
            full_sample_size = full_join_size

        CombineSPN.__init__(self, full_join_size, schema_graph, relationship_list, table_set=table_set)
        RSPN.__init__(self, meta_types, null_values, full_sample_size,
                      column_names=column_names,
                      table_meta_data=table_meta_data)

    def add_dataset(self, dataset):
        """
        modifies the RSPN based on new dataset.
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

        cluster_center_update_dataset(self.mspn, dataset)
        self.full_sample_size += 1
        self.full_join_size += 1

    def learn(self, train_data, rdc_threshold=0.3, min_instances_slice=1, max_sampling_threshold_cols=10000,
              max_sampling_threshold_rows=100000, bloom_filters=False):

        # find scopes for variables which indicate not null column
        no_compression_scopes = None
        if self.column_names is not None:
            no_compression_scopes = []
            for table in self.table_set:
                table_obj = self.schema_graph.table_dictionary[table]
                for attribute in table_obj.no_compression:
                    column_name = table + '.' + attribute
                    if column_name in self.column_names:
                        no_compression_scopes.append(self.column_names.index(column_name))

        RSPN.learn(self, train_data, rdc_threshold=rdc_threshold, min_instances_slice=min_instances_slice,
                   max_sampling_threshold_cols=max_sampling_threshold_cols,
                   max_sampling_threshold_rows=max_sampling_threshold_rows,
                   no_compression_scopes=no_compression_scopes)

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
