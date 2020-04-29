import copy
import itertools
import logging
import pickle
import random
from collections import deque
from time import perf_counter

import numpy as np
import scipy.stats
from ensemble_compilation.graph_representation import Query, QueryType, AggregationType, AggregationOperationType
from ensemble_compilation.probabilistic_query import IndicatorExpectation, Expectation
from evaluation.utils import parse_what_if_query, all_operations_of_type
from spn.algorithms.Statistics import get_structure_stats
import bz2

np.random.seed(1)

logger = logging.getLogger(__name__)


class CombineSPN:
    """
    An SPN built over a sub-schema.

    Application a) estimate cardinality for arbitrary acyclic sub-query using
    the equation Full_join_size*E(1/multiplier * 1_{c_1 Λ… Λc_n}).

    E.g. We have an SPN built over Customer, Orders and Orderline. What is the
    cardinality for customers joined with orders made in 2017?

    Application b) cardinality per tuple of next neighbour:
     Full_join_size*E(1/multiplier * 1_{c_1 Λ… Λc_n}) / next_neighbour_size
    This term is required if we merge an SPN into the current cardinality estimation.

    E.g. We have an SPN built over Customer and one over Orders and Orderline.
    What is the cardinality for Customers, Orders and Orderline joined with
    Orders.year = 2019?

    Assumption is that is built over acyclic graph for now.

    """

    def __init__(self, full_join_size, schema_graph, relationship_list, table_set=None):

        self.full_join_size = full_join_size
        self.schema_graph = schema_graph

        # relationships joined for building the SPN
        self.relationship_set = set()
        if relationship_list is None:
            relationship_list = []
        for relationship in relationship_list:
            assert (self.schema_graph.relationship_dictionary.get(relationship) is not None)
            self.relationship_set.add(relationship)

        # set of tables included
        if table_set is None:
            self.table_set = set()
        else:
            self.table_set = table_set

        for relationship in relationship_list:
            relationship_obj = self.schema_graph.relationship_dictionary.get(relationship)
            self.table_set.add(relationship_obj.start)
            self.table_set.add(relationship_obj.end)

    def evaluate_indicator_expectation(self, indicator_expectation):
        raise NotImplementedError

    def evaluate_expectation(self, expectation):
        raise NotImplementedError

    def evaluate_indicator_expectation_batch(self, indicator_expectation, group_bys, group_by_tuples,
                                             standard_deviations=False):
        raise NotImplementedError

    def evaluate_expectation_batch(self, expectation, group_bys, group_by_tuples, standard_deviations=False):
        raise NotImplementedError

    def relevant_conditions(self, query, merged_tables=None):
        """Compute conditions for E(1/multiplier * 1_{c_1 Λ… Λc_n}) (Application A)."""

        condition_dict = query.table_where_condition_dict
        conditions = []

        if merged_tables is None:
            merged_tables = query.table_set.intersection(self.table_set)

        # Conditions from Query
        for table in condition_dict.keys():
            if table in merged_tables:
                for condition in condition_dict[table]:
                    conditions.append((table, condition))

        # We have to require the tables of the query to be not null
        # This can happen since we learn the SPN on full outer join
        for table in merged_tables:
            table_obj = self.schema_graph.table_dictionary[table]
            condition = table_obj.table_nn_attribute + ' IS NOT NULL'
            conditions.append((table, condition))

        return conditions

    def compute_multipliers(self, query):
        """Compute normalizing multipliers for E(1/multiplier * 1_{c_1 Λ… Λc_n}) (Application A).

        Idea: Do a BFS tree search. Only a relevant multiplier if relationship is from
        higher degree to lower degree. So we store each table in dict.
        """

        queue = deque()
        depth_dict = dict()
        # Usually for BFS we would need a set of visited nodes. This is not required here.
        # We can simply use depth_dict

        for table in query.table_set:
            queue.append(table)
            depth_dict[table] = 0

        depth_dict = self.compute_depths(queue, depth_dict)
        norm_multipliers = []

        # evaluate for every relationship if normalization is necessary
        for relationship in self.relationship_set:
            if relationship not in query.relationship_set:
                relationship_obj = self.schema_graph.relationship_dictionary[relationship]

                # only queries directed to query have to be included
                if depth_dict[relationship_obj.start] > depth_dict[relationship_obj.end]:
                    norm_multipliers.append((relationship_obj.end, relationship_obj.multiplier_attribute_name_nn))

        return norm_multipliers

    def compute_depths(self, queue, depth_dict):
        """
        Do a BFS to compute min-distance of tables to set of tables already in queue.
        """

        # while not empty
        while queue:

            # BFS
            table = queue.popleft()

            # list neighbours
            table_obj = self.schema_graph.table_dictionary[table]

            for relationship in table_obj.incoming_relationships:
                # only consider if part of relationships of combine-SPN
                if relationship.identifier in self.relationship_set:

                    # potentially new table
                    potential_new_table = relationship.start
                    if potential_new_table not in depth_dict.keys():
                        queue.append(potential_new_table)
                        depth_dict[potential_new_table] = depth_dict[table] + 1

            for relationship in table_obj.outgoing_relationships:
                # only consider if part of relationships of combine-SPN
                if relationship.identifier in self.relationship_set:

                    # potentially new table
                    potential_new_table = relationship.end
                    if potential_new_table not in depth_dict.keys():
                        queue.append(potential_new_table)
                        depth_dict[potential_new_table] = depth_dict[table] + 1

        return depth_dict

    def compute_mergeable_relationships(self, query, start_table):
        """
        Compute which relationships are merged starting from a certain table (Application B)
        """

        relationships = []
        queue = deque()
        queue.append(start_table)

        while queue:
            # BFS
            table = queue.popleft()

            # list neighbours
            table_obj = self.schema_graph.table_dictionary[table]

            for relationship in table_obj.incoming_relationships:
                # only mergeable if part of SPN and still to be merged in query
                if relationship.identifier in self.relationship_set and \
                        relationship.identifier in query.relationship_set and \
                        relationship.identifier not in relationships:
                    relationships.append(relationship.identifier)
                    queue.append(relationship.start)

            for relationship in table_obj.outgoing_relationships:
                # only mergeable if part of SPN and still to be merged in query
                if relationship.identifier in self.relationship_set and \
                        relationship.identifier in query.relationship_set and \
                        relationship.identifier not in relationships:
                    relationships.append(relationship.identifier)
                    queue.append(relationship.end)

        return relationships


def read_ensemble(ensemble_locations, build_reverse_dict=False):
    """
    Creates union of all SPNs in the different ensembles.
    :param min_sample_ratio:
    :param ensemble_locations: list of file locations of ensembles.
    :return:
    """
    if not isinstance(ensemble_locations, list):
        ensemble_locations = [ensemble_locations]

    ensemble = SPNEnsemble(None)
    for ensemble_location in ensemble_locations:
        with open(ensemble_location, 'rb') as handle:
            current_ensemble = pickle.load(handle)
            ensemble.schema_graph = current_ensemble.schema_graph
            for spn in current_ensemble.spns:
                logging.debug(f"Including SPN with table_set {spn.table_set} with sampling ratio"
                              f"({spn.full_sample_size} / {spn.full_join_size})")
                # logging.debug(f"Stats: ({get_structure_stats(spn.mspn)})")
                # build reverse dict.
                if build_reverse_dict:
                    _build_reverse_spn_dict(spn)
                ensemble.add_spn(spn)
    return ensemble


def what_if_scenario(what_if_query, ensemble):
    """
    Create SPN ensemble for what if scenario

    :param what_if_query: the query specifying the subpopulation and the percentage change
    :param ensemble: the original ensemble
    :return: new SPN ensemble where the subpopulation that fulfills conditions occurs more or less frequently depending
    on the percentage change
    """
    what_if_start_t = perf_counter()
    conditions, percentage_change = parse_what_if_query(what_if_query, ensemble.schema_graph)

    what_if_ensemble = copy.deepcopy(ensemble)
    # reset cache to make what if work
    what_if_ensemble.cached_expecation_vals = dict()
    affected_tables = set([condition[0] for condition in conditions])

    for spn in what_if_ensemble.spns:
        if len(spn.table_set.intersection(affected_tables)) == 0:
            continue
        projected_conditions = [condition for condition in conditions if condition[0] in spn.table_set]
        spn_conditions = spn._parse_conditions(projected_conditions)
        spn_conditions = spn._add_null_values_to_ranges(spn_conditions)
        evidence = spn_conditions[0]

        # make sure multiplier means remain unchanged
        force_equal_mean_cols = []
        for i, column_name in enumerate(spn.column_names):
            for relationship in what_if_ensemble.schema_graph.relationships:
                if relationship.multiplier_attribute_name in column_name or relationship.multiplier_attribute_name_nn in column_name:
                    force_equal_mean_cols.append(i)

        transform_what_if(spn, evidence, percentage_change, force_equal_mean_cols=force_equal_mean_cols,
                          null_values=spn.null_values, transform_copy=False)

    what_if_end_t = perf_counter()
    logger.debug(f"Created new what if ensemble for conditions {conditions} ({'+' if percentage_change > 0 else ''}"
                 f"{percentage_change * 100:.2f}%) in {what_if_end_t - what_if_start_t} secs.")

    return what_if_ensemble


def _build_reverse_spn_dict(spn):
    spn.table_meta_data['inverted_columns_dict'] = dict()
    spn.table_meta_data['inverted_fd_dict'] = dict()
    for table in spn.table_meta_data.keys():
        if table == 'inverted_columns_dict' or table == 'inverted_fd_dict':
            continue
        categorical_colums = spn.table_meta_data[table]['categorical_columns_dict']
        for categorical_column in categorical_colums.keys():
            inverted_dictionary = dict()
            for k, v in categorical_colums[categorical_column].items():
                inverted_dictionary[v] = k
            spn.table_meta_data['inverted_columns_dict'][categorical_column] = inverted_dictionary
        if spn.table_meta_data[table].get('fd_dict') is not None:
            fd_colums = spn.table_meta_data[table]['fd_dict']
            # dwdate.d_dayofweek -> dwdate.d_daynuminweek
            for dest_column, source_column_dictionary in fd_colums.items():
                for source_column, value_dictionary in source_column_dictionary.items():
                    if spn.table_meta_data['inverted_fd_dict'].get(source_column) is None:
                        spn.table_meta_data['inverted_fd_dict'][source_column] = dict()
                    inverted_dictionary = dict()
                    for k, v_list in value_dictionary.items():
                        for v in v_list:
                            inverted_dictionary[v] = k
                    spn.table_meta_data['inverted_fd_dict'][source_column][dest_column] = inverted_dictionary
        else:
            spn.table_meta_data[table]['fd_dict'] = dict()


def evaluate_factors_group_by(artificially_added_conditions, confidence_intervals, debug,
                              factor_values_full, factors_full, result_tuples, technical_group_by_scopes,
                              confidence_interval_samples=None):
    # see if we can exclude inverted factors, quadratic complexity but usually only very few factors
    factors_to_be_deleted = set()
    for left_factor in factors_full:
        if not isinstance(left_factor, IndicatorExpectation):
            continue
        for right_factor in factors_full:
            if not isinstance(right_factor, IndicatorExpectation):
                continue
            if left_factor.is_inverse(right_factor):
                factors_to_be_deleted.add(left_factor)
                factors_to_be_deleted.add(right_factor)
                logger.debug("Removed two factors for evaluation.")
    factors = [factor for factor in factors_full if factor not in factors_to_be_deleted]
    factor_values = [factor_value for i, factor_value in enumerate(factor_values_full) if
                     not factors_full[i] in factors_to_be_deleted]

    card_start_t = perf_counter()
    # evaluate factors to obtain cardinality and formula
    cardinalities = np.ones((len(result_tuples), 1)) * factors[0]

    if confidence_intervals:
        # idea: compute indicator stds with bernoulli approach
        # store all indicator exps in [:,1]
        no_exp = len([factor for factor in factors if isinstance(factor, Expectation)])
        factor_exps = np.ones((len(result_tuples), no_exp + 2))
        factor_exps[:, 0] = factors[0]
        factor_stds = np.zeros((len(result_tuples), no_exp + 2))

    exps_counter = 0
    for i, factor in enumerate(factors):
        if i == 0:
            continue
        if isinstance(factor, IndicatorExpectation):

            def project_tuple(orig_tuple, projection_idxs):
                projected_tuple = tuple()
                for i in projection_idxs:
                    if isinstance(orig_tuple[i], list):
                        orig_tuple[i].sort()
                        projected_tuple += (tuple(orig_tuple[i]),)
                        continue
                    projected_tuple += (orig_tuple[i],)
                return projected_tuple

            def project_list_tuple(orig_tuple):
                projected_tuple = tuple()
                for i in range(len(orig_tuple)):
                    if isinstance(orig_tuple[i], tuple):
                        projected_tuple += (list(orig_tuple[i]),)
                        continue
                    projected_tuple += (orig_tuple[i],)
                return projected_tuple

            # see which group by conditions apply to this factor
            specific_group_by_scopes = []
            for group_idx, artificially_added_condition in enumerate(artificially_added_conditions):
                if artificially_added_condition in factor.conditions:
                    factor.conditions.remove(artificially_added_condition)
                    specific_group_by_scopes.append(group_idx)
            # factor has to be recomputed for different group by scopes
            if len(specific_group_by_scopes) > 0:
                specific_technical_group_by_scopes = [technical_group_by_scopes[i] for i in
                                                      specific_group_by_scopes]
                specific_result_tuples = [project_tuple(result_tuple, specific_group_by_scopes) for result_tuple in
                                          result_tuples]
                # remember which unique projected values appeared and store them in order
                different_specific_result_tuples = []
                specific_result_dict = dict()
                for j, specific_result in enumerate(set(specific_result_tuples)):
                    specific_result_dict[specific_result] = j
                    different_specific_result_tuples.append(specific_result)
                different_specific_result_tuples_as_list = [project_list_tuple(result_tuple) for result_tuple in
                                                            different_specific_result_tuples]

                _, unprojected_exps = \
                    factor.spn.evaluate_indicator_expectation_batch(factor,
                                                                    specific_technical_group_by_scopes,
                                                                    different_specific_result_tuples_as_list,
                                                                    standard_deviations=False)
                exps = np.ones((len(result_tuples), 1))
                for idx, result_tuple in enumerate(result_tuples):
                    projected_tuple = project_tuple(result_tuple, specific_group_by_scopes)
                    unprojected_idx = specific_result_dict[projected_tuple]

                    exps[idx] = unprojected_exps[unprojected_idx]

                cardinalities *= exps
                if confidence_intervals:
                    factor_exps[:, 1] *= np.reshape(exps, len(result_tuples))

            elif confidence_intervals:
                cardinalities *= factor_values[i]
                factor_exps[:, 1] *= factor_values[i]

            else:
                cardinalities *= factor_values[i]

        elif isinstance(factor, Expectation):
            for artificially_added_condition in artificially_added_conditions:
                if artificially_added_condition in factor.conditions:
                    factor.conditions.remove(artificially_added_condition)

            stds, exps = factor.spn.evaluate_expectation_batch(factor, technical_group_by_scopes, result_tuples,
                                                               standard_deviations=confidence_intervals)
            cardinalities = np.multiply(exps, cardinalities)
            if confidence_intervals:
                ci_index = exps_counter + 2
                factor_exps[:, ci_index] = np.reshape(exps, len(result_tuples))
                factor_stds[:, ci_index] = np.reshape(stds, len(result_tuples))

            exps_counter += 1
        else:
            cardinalities *= factor
            if confidence_intervals:
                factor_exps[:, 0] *= factor

    cardinality_stds = None
    if confidence_intervals:
        assert confidence_interval_samples is not None, \
            "confidence_interval_samples is required for confidence interval calculation"

        bernoulli_p = factor_exps[:, 1]
        factor_stds[:, 1] = np.reshape(np.sqrt(bernoulli_p * (1 - bernoulli_p) / confidence_interval_samples),
                                       len(result_tuples))
        cardinality_stds = std_of_products(factor_exps, factor_stds)

    card_end_t = perf_counter()
    if debug:
        logger.debug(f"\t\tcomputed all cardinalities in {card_end_t - card_start_t} secs.")
    logger.debug(f"\t\taverage_cardinality: {cardinalities.mean()}")

    return cardinality_stds, cardinalities


def std_of_products(exps, stds):
    """
    Computes the std of independent random variables.
    :param exps:
    :param stds:
    :param non_constant_factors:
    :return:
    """
    # product(var(X_i) + E(X_i)^2)-product(E(X_i))^2
    std_shape = (exps.shape[0], 1)

    product_left = np.ones(std_shape)
    product_right = np.ones(std_shape)

    for i in range(exps.shape[1]):
        # var(X_i) + E(X_i)^2
        product_left *= np.reshape(np.square(stds[:, i]) + np.square(exps[:, i]), std_shape)
        # E(X_i)^2
        product_right *= np.reshape(np.square(exps[:, i]), std_shape)

    return np.sqrt(product_left - product_right)


def evaluate_factors(dry_run, factors_full, cached_expecation_vals, confidence_intervals=False,
                     confidence_interval_samples=None, gen_code_stats=None):
    # see if we can exclude inverted factors, quadratic complexity but usually only very few factors
    factors_to_be_deleted = set()
    for left_factor in factors_full:
        if not isinstance(left_factor, IndicatorExpectation):
            continue
        for right_factor in factors_full:
            if not isinstance(right_factor, IndicatorExpectation):
                continue
            if left_factor.is_inverse(right_factor):
                factors_to_be_deleted.add(left_factor)
                factors_to_be_deleted.add(right_factor)
                logger.debug("Removed two factors for evaluation.")
    factors = [factor for factor in factors_full if factor not in factors_to_be_deleted]

    # evaluate factors to obtain cardinality and formula
    formula = None
    cardinality = None
    values = []
    non_constant_factors = []
    if confidence_intervals:
        # idea: compute indicator stds with bernoulli approach
        # store all indicator exps in [:,1]
        no_exp = len([factor for factor in factors if isinstance(factor, Expectation)])
        factor_exps = np.ones((1, no_exp + 2))
        factor_exps[:, 0] = factors[0]
        factor_stds = np.zeros((1, no_exp + 2))

    exps_counter = 0
    for i, factor in enumerate(factors):
        if formula is None:
            assert i == 0
            formula = str(factor)
            cardinality = factor
            values.append(factor)
            continue

        formula += " * " + str(factor)
        if not dry_run:
            if isinstance(factor, IndicatorExpectation):
                if cached_expecation_vals.get(hash(factor)) is not None:
                    exp = cached_expecation_vals[hash(factor)]
                else:
                    _, exp = factor.spn.evaluate_indicator_expectation(factor, gen_code_stats=gen_code_stats,
                                                                       standard_deviations=False)
                    cached_expecation_vals[hash(factor)] = exp

                if confidence_intervals:
                    factor_exps[:, 1] *= exp
                non_constant_factors.append(i)
                values.append(exp)
                cardinality *= exp

            elif isinstance(factor, Expectation):
                if not confidence_intervals and cached_expecation_vals.get(hash(factor)) is not None:
                    std, exp = cached_expecation_vals[hash(factor)]
                else:
                    std, exp = factor.spn.evaluate_expectation(factor, standard_deviations=confidence_intervals,
                                                               gen_code_stats=gen_code_stats)
                    if confidence_intervals:
                        ci_index = exps_counter + 2
                        factor_exps[:, ci_index] = exp
                        factor_stds[:, ci_index] = std
                    cached_expecation_vals[hash(factor)] = std, exp
                non_constant_factors.append(i)
                values.append(exp)
                cardinality *= exp

                exps_counter += 1

            else:
                cardinality *= factor
                values.append(factor)
                if confidence_intervals:
                    factor_exps[:, 0] *= factor

    if confidence_intervals:
        assert confidence_interval_samples is not None, \
            "confidence_interval_samples is required for confidence interval calculation"

        bernoulli_p = factor_exps[:, 1]
        factor_stds[:, 1] = np.sqrt(bernoulli_p * (1 - bernoulli_p) / confidence_interval_samples)
        cardinality_stds = std_of_products(factor_exps, factor_stds)
        return cardinality_stds, values, cardinality, formula
    else:
        return values, cardinality, formula


def infer_column(condition):
    column = None
    if '<=' in condition:
        column, _ = condition.split('<=', 1)
    elif '>=' in condition:
        column, _ = condition.split('>=', 1)
    elif '>' in condition:
        column, _ = condition.split('>', 1)
    elif '<' in condition:
        column, _ = condition.split('<', 1)
    elif '=' in condition:
        column, _ = condition.split('=', 1)
    elif 'NOT IN' in condition:
        column, _ = condition.split('NOT IN', 1)
    elif 'IN' in condition:
        column, _ = condition.split('IN', 1)

    assert column is not None, "Condition not recognized"
    return column.strip()


class SPNEnsemble:
    """
    Several SPNs combined.

    Assumptions:
    - SPNs do not partition the entire graph.
    - SPNs represent trees.
    - Queries are trees. (This could be relaxed.)
    - For FK relationship referenced entity exists, e.g. every order has a customer. (Not sure about this one)
    """

    def __init__(self, schema_graph, spns=None):
        self.schema_graph = schema_graph
        self.spns = spns
        self.cached_expecation_vals = dict()
        if self.spns is None:
            self.spns = []

    def use_generated_code(self):
        for spn in self.spns:
            assert hasattr(spn, 'id'), "Assigned ids are required to employ generated code. Was this step done?"
            spn.use_generated_code = True
            # todo. warm start. compute dummy expectation on this spn.

    def save(self, ensemble_path, compress=False):
        if compress:
            with bz2.BZ2File(ensemble_path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(ensemble_path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def add_spn(self, spn):
        """Add an SPN to ensemble"""
        self.spns.append(spn)

    def _cardinality_greedy(self, query, rdc_spn_selection=False, rdc_attribute_dict=None, dry_run=False,
                            merge_indicator_exp=True, exploit_overlapping=False, return_factor_values=False,
                            exploit_incoming_multipliers=True, prefer_disjunct=False, gen_code_stats=None):
        """
        Find first SPN for cardinality estimate.
        """
        # Greedily select first SPN
        first_spn, next_mergeable_relationships, next_mergeable_tables = self._greedily_select_first_cardinality_spn(
            query, rdc_spn_selection=rdc_spn_selection, rdc_attribute_dict=rdc_attribute_dict)

        return self._cardinality_with_injected_start(query, first_spn, next_mergeable_relationships,
                                                     next_mergeable_tables, rdc_spn_selection=rdc_spn_selection,
                                                     rdc_attribute_dict=rdc_attribute_dict,
                                                     dry_run=dry_run,
                                                     merge_indicator_exp=merge_indicator_exp,
                                                     exploit_overlapping=exploit_overlapping,
                                                     return_factor_values=return_factor_values,
                                                     exploit_incoming_multipliers=exploit_incoming_multipliers,
                                                     prefer_disjunct=prefer_disjunct,
                                                     gen_code_stats=gen_code_stats)

    def _evaluate_group_by_spn_ensembles(self, query):
        """
        Go over all Group By attributes, find best SPN with maximal where conditions. Merge features that have same SPN.
        """

        spn_group_by_dict = dict()
        group_by_list = [table + '.' + attribute for table, attribute in query.group_bys]
        for i, grouping_attribute in enumerate(group_by_list):
            max_matching_where_cond = -1
            grouping_spn = None
            # search for spn with maximal matching where conditions
            for spn in self.spns:
                potential_group_by_columns = set(spn.column_names)
                for table in spn.table_set:
                    potential_group_by_columns = potential_group_by_columns.union(
                        spn.table_meta_data[table]['fd_dict'].keys())

                if grouping_attribute not in potential_group_by_columns:
                    continue
                where_conditions = set(query.table_where_condition_dict.keys()).intersection(spn.table_set)
                matching_spns = 0
                if spn in spn_group_by_dict.keys():
                    matching_spns = 1
                if len(where_conditions) > max_matching_where_cond or \
                        (len(where_conditions) == max_matching_where_cond and matching_spns > 0):
                    max_matching_where_cond = len(where_conditions)
                    grouping_spn = spn
            # use this spn in group by dictionary
            if spn_group_by_dict.get(grouping_spn) is None:
                spn_group_by_dict[grouping_spn] = []
            spn_group_by_dict[grouping_spn].append(grouping_attribute)

        group_by_permutation = np.zeros(len(group_by_list), dtype=int)
        dict_items = list(spn_group_by_dict.items())
        # permutation of group by queries
        attribute_counter = 0
        for spn, attribute_list in dict_items:
            for attribute in attribute_list:
                group_by_permutation[group_by_list.index(attribute)] = attribute_counter
                attribute_counter += 1

        result_tuples = None
        result_tuples_translated = None
        group_bys_scopes = []
        for spn, attribute_list in dict_items:
            conditions = spn.relevant_conditions(query)
            group_bys_scope, temporary_results, temporary_results_translated = spn.evaluate_group_by_combinations(
                attribute_list,
                conditions)
            group_bys_scopes += group_bys_scope
            if result_tuples is None:
                result_tuples = temporary_results
                result_tuples_translated = temporary_results_translated
            else:
                result_tuples = [result_tuple + temporary_result for result_tuple in result_tuples for temporary_result
                                 in temporary_results]
                result_tuples_translated = [result_tuple + temporary_result for result_tuple in result_tuples_translated
                                            for temporary_result in temporary_results_translated]

        # reorder by tuple permutation
        # group by scopes do not have to be reordered
        group_bys_scopes = [group_bys_scopes[i] for i in group_by_permutation]
        result_tuples = [tuple([result_tuple[i] for i in group_by_permutation]) for result_tuple in result_tuples]
        result_tuples_translated = [tuple([result_tuple[i] for i in group_by_permutation]) for result_tuple in
                                    result_tuples_translated]

        return group_bys_scopes, result_tuples, result_tuples_translated

    # def predict(self, conditions, regression_column):
    #     """
    #     Conditions
    #     :param conditions: dictionary of table, tuple condition pairs
    #     :param feature:
    #     :return:
    #     """
    #
    #     max_where_conditions = -1
    #     prediction_spn = None
    #
    #     for spn in self.spns:
    #         # if spn contains all features consider it a candidate
    #         if regression_column in spn.column_names:
    #
    #             where_conditions = [condition for condition in conditions if condition[0] in spn.table_set]
    #
    #             if len(where_conditions) > max_where_conditions:
    #                 prediction_spn = spn
    #                 max_where_conditions = len(where_conditions)
    #
    #     assert prediction_spn is not None, "Did not find SPN offering this feature"
    #
    #     ranges = prediction_spn._parse_conditions(conditions)
    #
    #     return prediction_spn.predict(ranges, regression_column)

    def evaluate_query(self, query, rdc_spn_selection=False, pairwise_rdc_path=None,
                       dry_run=False, merge_indicator_exp=True, max_variants=10,
                       exploit_overlapping=False, debug=False, display_intermediate_results=False,
                       exploit_incoming_multipliers=True, confidence_intervals=False,
                       confidence_sample_size=None, return_expectation=False):
        """
        Evaluates any query with or without a group by.
        :param query:
        :param dry_run:
        :param merge_indicator_exp:
        :param max_variants:
        :param exploit_overlapping:
        :return:
        """

        result_tuples = None
        technical_group_by_scopes = []
        if len(query.group_bys) > 0:
            group_by_start_t = perf_counter()
            # tuples that should appear in the group by clause
            group_bys_scopes, result_tuples, result_tuples_translated = self._evaluate_group_by_spn_ensembles(query)
            group_by_end_t = perf_counter()
            technical_group_by_scopes = [tuple(group_bys_scope.split('.', 1)) for group_bys_scope in group_bys_scopes]
            if debug:
                logger.debug(f"\t\tcomputed {len(result_tuples)} group by statements "
                             f"in {group_by_end_t - group_by_start_t} secs.")

        # if cardinality query simply return it
        if query.query_type == QueryType.CARDINALITY or any(
                [aggregation_type == AggregationType.SUM or aggregation_type == AggregationType.COUNT
                 for _, aggregation_type, _ in query.aggregation_operations]):

            prot_card_start_t = perf_counter()

            # First get the prototypical factors for concrete group by tuple
            prototype_query = copy.deepcopy(query)
            artificially_added_conditions = []
            for group_by_idx, (table, attribute) in enumerate(query.group_bys):
                # add condition for first group bys
                condition = attribute + '=' + str(result_tuples_translated[0][group_by_idx])
                artificially_added_conditions.append((table, condition,))
                prototype_query.add_where_condition(table, condition)
            _, factors, cardinalities, factor_values = self.cardinality(prototype_query,
                                                                        rdc_spn_selection=rdc_spn_selection,
                                                                        pairwise_rdc_path=pairwise_rdc_path,
                                                                        dry_run=False,
                                                                        merge_indicator_exp=merge_indicator_exp,
                                                                        max_variants=max_variants,
                                                                        exploit_overlapping=exploit_overlapping,
                                                                        return_factor_values=True,
                                                                        exploit_incoming_multipliers=exploit_incoming_multipliers)
            prot_card_end_t = perf_counter()
            if debug:
                if len(query.group_bys) == 0:
                    logger.debug(f"\t\tpredicted cardinality: {cardinalities}")
                logger.debug(f"\t\tcomputed prototypical cardinality in {prot_card_end_t - prot_card_start_t} secs.")
            if len(query.group_bys) == 0 and confidence_intervals:
                _, factors_no_overlap, _, _ = self.cardinality(prototype_query,
                                                               rdc_spn_selection=rdc_spn_selection,
                                                               pairwise_rdc_path=pairwise_rdc_path,
                                                               dry_run=False,
                                                               merge_indicator_exp=False,
                                                               max_variants=max_variants,
                                                               exploit_overlapping=exploit_overlapping,
                                                               return_factor_values=True,
                                                               exploit_incoming_multipliers=exploit_incoming_multipliers,
                                                               prefer_disjunct=True)
                cardinality_stds, _, redundant_cardinality, _ = evaluate_factors(False, factors_no_overlap,
                                                                                 self.cached_expecation_vals,
                                                                                 confidence_intervals=True,
                                                                                 confidence_interval_samples=confidence_sample_size)
            if len(query.group_bys) > 0:
                _, cardinalities = evaluate_factors_group_by(
                    artificially_added_conditions, False,
                    debug, factor_values, factors, result_tuples,
                    technical_group_by_scopes)

                if confidence_intervals:
                    _, factors_no_overlap, _, factor_values_no_overlap = self.cardinality(prototype_query,
                                                                                          rdc_spn_selection=rdc_spn_selection,
                                                                                          pairwise_rdc_path=pairwise_rdc_path,
                                                                                          dry_run=False,
                                                                                          merge_indicator_exp=False,
                                                                                          max_variants=max_variants,
                                                                                          exploit_overlapping=exploit_overlapping,
                                                                                          return_factor_values=True,
                                                                                          exploit_incoming_multipliers=exploit_incoming_multipliers,
                                                                                          prefer_disjunct=True)
                    cardinality_stds, _ = evaluate_factors_group_by(
                        artificially_added_conditions, confidence_intervals,
                        debug, factor_values_no_overlap, factors_no_overlap, result_tuples,
                        technical_group_by_scopes, confidence_interval_samples=confidence_sample_size)

            # Bernoulli bound
            # if confidence_intervals:
            #     full_join_query = copy.deepcopy(query)
            #     full_join_query.conditions = []
            #     full_join_query.table_where_condition_dict = dict()
            #     _, _, full_join_size = self.cardinality(full_join_query, dry_run=False,
            #                                             merge_indicator_exp=merge_indicator_exp,
            #                                             max_variants=max_variants,
            #                                             exploit_overlapping=exploit_overlapping,
            #                                             return_factor_values=False,
            #                                             exploit_incoming_multipliers=exploit_incoming_multipliers)
            #
            #     bernoulli_p = cardinalities / full_join_size
            #     bernoulli_stds = full_join_size * np.sqrt(bernoulli_p * (1 - bernoulli_p) / 10000000)
            #     cardinality_stds = np.clip(cardinality_stds, bernoulli_stds, np.inf)

        def build_confidence_interval(prediction, confidence_interval_std):

            z_factor = scipy.stats.norm.ppf(0.95)
            lower_bound = prediction - z_factor * confidence_interval_std.item()
            upper_bound = prediction + z_factor * confidence_interval_std.item()

            return lower_bound, upper_bound

        if query.query_type == QueryType.CARDINALITY:
            if confidence_intervals:
                return build_confidence_interval(cardinalities, cardinality_stds), cardinalities
            return None, cardinalities

        result_values = None
        if all_operations_of_type(AggregationType.SUM, query) or all_operations_of_type(AggregationType.AVG, query):

            operation = None

            if confidence_intervals:
                if result_tuples is not None:
                    avg_exps = np.zeros((len(result_tuples), 1))
                    avg_stds = np.zeros((len(result_tuples), 1))
                else:
                    avg_exps = np.zeros((1, 1))
                    avg_stds = np.zeros((1, 1))

            for aggregation_operation_type, aggregation_type, factors in query.aggregation_operations:
                # just the operation on the aggregations
                if aggregation_operation_type == AggregationOperationType.PLUS or \
                        aggregation_operation_type == AggregationOperationType.MINUS:
                    operation = aggregation_operation_type

                # which aggregation
                elif aggregation_operation_type == AggregationOperationType.AGGREGATION:

                    # Either sum or avg value. In both cases expectation is required.
                    exp_start_t = perf_counter()
                    # todo. incorporate rdc values
                    expectation_spn, expectation = self._greedily_select_expectation_spn(query, factors)
                    if confidence_intervals:
                        current_stds, aggregation_result = expectation_spn.evaluate_expectation_batch(
                            expectation,
                            technical_group_by_scopes,
                            result_tuples,
                            standard_deviations=True)
                        avg_stds = np.sqrt(np.square(avg_stds) + np.square(current_stds))

                    else:
                        _, aggregation_result = expectation_spn.evaluate_expectation_batch(expectation,
                                                                                           technical_group_by_scopes,
                                                                                           result_tuples)
                    exp_end_t = perf_counter()
                    if debug:
                        logger.debug(f"\t\tcomputed expectation in {exp_end_t - exp_start_t} secs.")

                    logger.debug(
                        f"\t\taverage expectation: {np.array([aggregation_result]).mean()} for {expectation.features}")

                    # add or subtract current SUM or AVG from result
                    if result_values is None:
                        result_values = aggregation_result
                        if confidence_intervals:
                            avg_exps += aggregation_result
                    elif operation == AggregationOperationType.PLUS:
                        result_values += aggregation_result
                        if confidence_intervals:
                            avg_exps += aggregation_result
                    elif operation == AggregationOperationType.MINUS:
                        result_values -= aggregation_result
                        if confidence_intervals:
                            avg_exps -= aggregation_result
                    else:
                        raise NotImplementedError

            if all_operations_of_type(AggregationType.SUM, query):
                if confidence_intervals:
                    confidence_interval_stds = std_of_products(
                        np.concatenate((avg_exps, np.reshape(cardinalities, cardinality_stds.shape)), axis=1),
                        np.concatenate((avg_stds, cardinality_stds), axis=1))

                result_values *= cardinalities
            elif confidence_intervals:
                confidence_interval_stds = avg_stds

        # single count
        elif all_operations_of_type(AggregationType.COUNT, query):
            no_count_ops = len([aggregation_type for aggregation_operation_type, aggregation_type, _ in
                                query.aggregation_operations if
                                aggregation_operation_type == AggregationOperationType.AGGREGATION])
            assert no_count_ops == 1, "Only a single count operation is supported."

            result_values = cardinalities
            if confidence_intervals:
                confidence_interval_stds = cardinality_stds

        # mixed operations
        else:
            raise NotImplementedError("Mixed operations are currently not implemented.")

        # concatenate group by attribute and value if there is a group by
        if len(query.group_bys) > 0:
            result_tuples = [result_tuple + (result_values[i].item(),) for i, result_tuple in
                             enumerate(result_tuples_translated)]

            if confidence_intervals:
                confidence_values = []

                for i in range(confidence_interval_stds.shape[0]):
                    confidence_values.append(
                        build_confidence_interval(result_values[i][-1], confidence_interval_stds[i]))
                return confidence_values, result_tuples

            return None, result_tuples

        # if no group by queries return single value
        if confidence_intervals:
            return build_confidence_interval(result_values, confidence_interval_stds), result_values

        if return_expectation:
            return None, result_values, expectation_spn, expectation

        return None, result_values

    def cardinality(self, query, rdc_spn_selection=False, pairwise_rdc_path=None,
                    dry_run=False, merge_indicator_exp=True, max_variants=10, exploit_overlapping=False,
                    return_factor_values=False, exploit_incoming_multipliers=True, prefer_disjunct=False,
                    gen_code_stats=None):
        """
        Uses several ways to approximate the cardinality and returns the median for cardinality

        :param exploit_overlapping:
        :param max_variants:
        :param query:
        :param dry_run:
        :param merge_indicator_exp:
        :return:
        """
        rdc_attribute_dict = None
        if rdc_spn_selection:
            with open(pairwise_rdc_path, 'rb') as handle:
                rdc_attribute_dict = pickle.load(handle)

        possible_starts = self._possible_first_spns(query)
        # no where conditions given
        if len(possible_starts) == 0 or max_variants == 1:
            return self._cardinality_greedy(query, rdc_spn_selection=rdc_spn_selection,
                                            rdc_attribute_dict=rdc_attribute_dict,
                                            dry_run=dry_run, merge_indicator_exp=merge_indicator_exp,
                                            exploit_overlapping=exploit_overlapping,
                                            return_factor_values=return_factor_values,
                                            exploit_incoming_multipliers=exploit_incoming_multipliers,
                                            prefer_disjunct=prefer_disjunct, gen_code_stats=gen_code_stats)
        if len(possible_starts) > max_variants:
            random.shuffle(possible_starts)
            possible_starts = possible_starts[:max_variants]
        results = []
        for first_spn, next_mergeable_relationships, next_mergeable_tables in possible_starts:
            results.append(self._cardinality_with_injected_start(query, first_spn, next_mergeable_relationships,
                                                                 next_mergeable_tables,
                                                                 rdc_spn_selection=rdc_spn_selection,
                                                                 rdc_attribute_dict=rdc_attribute_dict,
                                                                 dry_run=dry_run,
                                                                 merge_indicator_exp=merge_indicator_exp,
                                                                 exploit_overlapping=exploit_overlapping,
                                                                 return_factor_values=return_factor_values,
                                                                 exploit_incoming_multipliers=exploit_incoming_multipliers,
                                                                 prefer_disjunct=prefer_disjunct,
                                                                 gen_code_stats=gen_code_stats))

        # it does not make sense to sort by cardinality if they are not yet computed
        results.sort(key=lambda x: x[2])
        return results[int(len(results) / 2)]

    def _cardinality_with_injected_start(self, query, first_spn, next_mergeable_relationships, next_mergeable_tables,
                                         rdc_spn_selection=False, rdc_attribute_dict=None, dry_run=False,
                                         merge_indicator_exp=True, exploit_overlapping=False,
                                         return_factor_values=False, exploit_incoming_multipliers=True,
                                         prefer_disjunct=False, gen_code_stats=None):
        """
        Always use SPN that matches most where conditions.

        :param query:
        :param first_spn:
        :param next_mergeable_relationships:
        :param next_mergeable_tables:
        :param dry_run:
        :param merge_indicator_exp:
        :return:
        """
        factors = []

        # only operate on copy so that query object is not changed
        # for greedy strategy it does not matter whether query is changed
        # optimized version of:
        # original_query = copy.deepcopy(query)
        # query = copy.deepcopy(query)
        original_query = query.copy_cardinality_query()
        query = query.copy_cardinality_query()

        # First SPN: Full_join_size*E(outgoing_mult * 1/multiplier * 1_{c_1 Λ… Λc_n})
        # Again create auxilary query because intersection of query relationships and spn relationships
        # is not necessarily a tree.
        auxilary_query = Query(self.schema_graph)
        for relationship in next_mergeable_relationships:
            auxilary_query.add_join_condition(relationship)
        auxilary_query.table_set.update(next_mergeable_tables)
        auxilary_query.table_where_condition_dict = query.table_where_condition_dict

        factors.append(first_spn.full_join_size)
        conditions = first_spn.relevant_conditions(auxilary_query)
        multipliers = first_spn.compute_multipliers(auxilary_query)

        # E(1/multipliers * 1_{c_1 Λ… Λc_n})
        expectation = IndicatorExpectation(multipliers, conditions, spn=first_spn, table_set=auxilary_query.table_set)
        factors.append(expectation)

        # mark tables as merged, remove merged relationships
        merged_tables = next_mergeable_tables
        query.relationship_set -= set(next_mergeable_relationships)

        # remember which SPN was used to merge tables
        corresponding_exp_dict = {}
        for table in merged_tables:
            corresponding_exp_dict[table] = expectation
        extra_multplier_dict = {}

        # merge subsequent relationships
        while len(query.relationship_set) > 0:

            # for next joins:
            # if not exploit_overlapping: cardinality next subquery / next_neighbour.table_size

            # compute set of next joinable neighbours
            next_neighbours, neighbours_relationship_dict = self._next_neighbours(query, merged_tables)

            # compute possible next merges and select greedily
            next_spn, next_neighbour, next_mergeable_relationships = self._greedily_select_next_table(original_query,
                                                                                                      query,
                                                                                                      next_neighbours,
                                                                                                      exploit_overlapping,
                                                                                                      merged_tables,
                                                                                                      prefer_disjunct=prefer_disjunct,
                                                                                                      rdc_spn_selection=rdc_spn_selection,
                                                                                                      rdc_attribute_dict=rdc_attribute_dict)

            # if outgoing: outgoing_mult appended to multipliers
            relationship_to_neighbour = neighbours_relationship_dict[next_neighbour]
            relationship_obj = self.schema_graph.relationship_dictionary[relationship_to_neighbour]

            incoming_relationship = True
            if relationship_obj.start == next_neighbour:
                incoming_relationship = False
                # outgoing relationship. Has to be included by E(outgoing_mult | C...)
                if merge_indicator_exp:
                    # For this computation we simply add the multiplier to the respective indicator expectation.
                    end_table = relationship_obj.end
                    indicator_expectation_outgoing_spn = corresponding_exp_dict[end_table]
                    indicator_expectation_outgoing_spn.nominator_multipliers.append(
                        (end_table, relationship_obj.multiplier_attribute_name))
                else:
                    # E(outgoing_mult | C...) weighted by normalizing_multipliers
                    end_table = relationship_obj.end
                    feature = (end_table, relationship_obj.multiplier_attribute_name)

                    # Search SPN with maximal considered conditions
                    max_considered_where_conditions = -1
                    spn_for_exp_computation = None

                    for spn in self.spns:
                        # attribute not even available
                        if hasattr(spn, 'column_names'):
                            if end_table + '.' + relationship_obj.multiplier_attribute_name not in spn.column_names:
                                continue
                        conditions = spn.relevant_conditions(original_query)
                        if len(conditions) > max_considered_where_conditions:
                            max_considered_where_conditions = len(conditions)
                            spn_for_exp_computation = spn

                    assert spn_for_exp_computation is not None, "No SPN found for expectation computation"

                    # if spn_for_exp_computation is already used for outgoing multiplier computation it should be used
                    # again. This captures correlations of multipliers better.
                    if extra_multplier_dict.get(spn_for_exp_computation) is not None:
                        expectation = extra_multplier_dict.get(spn_for_exp_computation)
                        expectation.features.append(feature)
                    else:
                        normalizing_multipliers = spn_for_exp_computation.compute_multipliers(original_query)
                        conditions = spn_for_exp_computation.relevant_conditions(original_query)

                        expectation = Expectation([feature], normalizing_multipliers, conditions,
                                                  spn=spn_for_exp_computation)
                        extra_multplier_dict[spn_for_exp_computation] = expectation
                        factors.append(expectation)

            # remove relationship_to_neighbour from query
            if relationship_to_neighbour in next_mergeable_relationships:
                next_mergeable_relationships.remove(relationship_to_neighbour)
            query.relationship_set.remove(relationship_to_neighbour)
            merged_tables.add(next_neighbour)

            # tables which are merged in the next step
            next_merged_tables = self._merged_tables(next_mergeable_relationships)
            next_merged_tables.add(next_neighbour)

            # find overlapping relationships (relationships already merged that also appear in next_spn)
            overlapping_relationships, overlapping_tables, no_overlapping_conditions = self._compute_overlap(
                next_neighbour, query, original_query,
                next_mergeable_relationships,
                next_merged_tables,
                next_spn)
            # remove neighbour
            overlapping_tables.remove(next_neighbour)

            # do not ignore overlap. Exploit knowledge of overlap.
            # in the computation use:
            # correct_indicator_expectation_with_overlap/ indicator_expectation_of_overlap

            # nominator query: indicator expectation of overlap + mergeable relationships
            nominator_query = Query(self.schema_graph)
            for relationship in overlapping_relationships:
                nominator_query.add_join_condition(relationship)
            for relationship in next_mergeable_relationships:
                nominator_query.add_join_condition(relationship)
            nominator_query.table_set.update(next_merged_tables)
            nominator_query.table_where_condition_dict = query.table_where_condition_dict
            conditions = next_spn.relevant_conditions(nominator_query,
                                                      merged_tables=next_merged_tables.union(overlapping_tables))
            multipliers = next_spn.compute_multipliers(nominator_query)

            nominator_expectation = IndicatorExpectation(multipliers, conditions, spn=next_spn,
                                                         table_set=next_merged_tables.union(overlapping_tables))

            # we can still exploit the outgoing multiplier if the multiplier is present
            if incoming_relationship and exploit_incoming_multipliers and len(overlapping_tables) == 0:
                nominator_expectation.nominator_multipliers \
                    .append((next_neighbour, relationship_obj.multiplier_attribute_name))

            factors.append(nominator_expectation)

            # denominator: indicator expectation of overlap
            denominator_query = Query(self.schema_graph)
            for relationship in overlapping_relationships:
                denominator_query.add_join_condition(relationship)
            denominator_query.table_set.update(next_merged_tables)
            denominator_query.table_where_condition_dict = query.table_where_condition_dict

            # constraints for next neighbor would not have any impact otherwise
            conditions = next_spn.relevant_conditions(denominator_query, merged_tables=overlapping_tables)

            next_neighbour_obj = self.schema_graph.table_dictionary[next_neighbour]
            # add not null condition for next neighbor
            conditions.append((next_neighbour, next_neighbour_obj.table_nn_attribute + " IS NOT NULL"))
            multipliers = next_spn.compute_multipliers(denominator_query)
            denominator_exp = IndicatorExpectation(multipliers, conditions, spn=next_spn, inverse=True,
                                                   table_set=overlapping_tables)

            # we can still exploit the outgoing multiplier if the multiplier is present
            if incoming_relationship and exploit_incoming_multipliers and len(overlapping_tables) == 0:
                denominator_exp.nominator_multipliers \
                    .append((next_neighbour, relationship_obj.multiplier_attribute_name))
            factors.append(denominator_exp)

            # mark tables as merged, remove merged relationships
            for table in next_merged_tables:
                merged_tables.add(table)
                corresponding_exp_dict[table] = nominator_expectation

            query.relationship_set -= set(next_mergeable_relationships)

        values, cardinality, formula = evaluate_factors(dry_run, factors, self.cached_expecation_vals,
                                                        gen_code_stats=gen_code_stats)

        if not return_factor_values:
            return formula, factors, cardinality
        else:
            return formula, factors, cardinality, values

    def _greedily_select_next_table(self, original_query, query, next_neighbours, exploit_overlapping, merged_tables,
                                    rdc_spn_selection=False, rdc_attribute_dict=None, prefer_disjunct=False):
        """
        Compute possible next merges and select greedily.
        """
        next_spn = None
        next_neighbour = None
        next_mergeable_relationships = None
        current_best_candidate_vector = None

        for spn in self.spns:

            if len(spn.table_set.intersection(merged_tables)) > 0 and prefer_disjunct:
                continue

            possible_neighbours = spn.table_set.intersection(next_neighbours)

            # for one SPN we can have several starting points
            for neighbour in possible_neighbours:

                # plus 1 because we can also merge edge directing to neighbour
                mergeable_relationships = spn.compute_mergeable_relationships(query, neighbour)
                no_mergeable_relationships = len(mergeable_relationships) + 1

                mergeable_tables = self._merged_tables(mergeable_relationships)
                mergeable_tables.add(neighbour)

                where_condition_tables = set(query.table_where_condition_dict.keys()).intersection(mergeable_tables)
                unnecessary_tables = len(spn.table_set) - len(mergeable_tables)

                if not exploit_overlapping:
                    current_candidate_vector = (len(where_condition_tables), no_mergeable_relationships,
                                                -unnecessary_tables)

                else:
                    # find overlapping relationships (relationships already merged that also appear in next_spn)
                    _, overlapping_tables, no_overlapping_conditions = self._compute_overlap(
                        next_neighbour, query, original_query, mergeable_relationships, mergeable_tables, spn)

                    unnecessary_tables = len(spn.table_set.difference(mergeable_tables).difference(overlapping_tables))
                    current_candidate_vector = (len(where_condition_tables), no_mergeable_relationships,
                                                no_overlapping_conditions, -unnecessary_tables)

                # if rdc based selection is active we should this should be the first part of the candidate vector
                if rdc_spn_selection:
                    # find attributes with where conditions
                    rdc_sum = self.merged_rdc_sum(mergeable_tables, query, rdc_attribute_dict)
                    current_candidate_vector = (rdc_sum,) + current_candidate_vector

                if current_best_candidate_vector is None or \
                        current_candidate_vector > current_best_candidate_vector:
                    next_spn = spn
                    next_neighbour = neighbour
                    next_mergeable_relationships = mergeable_relationships
                    current_best_candidate_vector = current_candidate_vector

        if next_spn is None:
            # recursive call with prefer false because there is no disjunct candidate
            return self._greedily_select_next_table(original_query, query, next_neighbours, exploit_overlapping,
                                                    merged_tables, prefer_disjunct=False)

        return next_spn, next_neighbour, next_mergeable_relationships

    def merged_rdc_sum(self, mergeable_tables, query, rdc_attribute_dict):
        merged_where_columns = set()
        for table, conditions in query.table_where_condition_dict.items():
            if table not in mergeable_tables:
                continue
            for condition in conditions:
                column = infer_column(condition)
                merged_where_columns.add(table + '.' + column)
        rdc_sum = sum([rdc_attribute_dict[column_combination]
                       for column_combination in itertools.combinations(list(merged_where_columns), 2)
                       if rdc_attribute_dict.get(column_combination) is not None])
        return rdc_sum

    def _greedily_select_expectation_spn(self, query, features):
        """
        Select first SPN by maximization of applicable where selections.
        """

        max_where_conditions = -1
        first_spn = None
        expectation = None

        for spn in self.spns:
            # if spn contains all features consider it a candidate
            features_col_names = set([table + '.' + feature for table, feature in features])
            if len(features_col_names.difference(spn.column_names)) == 0:

                where_conditions = set(query.table_where_condition_dict.keys()).intersection(spn.table_set)

                if len(where_conditions) > max_where_conditions:
                    first_spn = spn
                    conditions = spn.relevant_conditions(query)
                    normalizing_multipliers = spn.compute_multipliers(query)
                    expectation = Expectation(features, normalizing_multipliers, conditions, spn=first_spn)

        assert first_spn is not None, "Did not find SPN offering all features"

        return first_spn, expectation

    def _greedily_select_first_cardinality_spn(self, query, rdc_spn_selection=False, rdc_attribute_dict=None):
        """
        Select first SPN by maximization of applicable where selections.
        """
        first_spn = None
        next_mergeable_relationships = None
        next_mergeable_tables = None
        current_best_candidate_vector = None

        for spn in self.spns:
            # to get mergeable relationships we could use
            # intersection_relationships = query.relationship_set.intersection(spn.relationship_set)
            # However, this does not work if mergeable relationships are not connected
            for start_table in spn.table_set:
                if start_table not in query.table_set:
                    continue

                mergeable_relationships = spn.compute_mergeable_relationships(query, start_table)
                no_mergeable_relationships = len(mergeable_relationships) + 1

                mergeable_tables = self._merged_tables(mergeable_relationships)
                mergeable_tables.add(start_table)

                where_conditions = set(query.table_where_condition_dict.keys()).intersection(mergeable_tables)
                unnecessary_tables = len(spn.table_set.difference(query.table_set))

                current_candidate_vector = (len(where_conditions), no_mergeable_relationships, -unnecessary_tables)

                if rdc_spn_selection:
                    rdc_sum = self.merged_rdc_sum(mergeable_tables, query, rdc_attribute_dict)
                    current_candidate_vector = (rdc_sum,) + current_candidate_vector

                if current_best_candidate_vector is None or current_candidate_vector > current_best_candidate_vector:
                    current_best_candidate_vector = current_candidate_vector
                    first_spn = spn
                    next_mergeable_relationships = mergeable_relationships
                    next_mergeable_tables = mergeable_tables

        return first_spn, next_mergeable_relationships, next_mergeable_tables

    def _possible_first_spns(self, query):
        """
        Select possible first spns.
        """

        possible_starts = []

        for spn in self.spns:
            considered_start_tables = set()
            for start_table in spn.table_set.intersection(query.table_set):

                # e.g. customer and order part of same spn
                if start_table in considered_start_tables:
                    continue
                mergeable_relationships = spn.compute_mergeable_relationships(query, start_table)
                mergeable_tables = self._merged_tables(mergeable_relationships)
                mergeable_tables.add(start_table)
                no_where_conditions = len(set(query.table_where_condition_dict.keys()).intersection(mergeable_tables))

                if no_where_conditions == 0:
                    continue

                if start_table in query.table_set:
                    mergeable_tables.add(start_table)
                considered_start_tables.update(mergeable_tables)

                possible_starts.append((spn, mergeable_relationships, mergeable_tables))

        return possible_starts

    def _merged_tables(self, mergeable_relationships):
        """
        Compute merged tables if different relationships are merged.
        """

        merged_tables = set()

        for relationship in mergeable_relationships:
            relationship_obj = self.schema_graph.relationship_dictionary[relationship]
            merged_tables.add(relationship_obj.start)
            merged_tables.add(relationship_obj.end)

        return merged_tables

    def _next_neighbours(self, query, merged_tables):
        """
        List tables which have direct edge to already merged tables. Should be merged in next step.
        """

        next_neighbours = set()
        neighbours_relationship_dict = {}

        for relationship in query.relationship_set:
            relationship_obj = self.schema_graph.relationship_dictionary[relationship]
            if relationship_obj.start in merged_tables and \
                    relationship_obj.end not in merged_tables:

                neighbour = relationship_obj.end
                next_neighbours.add(neighbour)
                neighbours_relationship_dict[neighbour] = relationship

            elif relationship_obj.end in merged_tables and \
                    relationship_obj.start not in merged_tables:

                neighbour = relationship_obj.start
                next_neighbours.add(neighbour)
                neighbours_relationship_dict[neighbour] = relationship

        return next_neighbours, neighbours_relationship_dict

    def _compute_overlap(self, next_neighbour, query, original_query, next_mergeable_relationships, next_merged_tables,
                         next_spn):
        """
        Find overlapping relationships (relationships already merged that also appear in next_spn)

        :param next_neighbour:
        :param original_query:
        :param next_mergeable_relationships:
        :param next_spn:
        :return:
        """
        overlapping_relationships = set()
        overlapping_tables = {next_neighbour}
        new_overlapping_table = True
        while new_overlapping_table:
            new_overlapping_table = False
            for relationship_obj in self.schema_graph.relationships:
                if relationship_obj.identifier in original_query.relationship_set \
                        and relationship_obj.identifier not in overlapping_relationships \
                        and relationship_obj.identifier not in next_mergeable_relationships \
                        and relationship_obj.identifier in next_spn.relationship_set:
                    if relationship_obj.start in overlapping_tables \
                            and relationship_obj.end not in overlapping_tables:
                        new_overlapping_table = True
                        overlapping_tables.add(relationship_obj.end)
                        overlapping_relationships.add(relationship_obj.identifier)
                    elif relationship_obj.start not in overlapping_tables \
                            and relationship_obj.end in overlapping_tables:
                        new_overlapping_table = True
                        overlapping_tables.add(relationship_obj.start)
                        overlapping_relationships.add(relationship_obj.identifier)

        # overlapping conditions
        no_overlapping_conditions = len(set(query.table_where_condition_dict.keys())
                                        .intersection(overlapping_tables.difference(next_merged_tables)))

        return overlapping_relationships, overlapping_tables, no_overlapping_conditions
