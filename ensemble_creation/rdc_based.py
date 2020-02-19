import itertools
import logging
import pickle
from random import randint
from time import perf_counter

import networkx as nx
import numpy as np
from spn.algorithms.splitting.RDC import rdc_cca, rdc_transformer
from spn.structure.Base import Context

from aqp_spn.aqp_spn import AQPSPN
from data_preparation.join_data_preparation import JoinDataPreparator
from ensemble_compilation.physical_db import DBConnection
from ensemble_compilation.spn_ensemble import SPNEnsemble
from ensemble_creation.naive import RATIO_MIN_INSTANCE_SLICE
from ensemble_creation.utils import create_random_join

logger = logging.getLogger(__name__)


def generate_candidate_solution(pairwise_max_rdc, table_index_dict, prep, max_budget, schema, max_no_relationships,
                                rdc_threshold):
    spn_relationships_list = set()
    all_merged_tables = set()
    learning_costs = 0

    # Basis of either binary or single SPNs
    # create every relationship above threshold
    for relationship_obj in schema.relationships:
        relationship_list = [relationship_obj.identifier]
        merged_tables = [relationship_obj.start, relationship_obj.end]

        if candidate_rdc_sum_means(pairwise_max_rdc, table_index_dict,
                                   [(relationship_list, merged_tables)]) > rdc_threshold:
            # learning_costs += learning_cost(prep, [relationship_list])
            all_merged_tables.update(merged_tables)
            spn_relationships_list.add((frozenset(relationship_list), frozenset(merged_tables)))

    # add remaining single tables
    for table in set([table.table_name for table in schema.tables]).difference(all_merged_tables):
        # learning_costs += learning_cost(prep, None, single_table=table)
        spn_relationships_list.add((frozenset(), frozenset([table])))

    # In addition randomly select larger joins
    rejected_candidates = 0
    while rejected_candidates < 5:
        no_joins = randint(2, max_no_relationships)
        relationship_list, merged_tables = create_random_join(schema, no_joins)
        current_costs = learning_cost(prep, [relationship_list])

        # Already in ensemble
        if (frozenset(relationship_list), frozenset(merged_tables)) in spn_relationships_list:
            rejected_candidates += 1
            continue

        # does not offer any benefit
        if candidate_rdc_sum_means(pairwise_max_rdc, table_index_dict,
                                   [(relationship_list, merged_tables)]) <= rdc_threshold:
            rejected_candidates += 1
            continue

        # Can not be added because of budget
        if learning_costs + current_costs > max_budget:
            break

        # Can be added
        all_merged_tables.update(merged_tables)
        learning_costs += current_costs
        spn_relationships_list.add((frozenset(relationship_list), frozenset(merged_tables)))

    return frozenset(spn_relationships_list), learning_costs


def learning_cost(prep, spn_relationships_list, single_table=None):
    if single_table is not None:
        assert spn_relationships_list is None, "Specify either single table or relationship list"
        return prep.column_number(single_table=single_table) ** 2

    # estimate learning costs
    learning_cost_estimate = sum(
        [prep.column_number(relationship_list=relationship_list) ** 2 for relationship_list in spn_relationships_list])

    return learning_cost_estimate


def candidate_rdc_sum_means(pairwise_max_rdc, table_index_dict, ensemble_candidate):
    # find pairs of tables
    rdc_mean_sum = 0
    for relationship_list, merged_tables in ensemble_candidate:
        if len(relationship_list) == 0:
            continue
        included_table_idxs = [table_index_dict[table] for table in merged_tables]
        rdc_vals = [pairwise_max_rdc[(min(left_idx, right_idx), max(left_idx, right_idx))] for left_idx, right_idx
                    in itertools.combinations(included_table_idxs, 2)]
        rdc_mean_sum += sum(rdc_vals) / len(rdc_vals)

    return rdc_mean_sum


def candidate_evaluation(schema, meta_data_path, sample_size, spn_sample_size, max_table_data, ensemble_path,
                         physical_db_name, postsampling_factor, ensemble_budget_factor, max_no_joins, rdc_learn,
                         pairwise_rdc_path, rdc_threshold=0.15, random_solutions=10000, bloom_filters=False,
                         incremental_learning_rate=0, incremental_condition=None):

    assert incremental_learning_rate==0 or incremental_condition is None
    prep = JoinDataPreparator(meta_data_path + "/meta_data_sampled.pkl", schema, max_table_data=max_table_data)

    # build graph from schema
    table_index_dict = {table.table_name: i for i, table in enumerate(schema.tables)}
    inverse_table_index_dict = {table_index_dict[k]: k for k in table_index_dict.keys()}
    G = nx.Graph()
    for relationship in schema.relationships:
        start_idx = table_index_dict[relationship.start]
        end_idx = table_index_dict[relationship.end]
        G.add_edge(start_idx, end_idx, relationship=relationship)

    # iterate over pairs
    all_pairs = dict(nx.all_pairs_shortest_path(G))
    all_pair_list = []
    for left_idx, right_idx_dict in all_pairs.items():
        for right_idx, shortest_path_list in right_idx_dict.items():
            if left_idx >= right_idx:
                continue
            all_pair_list.append((left_idx, right_idx, shortest_path_list,))

    # sort by length of path
    # for every pair: compute rdc value
    rdc_attribute_dict = dict()
    rdc_start_t = perf_counter()
    all_pair_list.sort(key=lambda x: len(x[2]))
    pairwise_max_rdc = {comb: 0 for comb in itertools.combinations(range(len(table_index_dict)), 2)}
    for left_idx, right_idx, shortest_path_list in all_pair_list:
        left_table = inverse_table_index_dict[left_idx]
        right_table = inverse_table_index_dict[right_idx]
        logger.debug(f"Evaluating {left_table} and {right_table}")
        relationship_list = [G[shortest_path_list[i]][shortest_path_list[i + 1]]['relationship'].identifier
                             for i in range(len(shortest_path_list) - 1)]
        df_samples, meta_types, _ = prep.generate_join_sample(
            relationship_list=relationship_list,
            min_start_table_size=1, sample_rate=1.0,
            drop_redundant_columns=True,
            max_intermediate_size=sample_size * postsampling_factor[0])
        rdc_value = max_rdc(schema, left_table, right_table, df_samples, meta_types, rdc_attribute_dict)
        if rdc_value > rdc_threshold:
            pairwise_max_rdc[(left_idx, right_idx)] = rdc_value

        logger.debug(f"Max RDC between {left_table} and {right_table}: {pairwise_max_rdc[(left_idx, right_idx)]}")
    logger.info(f"Computed {len(all_pair_list)} rdc values in {perf_counter() - rdc_start_t} secs")

    # save pairwise rdc values
    with open(pairwise_rdc_path, 'wb') as f:
        pickle.dump(rdc_attribute_dict, f, pickle.HIGHEST_PROTOCOL)

    # evaluate the budgets
    eval_start_t = perf_counter()

    # maximum budget relative to costs of creating SPN for all single tables
    budget = ensemble_budget_factor * sum(
        [prep.column_number(single_table=table.table_name) ** 2 for table in schema.tables])

    # generate random candidates
    ensemble_candidates = set([generate_candidate_solution(pairwise_max_rdc, table_index_dict, prep, budget, schema,
                                                           max_no_joins, rdc_threshold) for i in
                               range(random_solutions)])

    candidates = [(ensemble_candidate[0], ensemble_candidate[1],
                   candidate_rdc_sum_means(pairwise_max_rdc, table_index_dict, ensemble_candidate[0])) for
                  ensemble_candidate in ensemble_candidates]
    # sort by rdc value, then by learning costs
    candidates.sort(key=lambda x: (-x[2], x[1]))
    optimal_candidate = list(candidates[0][0])

    # learn large joins first
    optimal_candidate.sort(key=lambda x: -len(x[1]))
    logger.info(f"Computed optimal solution out of {random_solutions} candidates "
                f"in {perf_counter() - eval_start_t} secs")
    for _, merged_tables in optimal_candidate:
        spn_tables = " - ".join(merged_tables)
        logger.info(f"\t\t{spn_tables}")

    spn_ensemble = SPNEnsemble(schema)
    prep = JoinDataPreparator(meta_data_path + "/meta_data.pkl", schema, max_table_data=max_table_data)
    if physical_db_name is not None:
        db_connection = DBConnection(db=physical_db_name)

    for relationship_list, merged_tables in optimal_candidate:
        logger.info(f"Learning SPN for {str(relationship_list)}.")

        # compute join sample
        join_start_t = perf_counter()
        logger.debug(f"Using postsampling_factor {postsampling_factor[len(merged_tables) - 1]}.")
        logger.debug(f"Using spn_sample_size {spn_sample_size[len(merged_tables) - 1]}.")
        if len(relationship_list) > 0:
            df_samples, df_inc_samples, meta_types, null_values, full_join_est = prep.generate_n_samples_with_incremental_part(
                spn_sample_size[len(merged_tables) - 1],
                relationship_list=list(relationship_list),
                post_sampling_factor=postsampling_factor[len(merged_tables) - 1],
                incremental_learning_rate=incremental_learning_rate,
                incremental_condition=incremental_condition)
        else:
            assert len(merged_tables) == 1
            df_samples, df_inc_samples, meta_types, null_values, full_join_est = prep.generate_n_samples_with_incremental_part(
                spn_sample_size[len(merged_tables) - 1],
                single_table=list(merged_tables)[0],
                post_sampling_factor=postsampling_factor[len(merged_tables) - 1],
                incremental_learning_rate=incremental_learning_rate,
                incremental_condition=incremental_condition)
        logger.info(f"Computed join for {str(relationship_list)}, (tables: {str(merged_tables)}) "
                    f"in {perf_counter() - join_start_t} secs")
        logging.info(f"card(learning): {len(df_samples)}, card(df_inc_samples): {len(df_inc_samples)}")
        if not incremental_condition is None:
            condition_percentage  = int(100.0 * len(df_inc_samples)/len(df_samples))
            logger.info(f"set incremental_learning_rate to {condition_percentage}%, based on condition {incremental_condition}")

        # cardinality
        if physical_db_name is not None:
            true_card_start_t = perf_counter()
            if len(relationship_list) > 0:
                where_cond = " AND ".join([relationship for relationship in relationship_list])
                table_list = ", ".join(prep.corresponding_tables(relationship_list))
                sql_query = f"SELECT COUNT(*) FROM {table_list} WHERE {where_cond}"
            else:
                sql_query = f"SELECT COUNT(*) FROM {list(merged_tables)[0]}"
            cardinality_true = db_connection.get_result(sql_query)
            logger.info(f"Computed full join size {cardinality_true} in {perf_counter() - true_card_start_t} secs")
            logger.info(f"Predicted full join size of {full_join_est} but real join size was {cardinality_true}.")

            if incremental_condition is not None:
                db_connection = DBConnection(db=physical_db_name)
                sql_percentage_titles = """select 100.0 - 100.0*count(*)/(select count(*) 
									                                from aka_title 
									                               where production_year is not null) percentage 
                                             from aka_title title
                                            where """ + incremental_condition
                percentage = db_connection.get_result(sql_percentage_titles)
                logging.debug(f"sql (for incremental learning_rate calculation): {sql_percentage_titles}: result: {percentage}")
                # incremental_learning_rate = percentage
            else:
                percentage = incremental_learning_rate

        else:
            cardinality_true = full_join_est

        # learn spn
        if len(relationship_list) > 0:
            aqp_spn = AQPSPN(meta_types, null_values, cardinality_true, schema,
                             list(relationship_list), full_sample_size=len(df_samples),
                             column_names=list(df_samples.columns), table_meta_data=prep.table_meta_data)
        else:
            aqp_spn = AQPSPN(meta_types, null_values, cardinality_true, schema,
                             [], full_sample_size=len(df_samples), table_set={list(merged_tables)[0]},
                             column_names=list(df_samples.columns), table_meta_data=prep.table_meta_data)

        min_instance_slice = RATIO_MIN_INSTANCE_SLICE * min(spn_sample_size[len(merged_tables) - 1], len(df_samples))
        logger.debug(f"Using {len(df_samples)} samples.")
        logger.debug(f"Using min_instance_slice parameter {min_instance_slice}.")
        logger.info(f"SPN training phase with {len(df_samples)} samples")
        spn_learn_start_t = perf_counter()
        aqp_spn.learn(df_samples.values, min_instances_slice=min_instance_slice, bloom_filters=bloom_filters,
                      rdc_threshold=rdc_learn)
        spn_learn_end_t = perf_counter()
        omit_incremental = False
        spn_inc_learn_start_t=0
        spn_inc_learn_end_t=0
        if (incremental_learning_rate>0 or incremental_condition is not None):
            if (omit_incremental):
                logger.info(f"no additional incremental SPN training phase with {len(df_inc_samples)} samples ")
            else:
                logger.info(f"additional incremental SPN training phase with {len(df_inc_samples)} samples ")

                spn_inc_learn_start_t = perf_counter()
                aqp_spn.learn_incremental(df_inc_samples.values)
                spn_inc_learn_end_t = perf_counter()
        logging.info(f"learning time:{round(spn_learn_end_t-spn_learn_start_t,2)} ({len(df_samples)} datasets), incremental learning time: {round(spn_inc_learn_end_t-spn_inc_learn_start_t,2)} ({len(df_inc_samples)} datasets), incremental_condition: {incremental_condition}, incremental-learning-rate: {percentage}% [TIME]")
        spn_ensemble.add_spn(aqp_spn)

    if incremental_learning_rate and omit_incremental:
        ensemble_path += f'/ensemble_join_{max_no_joins}_budget_{ensemble_budget_factor}_{spn_sample_size[0]}_only_{int(100-incremental_learning_rate)}_percent.pkl'
    else:
        ensemble_path += f'/ensemble_join_{max_no_joins}_budget_{ensemble_budget_factor}_{spn_sample_size[0]}.pkl'
    logger.info(f"Saving ensemble to {ensemble_path}")
    spn_ensemble.save(ensemble_path)


def max_rdc(schema, left_table, right_table, df_samples, meta_types, rdc_attribute_dict,
            max_sampling_threshold_rows=10000, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, debug=True):
    # only keep columns of left or right table
    irrelevant_cols = []
    relevant_meta_types = []
    for i, column in enumerate(df_samples.columns):
        not_of_left_or_right = not (column.startswith(left_table + '.') or column.startswith(right_table + '.'))
        is_nn_attribute = (column == left_table + '.' + schema.table_dictionary[left_table].table_nn_attribute) or \
                          (column == right_table + '.' + schema.table_dictionary[right_table].table_nn_attribute)
        is_multiplier = False
        is_fk_field = False
        for relationship_obj in schema.relationships:  # [relationship_obj_list[0], relationship_obj_list[-1]]
            if relationship_obj.end + '.' + relationship_obj.end_attr == column or \
                    relationship_obj.start + '.' + relationship_obj.start_attr == column:
                is_fk_field = True
                break

            if relationship_obj.end + '.' + relationship_obj.multiplier_attribute_name_nn == column or \
                    relationship_obj.end + '.' + relationship_obj.multiplier_attribute_name == column:
                is_multiplier = True
                break
        is_uninformative = False

        if not_of_left_or_right or is_nn_attribute or is_multiplier or is_fk_field or is_uninformative:
            irrelevant_cols.append(column)
        else:
            relevant_meta_types.append(meta_types[i])

    df_samples.drop(columns=irrelevant_cols, inplace=True)

    left_column_names = [(i, column) for i, column in enumerate(df_samples.columns) if
                         column.startswith(left_table + '.')]
    right_column_names = [(i, column) for i, column in enumerate(df_samples.columns) if
                          column.startswith(right_table + '.')]
    left_columns = [i for i, column in left_column_names]
    right_columns = [i for i, column in right_column_names]

    data = df_samples.values
    # sample if necessary
    if data.shape[0] > max_sampling_threshold_rows:
        data = data[np.random.randint(data.shape[0], size=max_sampling_threshold_rows), :]

    n_features = data.shape[1]
    assert n_features == len(relevant_meta_types)

    ds_context = Context(meta_types=relevant_meta_types)
    ds_context.add_domains(data)

    rdc_features = rdc_transformer(
        data, relevant_meta_types, ds_context.domains, k=k, s=s, non_linearity=non_linearity, return_matrix=False
    )
    pairwise_comparisons = [(i, j) for i in left_columns for j in right_columns]

    from joblib import Parallel, delayed

    rdc_vals = Parallel(n_jobs=n_jobs, max_nbytes=1024, backend="threading")(
        delayed(rdc_cca)((i, j, rdc_features)) for i, j in pairwise_comparisons
    )

    for (i, j), rdc in zip(pairwise_comparisons, rdc_vals):
        if np.isnan(rdc):
            rdc = 0
        if debug:
            logger.debug(f"{df_samples.columns[i]}, {df_samples.columns[j]}: {rdc}")

    pairwise_comparisons = [(column_left, column_right) for i, column_left in left_column_names for j, column_right in
                            right_column_names]
    for (column_left, column_right), rdc in zip(pairwise_comparisons, rdc_vals):
        rdc_attribute_dict[(column_left, column_right)] = rdc
        rdc_attribute_dict[(column_right, column_left)] = rdc

    return max(rdc_vals)
