import csv
import logging
import pickle
from time import perf_counter

import math
import scipy

from ensemble_compilation.graph_representation import AggregationType
from ensemble_compilation.spn_ensemble import read_ensemble, logger
from evaluation.utils import parse_query, all_operations_of_type, save_csv

logger = logging.getLogger(__name__)


def evaluate_confidence_intervals(ensemble_location, query_filename, target_path, schema, ground_truth_path,
                                  confidence_sample_size, rdc_spn_selection, pairwise_rdc_path,
                                  max_variants=5, merge_indicator_exp=False,
                                  exploit_overlapping=False, min_sample_ratio=0, sample_size=10000000,
                                  true_result_upsampling_factor=300):  # 100
    """
    Loads ensemble and computes metrics for confidence interval evaluation
    :param ensemble_location:
    :param query_filename:
    :param target_csv_path:
    :param schema:
    :param max_variants:
    :param merge_indicator_exp:
    :param exploit_overlapping:
    :param min_sample_ratio:
    :return:
    """

    spn_ensemble = read_ensemble(ensemble_location, build_reverse_dict=True)
    csv_rows = []

    # read all queries
    with open(query_filename) as f:
        queries = f.readlines()
    # read ground truth
    with open(ground_truth_path, 'rb') as handle:
        ground_truth = pickle.load(handle)

    for query_no, query_str in enumerate(queries):

        query_str = query_str.strip()
        logger.info(f"Evaluating the confidence intervals for query {query_no}: {query_str}")

        query = parse_query(query_str.strip(), schema)
        aqp_start_t = perf_counter()
        confidence_intervals, aqp_result = spn_ensemble.evaluate_query(query, rdc_spn_selection=rdc_spn_selection,
                                                                       pairwise_rdc_path=pairwise_rdc_path,
                                                                       merge_indicator_exp=merge_indicator_exp,
                                                                       max_variants=max_variants,
                                                                       exploit_overlapping=exploit_overlapping,
                                                                       debug=False,
                                                                       confidence_intervals=True,
                                                                       confidence_sample_size=confidence_sample_size)
        aqp_end_t = perf_counter()
        latency = aqp_end_t - aqp_start_t
        logger.info(f"\t\t{'total_time:':<32}{latency} secs")

        true_result = ground_truth[query_no]

        type_all_ops = None
        if all_operations_of_type(AggregationType.SUM, query):
            type_all_ops = AggregationType.SUM
        elif all_operations_of_type(AggregationType.AVG, query):
            type_all_ops = AggregationType.AVG
        elif all_operations_of_type(AggregationType.COUNT, query):
            type_all_ops = AggregationType.COUNT

        if isinstance(aqp_result, list):
            for result_row in true_result:
                group_by_attributes = result_row[:-3]
                matching_aqp_rows = [(matching_idx, aqp_row) for matching_idx, aqp_row in enumerate(aqp_result)
                                     if aqp_row[:-1] == group_by_attributes]
                assert len(matching_aqp_rows) <= 1, "Multiple possible group by attributes found."
                if len(matching_aqp_rows) == 1:
                    matching_idx, matching_aqp_row = matching_aqp_rows[0]
                    true_aggregate, std, count = result_row[-3:]

                    if count <= 1:
                        # std is not defined in this case
                        continue

                    interval = confidence_intervals[matching_idx]
                    aqp_std, true_std, relative_confidence_interval_error, true_result, aqp_aggregate = evaluate_stds(
                        matching_aqp_row[-1],
                        interval, count,
                        sample_size, std,
                        true_aggregate, type_all_ops,
                        true_result_upsampling_factor)

                    logger.debug(f"\t\taqp_std: {aqp_std}")
                    logger.debug(f"\t\ttrue_std: {true_std}")

                    csv_rows.append({'query_no': query_no,
                                     'latency': latency,
                                     'aqp_std': aqp_std,
                                     'aqp_aggregate': aqp_aggregate,
                                     'true_std': true_std,
                                     'true_aggregate': true_result,
                                     'count': count,
                                     'relative_confidence_interval_error': relative_confidence_interval_error
                                     })
        else:
            true_aggregate, std, count = true_result[0][-3:]

            aqp_std, true_std, relative_confidence_interval_error, true_result, aqp_aggregate = evaluate_stds(
                aqp_result, confidence_intervals,
                count, sample_size, std,
                true_aggregate,
                type_all_ops,
                true_result_upsampling_factor)
            logger.debug(f"\t\taqp_std: {aqp_std}")
            logger.debug(f"\t\ttrue_std: {true_std}")

            csv_rows.append({'query_no': query_no,
                             'latency': latency,
                             'aqp_std': aqp_std,
                             'aqp_aggregate': aqp_aggregate,
                             'true_std': true_std,
                             'true_aggregate': true_result,
                             'count': count,
                             'relative_confidence_interval_error': relative_confidence_interval_error
                             })

    save_csv(csv_rows, target_path)


def evaluate_stds(aqp_result, confidence_intervals, count, sample_size, std, true_result, type_all_ops,
                  true_result_upsampling_factor):
    std = float(std)
    count = float(count)
    true_result = float(true_result)
    confidence_upper_bound = confidence_intervals[1]
    ci_length = confidence_upper_bound - aqp_result
    aqp_std = ci_length  # / scipy.stats.norm.ppf(0.95)
    if type_all_ops == AggregationType.AVG:
        # for normal random variable std/sqrt(n)
        true_std = std / math.sqrt(count)

    elif type_all_ops == AggregationType.COUNT:
        # for bernoulli: sqrt(n*p*(1-p))

        bernoulli_p = count / sample_size
        true_std = math.sqrt(sample_size * bernoulli_p * (1 - bernoulli_p)) * true_result_upsampling_factor
        true_result *= true_result_upsampling_factor

    elif type_all_ops == AggregationType.SUM:
        # model sum as product of 1_c * X

        bernoulli_p = count / sample_size
        bernoulli_std = math.sqrt(sample_size * bernoulli_p * (1 - bernoulli_p))

        rv_exp = true_result / count
        rv_std = std / math.sqrt(count)

        true_std = math.sqrt((bernoulli_std ** 2 + bernoulli_p ** 2) * (rv_std ** 2 + rv_exp ** 2) -
                             bernoulli_p ** 2 * rv_exp ** 2) * true_result_upsampling_factor
        true_result *= true_result_upsampling_factor

    true_std *= scipy.stats.norm.ppf(0.95)
    relative_confidence_interval_error = abs(aqp_std - true_std) / true_result
    return aqp_std, true_std, relative_confidence_interval_error, true_result, aqp_result
