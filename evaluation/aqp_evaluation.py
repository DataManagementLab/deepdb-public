import csv
import logging
import pickle
from enum import Enum
from time import perf_counter
import numpy as np

import math

from ensemble_compilation.physical_db import DBConnection
from ensemble_compilation.spn_ensemble import read_ensemble
from evaluation.utils import parse_query, save_csv

logger = logging.getLogger(__name__)


class ApproachType(Enum):
    MODEL_BASED = 0
    TABLESAMPLE = 1
    VERDICT_DB = 2
    APPROXIMATE_DB = 3
    WAVELET = 4
    STRATIFIED_SAMPLING = 5


def compute_ground_truth(target_path, physical_db_name, vacuum=False, query_filename=None, query_list=None):
    """
    Queries database for each query and stores result rows in dictionary.
    :param query_filename: where to take queries from
    :param target_path: where to store dictionary
    :param physical_db_name: name of the database
    :return:
    """

    db_connection = DBConnection(db=physical_db_name)
    # read all queries
    if query_list is not None:
        queries = query_list
    elif query_filename is not None:
        with open(query_filename) as f:
            queries = f.readlines()
    else:
        raise ValueError("Either query_list or query_filename have to be given")

    ground_truth = dict()
    ground_truth_times = dict()

    for query_no, query_str in enumerate(queries):

        query_str = query_str.strip()
        logger.debug(f"Computing ground truth for AQP query {query_no}: {query_str}")

        aqp_start_t = perf_counter()
        query_str = query_str.strip()
        rows = db_connection.get_result_set(query_str)
        ground_truth[query_no] = rows
        aqp_end_t = perf_counter()
        ground_truth_times[query_no] = aqp_end_t - aqp_start_t
        logger.info(f"\t\ttotal time for query execution: {aqp_end_t - aqp_start_t} secs")

        if vacuum:
            vacuum_start_t = perf_counter()
            db_connection.vacuum()
            vacuum_end_t = perf_counter()
            logger.info(f"\t\tvacuum time: {vacuum_end_t - vacuum_start_t} secs")

        dump_ground_truth(ground_truth, ground_truth_times, target_path)
    dump_ground_truth(ground_truth, ground_truth_times, target_path)


def dump_ground_truth(ground_truth, ground_truth_times, target_path):
    with open(target_path, 'wb') as f:
        logger.debug(f"\t\tSaving ground truth dictionary to {target_path}")
        pickle.dump(ground_truth, f, pickle.HIGHEST_PROTOCOL)
    with open(target_path + '_times.pkl', 'wb') as f:
        logger.debug(f"\t\tSaving ground truth dictionary times to {target_path}")
        pickle.dump(ground_truth_times, f, pickle.HIGHEST_PROTOCOL)


def compute_relative_error(true, predicted, debug=False):
    true = float(true)
    predicted = float(predicted)
    relative_error = (true - predicted) / true
    if debug:
        logger.debug(f"\t\tpredicted     : {predicted:.2f}")
        logger.debug(f"\t\ttrue          : {true:.2f}")
        logger.debug(f"\t\trelative_error: {100 * relative_error:.2f}%")
    return abs(relative_error)


def evaluate_aqp_queries(ensemble_location, query_filename, target_path, schema, ground_truth_path,
                         rdc_spn_selection, pairwise_rdc_path, max_variants=5, merge_indicator_exp=False,
                         exploit_overlapping=False, min_sample_ratio=0, debug=False,
                         show_confidence_intervals=True):
    """
    Loads ensemble and computes metrics for AQP query evaluation
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
        logger.info(f"Evaluating AQP query {query_no}: {query_str}")

        query = parse_query(query_str.strip(), schema)
        aqp_start_t = perf_counter()
        confidence_intervals, aqp_result = spn_ensemble.evaluate_query(query, rdc_spn_selection=rdc_spn_selection,
                                                                       pairwise_rdc_path=pairwise_rdc_path,
                                                                       merge_indicator_exp=merge_indicator_exp,
                                                                       max_variants=max_variants,
                                                                       exploit_overlapping=exploit_overlapping,
                                                                       debug=debug,
                                                                       confidence_intervals=show_confidence_intervals)
        aqp_end_t = perf_counter()
        latency = aqp_end_t - aqp_start_t
        logger.info(f"\t\t{'total_time:':<32}{latency} secs")

        if ground_truth is not None:
            true_result = ground_truth[query_no]
            if isinstance(aqp_result, list):

                average_relative_error, bin_completeness, false_bin_percentage, total_bins, \
                confidence_interval_precision, confidence_interval_length, _ = \
                    evaluate_group_by(aqp_result, true_result, confidence_intervals)

                logger.info(f"\t\t{'total_bins: ':<32}{total_bins}")
                logger.info(f"\t\t{'bin_completeness: ':<32}{bin_completeness * 100:.2f}%")
                logger.info(f"\t\t{'average_relative_error: ':<32}{average_relative_error * 100:.2f}%")
                logger.info(f"\t\t{'false_bin_percentage: ':<32}{false_bin_percentage * 100:.2f}%")
                if show_confidence_intervals:
                    logger.info(
                        f"\t\t{'confidence_interval_precision: ':<32}{confidence_interval_precision * 100:>.2f}%")
                    logger.info(f"\t\t{'confidence_interval_length: ':<32}{confidence_interval_length * 100:>.2f}%")

            else:

                true_result = true_result[0][0]
                predicted_value = aqp_result

                logger.info(f"\t\t{'predicted:':<32}{predicted_value}")
                logger.info(f"\t\t{'true:':<32}{true_result}")
                # logger.info(f"\t\t{'confidence_interval:':<32}{confidence_intervals}")
                relative_error = compute_relative_error(true_result, predicted_value)
                logger.info(f"\t\t{'relative_error:':<32}{relative_error * 100:.2f}%")
                if show_confidence_intervals:
                    confidence_interval_precision, confidence_interval_length = evaluate_confidence_interval(
                        confidence_intervals,
                        true_result,
                        predicted_value)
                    logger.info(
                        f"\t\t{'confidence_interval_precision:':<32}{confidence_interval_precision * 100:>.2f}")
                    logger.info(f"\t\t{'confidence_interval_length: ':<32}{confidence_interval_length * 100:>.2f}%")
                total_bins = 1
                bin_completeness = 1
                average_relative_error = relative_error
            csv_rows.append({'approach': ApproachType.MODEL_BASED,
                             'query_no': query_no,
                             'latency': latency,
                             'average_relative_error': average_relative_error * 100,
                             'bin_completeness': bin_completeness * 100,
                             'total_bins': total_bins,
                             'query': query_str,
                             'sample_percentage': 100
                             })
        else:
            logger.info(f"\t\tpredicted: {aqp_result}")

    save_csv(csv_rows, target_path)


def evaluate_confidence_interval(confidence_interval, true_result, predicted):
    in_interval = 0
    if confidence_interval[0] <= true_result <= confidence_interval[1]:
        in_interval = 1
    relative_interval_size = (confidence_interval[1] - predicted) / predicted
    return in_interval, relative_interval_size


def evaluate_group_by(aqp_result, true_result, confidence_intervals, medians=False, debug=False):
    group_by_combinations_found = 0
    avg_relative_errors = []
    confidence_interval_precision = 0
    confidence_interval_length = 0

    for result_row in true_result:
        group_by_attributes = result_row[:-1]
        matching_aqp_rows = [(matching_idx, aqp_row) for matching_idx, aqp_row in enumerate(aqp_result)
                             if aqp_row[:-1] == group_by_attributes]
        assert len(matching_aqp_rows) <= 1, "Multiple possible group by attributes found."
        if len(matching_aqp_rows) == 1:
            matching_idx = matching_aqp_rows[0][0]
            matching_aqp_row = matching_aqp_rows[0][1]

            group_by_combinations_found += 1
            assert matching_aqp_row[:-1] == result_row[:-1]
            relative_error = compute_relative_error(result_row[-1], matching_aqp_row[-1], debug=debug)
            avg_relative_errors.append(relative_error)
            if confidence_intervals:
                in_interval, relative_interval_size = evaluate_confidence_interval(confidence_intervals[matching_idx],
                                                                                   result_row[-1],
                                                                                   matching_aqp_row[-1])
                confidence_interval_precision += in_interval
                confidence_interval_length += relative_interval_size

    bin_completeness = math.inf
    average_relative_error = math.inf
    false_bin_percentage = math.inf
    total_bins = len(true_result)
    if group_by_combinations_found > 0:
        bin_completeness = group_by_combinations_found / len(true_result)
        if not medians:
            average_relative_error = sum(avg_relative_errors) / group_by_combinations_found
        else:
            average_relative_error = np.median(avg_relative_errors)
        false_bin_percentage = (len(aqp_result) - group_by_combinations_found) / len(aqp_result)
        confidence_interval_precision /= group_by_combinations_found
        confidence_interval_length /= group_by_combinations_found

    max_error = math.inf if len(avg_relative_errors) == 0 else max(avg_relative_errors)
    return average_relative_error, bin_completeness, false_bin_percentage, total_bins, confidence_interval_precision, confidence_interval_length, max_error
