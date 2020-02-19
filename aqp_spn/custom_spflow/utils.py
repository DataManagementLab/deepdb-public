import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


def default_slicer(data, cols, num_cond_cols=None):
    if num_cond_cols is None:
        if len(cols) == 1:
            return data[:, cols[0]].reshape((-1, 1))

        return data[:, cols]
    else:
        return np.concatenate((data[:, cols], data[:, -num_cond_cols:]), axis=1)


def compute_cartesian_product_completeness(col1, col2, ds_context, data, min_sample_size, max_sample_size,
                                           oversampling_cart_product=10, debug=False):
    """
    Compute how many distinct value combinations appear for pair of columns in data. A low value is an indicator for
    functional dependency or some different form of dependency.
    :param col1:
    :param col2:
    :param ds_context:
    :param data:
    :param min_sample_size:
    :param max_sample_size:
    :param oversampling_cart_product:
    :param debug:
    :return:
    """

    unique_tuples_start_t = time.perf_counter()
    len_cartesian_product = ds_context.no_unique_values[col1] * ds_context.no_unique_values[col2]
    sample_size = max(min(oversampling_cart_product * len_cartesian_product, max_sample_size), min_sample_size)

    sample_idx = np.random.randint(data.shape[0], size=sample_size)
    if sample_size < data.shape[0]:
        local_data_sample = data[sample_idx, :]
    else:
        local_data_sample = data
    value_combinations_sample = set(
        [(bin_data[0], bin_data[1],) for bin_data in
         default_slicer(local_data_sample, [col1, col2])])
    cartesian_product_completeness = len(value_combinations_sample) / len_cartesian_product
    unique_tuples_end_t = time.perf_counter()
    if debug:
        logging.debug(
            f"Computed unique combination set for scope ({col1}, {col2}) in "
            f"{unique_tuples_end_t - unique_tuples_start_t} sec.")
    return cartesian_product_completeness, value_combinations_sample, len_cartesian_product

