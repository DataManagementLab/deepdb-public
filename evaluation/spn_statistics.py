import csv
import logging
import os

from spn.structure.Base import Node, get_nodes_by_type

from ensemble_compilation.spn_ensemble import read_ensemble

logger = logging.getLogger(__name__)


def evaluate_spn_statistics(spn_path, target_csv_path, build_time_path):
    csv_list = []

    # SPN learn times
    for filename in os.listdir(spn_path):
        logger.debug(f'Reading {filename}')
        if not filename.startswith("ensemble") or filename.endswith('.zip'):
            continue

        spn_ensemble = read_ensemble(os.path.join(spn_path, filename))
        for spn in spn_ensemble.spns:
            num_nodes = len(get_nodes_by_type(spn.mspn, Node))
            upper_bound = 200 * len(spn.column_names) - 1
            # assert num_nodes <= upper_bound, "Num of nodes upper bound is wrong"
            csv_list.append((filename, spn.learn_time, spn.full_sample_size, spn.min_instances_slice, spn.rdc_threshold,
                             len(spn.relationship_set), len(spn.table_set),
                             " - ".join([table for table in spn.table_set]),
                             len(spn.column_names),
                             num_nodes,
                             upper_bound))

    # HDF create times
    with open(build_time_path) as f:
        hdf_preprocessing_time = int(f.readlines()[0])
        csv_list += [('generate_hdf', hdf_preprocessing_time, 0, 0, 0, 0, 0, "")]

    with open(target_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['filename', 'learn_time', 'full_sample_size', 'min_instances_slice', 'rdc_threshold', 'no_joins',
             'no_tables', 'tables', 'no_columns', 'structure_stats', 'upper_bound'])
        writer.writerows(csv_list)
