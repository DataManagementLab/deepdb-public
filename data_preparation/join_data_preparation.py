import copy
import logging
import pickle
import random

import dask.dataframe as dd
import gc
import math
import pandas as pd
from spn.structure.StatisticalTypes import MetaType

from data_preparation.prepare_single_tables import find_relationships
from ensemble_creation.utils import create_random_join

logger = logging.getLogger(__name__)


def prob_round(x):
    """
    Rounds up with probability proportional to decimal places.
    """

    floor = math.floor(x)
    if random.random() < x - floor:
        floor += 1
    return floor


class JoinDataPreparator:

    def __init__(self, meta_data_path, schema_graph, max_table_data=20000000, no_cache=True):
        self.meta_data_path = meta_data_path
        self.schema_graph = schema_graph
        with open(meta_data_path, 'rb') as handle:
            self.table_meta_data = pickle.load(handle)
        self.cached_tables = dict()
        self.max_table_data = max_table_data
        self.no_cache = no_cache

    def _find_start_table(self, relationship_list, min_start_table_size):
        """
        Finds start table for sampling.
        """

        table_dict = dict()
        table_set = set()

        def increase_table_in_dict(table):
            if table_dict.get(table) is None:
                table_dict[table] = 0
            else:
                table_dict[table] += 1

        for relationship in relationship_list:
            relationship_obj = self.schema_graph.relationship_dictionary.get(relationship)
            increase_table_in_dict(relationship_obj.end)
            table_set.add(relationship_obj.start)
            table_set.add(relationship_obj.end)

        sampled_tables = [table for table in table_set if self._sampling_rate(table) < 1]
        assert len(sampled_tables) <= 1, "Sampling is currently not supported for several tables in a join."
        # If sampling is used start with sampled table
        if len(sampled_tables) == 1:
            return sampled_tables[0], table_set

        start_table = None
        max_incoming = -1
        for table in table_dict.keys():
            if self.schema_graph.table_dictionary[table].table_size < min_start_table_size:
                continue

            if table_dict[table] > max_incoming:
                start_table = table
                max_incoming = table_dict[table]

        return start_table, table_set

    def column_number(self, relationship_list=None, single_table=None):
        """
        Returns the number of columns of the join.
        :param relationship_list:
        :param single_table:
        :return:
        """
        assert relationship_list is None or single_table is None, "Specify either of the two options"

        def is_multiplier(attribute):
            for relationship in self.schema_graph.relationships:
                if relationship.multiplier_attribute_name_nn == attribute:
                    return True
                if relationship.multiplier_attribute_name == attribute:
                    return True
            return False

        def result_columns(table):
            attributes = self.table_meta_data[table]['relevant_attributes']
            # for the multipliers we either have nn or normal multiplier in the result
            multipliers = [attribute for attribute in attributes if is_multiplier(attribute)]
            return len(attributes) - len(multipliers) / 2

        no_cols = 0
        if single_table is not None:
            no_cols = result_columns(single_table)
        elif relationship_list is not None:
            for table in self.corresponding_tables(relationship_list):
                no_cols += result_columns(table)
        else:
            raise ValueError("Specify either of the two options")

        return no_cols

    def corresponding_tables(self, relationship_list):
        """
        Returns all the tables belonging to the relationships.
        :param relationship_list:
        :return:
        """

        merged_tables = set()
        for relationship in relationship_list:
            relationship_obj = self.schema_graph.relationship_dictionary[relationship]
            merged_tables.add(relationship_obj.start)
            merged_tables.add(relationship_obj.end)
        return merged_tables

    def _next_relationship(self, relationship_list, joined_tables):
        """
        Returns (if possible) outgoing relationship, otherwise incoming. This is favorable because it keeps join size
        small (greedily).
        """

        for relationship in relationship_list:
            relationship_obj = self.schema_graph.relationship_dictionary[relationship]
            if relationship_obj.start in joined_tables:
                return relationship_obj, True

        for relationship in relationship_list:
            relationship_obj = self.schema_graph.relationship_dictionary[relationship]
            if relationship_obj.end in joined_tables:
                return relationship_obj, False

        raise ValueError("No more Relationships to be joined. Maybe relationships do not form a tree?")

    def _get_table_data(self, path, table):
        """
        Obtains a table from HDF files. If already read, use cached result.
        """

        # if cached, simply return
        if self.cached_tables.get(path) is not None:
            return self.cached_tables[path]

        table_data = pd.read_hdf(path, key='df')

        # drop irrelevant attributes
        del_irr_attr = []
        table_obj = self.schema_graph.table_dictionary[table]
        for irrelevant_attr in table_obj.irrelevant_attributes:
            full_irrelevant_attr_name = table + '.' + irrelevant_attr
            if full_irrelevant_attr_name in table_data.columns:
                del_irr_attr.append(full_irrelevant_attr_name)
        if len(del_irr_attr) > 0:
            table_data = table_data.drop(columns=del_irr_attr)

        if not self.no_cache:
            self.cached_tables[path] = table_data

        return table_data

    def _get_null_value(self, table, attribute):
        null_value_index = self.table_meta_data[table]['relevant_attributes_full'] \
            .index(attribute)
        return self.table_meta_data[table]['null_values_column'][null_value_index]

    def _sampling_rate(self, table_name):
        full_table_size = self.schema_graph.table_dictionary[table_name].table_size
        if self.schema_graph.table_dictionary[table_name].sample_rate * full_table_size > self.max_table_data:
            return self.max_table_data / full_table_size
        return self.schema_graph.table_dictionary[table_name].sample_rate

    def _size_estimate(self, single_table=None, relationship_list=None, min_start_table_size=1):
        """
        Estimates the size of the full join if no sampling for large tables (like lineorder) is used for the HDF files.
        Also estimates the size of the sample considering the fact that sampling might have been used.
        :param single_table:
        :param relationship_list:
        :param min_start_table_size:
        :return:
        """

        assert single_table is None or relationship_list is None, "Either specify a single table or a set of relations"
        assert single_table is not None or relationship_list is not None, "Provide either table or set of relations"

        if single_table is not None:
            return min(self.table_meta_data[single_table]['length'] * self._sampling_rate(single_table),
                       self.max_table_data), \
                   self.table_meta_data[single_table]['length']

        todo_relationships = copy.copy(relationship_list)

        start_table, table_set = self._find_start_table(todo_relationships, min_start_table_size)
        # this is just an estimate.
        sample_size_estimate = self.table_meta_data[start_table]['length'] * self._sampling_rate(start_table)
        full_join_size = self.table_meta_data[start_table]['length']
        joined_tables = {start_table}

        while len(todo_relationships) > 0:
            relationship_obj, outgoing = self._next_relationship(todo_relationships, joined_tables)
            # outgoing edge, e.g. orders joined, join customers
            if outgoing:
                next_joined_table = relationship_obj.end
                assert next_joined_table not in joined_tables, "Query graph is not a tree."

                edge_information = self.table_meta_data[next_joined_table].get(relationship_obj.identifier)
                if edge_information is not None:
                    sample_size_estimate += edge_information['length'] * self._sampling_rate(next_joined_table)
                    full_join_size += edge_information['length']
            # incoming edge
            else:
                next_joined_table = relationship_obj.start
                assert next_joined_table not in joined_tables, "Query graph is not a tree."

                table_meta_data = self.table_meta_data[relationship_obj.end]
                sample_size_estimate *= table_meta_data['incoming_relationship_means'][
                                            relationship_obj.identifier] * self._sampling_rate(next_joined_table)
                full_join_size *= table_meta_data['incoming_relationship_means'][relationship_obj.identifier]

                # e.g. orders without customers
                # for next neighbor this is an outgoing edge
                incoming_edge_information = self.table_meta_data[next_joined_table][relationship_obj.identifier]
                if incoming_edge_information['length'] > 0:
                    # add orders without customers to full join size
                    sample_size_estimate += incoming_edge_information['length'] * self._sampling_rate(next_joined_table)
                    full_join_size += incoming_edge_information['length']

            joined_tables.add(next_joined_table)
            todo_relationships.remove(relationship_obj.identifier)

        return sample_size_estimate, full_join_size

    def generate_n_samples(self, sample_size, post_sampling_factor=30, single_table=None, relationship_list=None,
                           min_start_table_size=1, drop_redundant_columns=True):
        """
        Generates approximately sample_size samples of join.
        :param sample_size:
        :param post_sampling_factor:
        :param single_table:
        :param relationship_list:
        :param min_start_table_size:
        :return:
        """
        sample_size_estimate, full_join_size = self._size_estimate(single_table=single_table,
                                                                   relationship_list=relationship_list,
                                                                   min_start_table_size=min_start_table_size)
        # Sampling of join necessary
        if sample_size_estimate > sample_size:
            sample_rate = min(sample_size / sample_size_estimate * post_sampling_factor, 1.0)
            df_full_samples, meta_types, null_values = self.generate_join_sample(
                single_table=single_table, relationship_list=relationship_list,
                min_start_table_size=min_start_table_size, sample_rate=sample_rate,
                drop_redundant_columns=drop_redundant_columns)

            if len(df_full_samples) > sample_size:
                df_full_samples = df_full_samples.sample(sample_size)
            return df_full_samples, meta_types, null_values, full_join_size

        # No sampling required
        df_full_samples, meta_types, null_values = self.generate_join_sample(single_table=single_table,
                                                                             relationship_list=relationship_list,
                                                                             min_start_table_size=min_start_table_size,
                                                                             sample_rate=1.0,
                                                                             drop_redundant_columns=drop_redundant_columns)
        if len(df_full_samples) > sample_size:
            return df_full_samples.sample(sample_size), meta_types, null_values, full_join_size
        return df_full_samples, meta_types, null_values, full_join_size

    def generate_n_samples_with_incremental_part(self, sample_size, post_sampling_factor=30, single_table=None, relationship_list=None,
                           min_start_table_size=1, drop_redundant_columns=True, incremental_learning_rate=0, incremental_condition=None):
        """
        Generates approximately sample_size samples of join.
        :param sample_size:
        :param post_sampling_factor:
        :param single_table:
        :param relationship_list:
        :param min_start_table_size:
        :return:
        """
        sample_size_estimate, full_join_size = self._size_estimate(single_table=single_table,
                                                                   relationship_list=relationship_list,
                                                                   min_start_table_size=min_start_table_size)
        logging.debug(f"generate_n_samples_with_incremental_part(sample_size={sample_size}, single_table={single_table}, relationship_list={relationship_list}, sample_size_estimate={sample_size_estimate}, incremental_learning_rate={incremental_learning_rate}, incremental_condition={incremental_condition})")
        # Sampling of join necessary
        sample_rate = 1.0
        if sample_size_estimate > sample_size:
            sample_rate = min(sample_size / sample_size_estimate * post_sampling_factor, 1.0)
        logging.debug(f"to many samples, reduce number of samples with sample_rate={sample_rate}")
        df_full_samples, meta_types, null_values = self.generate_join_sample(single_table=single_table,
                                                                             relationship_list=relationship_list,
                                                                             min_start_table_size=min_start_table_size,
                                                                             sample_rate=sample_rate,
                                                                             drop_redundant_columns=drop_redundant_columns)

        if len(df_full_samples) > sample_size:
            df_full_samples = df_full_samples.sample(sample_size)

        # split sample in initial learning and incremental learning part (if incremental_learning_rate > 0)
        #
        if incremental_learning_rate > 0:
            full_size = len(df_full_samples)
            split_position = int(full_size * (100.0 - incremental_learning_rate)/100.0)
            logging.debug(f"split position for dataset: {split_position} (full length: {full_size}, incremenatal_rate: {incremental_learning_rate})")
            df_learn_samples = df_full_samples.iloc[0:split_position, :]
            df_inc_samples =  df_full_samples.iloc[split_position:, :]
        elif incremental_condition != None:
            import re
            column, value = re.split(" *[<] *", incremental_condition)
            if value.isdigit():
                value = int(value)
            if (value is not None):
                df_learn_samples = df_full_samples[df_full_samples['title.production_year'] < value]
                df_inc_samples = df_full_samples[df_full_samples[column] >= value]
                logging.info(f"splitting dataset into {len(df_learn_samples)}:{len(df_inc_samples)} parts, according to condition ({incremental_condition}), incremental_rate: {100.0*len(df_inc_samples)/len(df_full_samples)}% @@@")
            else:
                print("Currently only '<' operator is supported for incremental_condition (i.e. title.production_year<2015)")
                sys.exit(1)
        else:
            df_inc_samples = pd.DataFrame([])
            df_learn_samples = df_full_samples
        logging.info(f"split full sample dataset into parts: initial learning size: {len(df_learn_samples)}, incremental: {len(df_inc_samples)}")
        return df_learn_samples, df_inc_samples, meta_types, null_values, full_join_size

    def generate_join_sample(self, single_table=None, relationship_list=None, min_start_table_size=1, sample_rate=1,
                             drop_redundant_columns=True, max_intermediate_size=math.inf,
                             split_condition=None):
        """
        Samples from FULL OUTER JOIN to provide training data for SPN.
        """

        assert single_table is None or relationship_list is None, "Either specify a single table or a set of relations"
        assert single_table is not None or relationship_list is not None, "Provide either table or set of relations"

        logging.debug(f"generate_join_sample(single_table={single_table}, relationship_list={relationship_list}, split_condition={split_condition})")
        if single_table is not None:

            df_samples = self._get_table_data(self.table_meta_data[single_table]['hdf_path'], single_table)
            if sample_rate < 1:
                df_samples = df_samples.sample(prob_round(len(df_samples) * sample_rate))

            # remove unnecessary multipliers and replace nans
            del_mul_attributes = []
            mul_columns = []
            for relationship_obj in self.schema_graph.relationships:

                # if multiplier to outside: remove multiplier_nn, nan imputation for multiplier
                if relationship_obj.end == single_table:
                    del_mul_attributes.append(
                        relationship_obj.end + '.' + relationship_obj.multiplier_attribute_name_nn)
                    mul_columns.append(relationship_obj.end + '.' + relationship_obj.multiplier_attribute_name)
            if drop_redundant_columns:
                df_samples = df_samples.drop(columns=del_mul_attributes)

            # remove unnecessary id field
            table_obj = self.schema_graph.table_dictionary[single_table]
            for pk_attribute in table_obj.primary_key:
                id_attribute = single_table + '.' + pk_attribute
                if id_attribute in df_samples.columns:
                    if drop_redundant_columns:
                        df_samples = df_samples.drop(columns=[id_attribute])

            # remove unnecessary fk id field
            del_fk_cols = []
            for outgoing_relationship in table_obj.outgoing_relationships:
                if outgoing_relationship.start_attr not in table_obj.keep_fk_attributes:
                    del_fk_cols.append(single_table + '.' + outgoing_relationship.start_attr)
            if drop_redundant_columns:
                df_samples = df_samples.drop(columns=del_fk_cols)

            # Final null value imputation of other columns
            # build null value data structure
            # build data structure reflecting the meta types
            meta_types = []
            null_values = []
            for column in df_samples.columns:

                matched = False
                # does it belong to any table
                if column in self.table_meta_data[single_table]['relevant_attributes_full']:
                    if self.table_meta_data[single_table]['categorical_columns_dict'].get(column) is not None:
                        meta_types.append(MetaType.DISCRETE)
                    else:
                        meta_types.append(MetaType.REAL)

                    if column in mul_columns:
                        null_values.append(None)
                    else:
                        null_value = self._get_null_value(single_table, column)
                        null_values.append(null_value)

                    matched = True

                assert matched, f"Unknown attribute {column}"

            assert len(meta_types) == len(null_values), "Amount of null values does not match"
            assert len(meta_types) == len(df_samples.columns)

            return df_samples, meta_types, null_values

        else:
            # relationship_list is not None
            todo_relationships = copy.copy(relationship_list)

            start_table, table_set = self._find_start_table(todo_relationships, min_start_table_size)
            start_table_sample_rate = self._sampling_rate(start_table)

            # sample from first first table
            logging.debug(f"reading first table '{start_table}'")
            df_samples = self._get_table_data(self.table_meta_data[start_table]['hdf_path'], start_table)
            if sample_rate < 1:
                df_samples = df_samples.sample(prob_round(len(df_samples) * sample_rate))

            joined_tables = {start_table}

            while len(todo_relationships) > 0:
                if len(df_samples) > max_intermediate_size:
                    df_samples = df_samples.sample(max_intermediate_size)

                relationship_obj, outgoing = self._next_relationship(todo_relationships, joined_tables)

                logger.debug(f"Joining {relationship_obj.identifier}. Current join size is {len(df_samples)}.")

                # outgoing edge, e.g. lineorders joined, join date
                if outgoing:

                    next_joined_table = relationship_obj.end
                    assert next_joined_table not in joined_tables, "Query graph is not a tree."

                    next_table_data = self._get_table_data(self.table_meta_data[next_joined_table]['hdf_path'],
                                                           next_joined_table)

                    # set indexes to make pandas join fast
                    left_attribute = relationship_obj.end + '.' + relationship_obj.end_attr
                    right_attribute = relationship_obj.start + '.' + relationship_obj.start_attr

                    df_samples = df_samples.set_index(right_attribute, drop=False)
                    next_table_data = next_table_data.set_index(left_attribute, drop=False)
                    # df_samples = df_samples.join(next_table_data, how='left') #20s
                    # df_samples = df_samples.merge(next_table_data, how='left', right_on=left_attribute,
                    #   left_index=True) #34s
                    df_samples = df_samples.merge(next_table_data, how='left', right_index=True,
                                                  left_on=right_attribute)

                    # e.g. customers without orders
                    # this is an outgoing edge
                    edge_information = self.table_meta_data[next_joined_table].get(relationship_obj.identifier)
                    if edge_information is not None:

                        wo_join_partners = self._get_table_data(edge_information['path'], next_joined_table)
                        if sample_rate * start_table_sample_rate < 1:
                            wo_join_partners = wo_join_partners.sample(
                                prob_round(len(wo_join_partners) * sample_rate * start_table_sample_rate))
                        df_samples = pd.concat([df_samples, wo_join_partners])
                        del wo_join_partners

                # incoming edge, e.g. date joined, join lineorders
                else:
                    next_joined_table = relationship_obj.start
                    assert next_joined_table not in joined_tables, "Query graph is not a tree."

                    next_table_data = self._get_table_data(self.table_meta_data[next_joined_table]['hdf_path'],
                                                           next_joined_table)

                    # set indexes to make pandas join fast
                    left_attribute = relationship_obj.end + '.' + relationship_obj.end_attr
                    right_attribute = relationship_obj.start + '.' + relationship_obj.start_attr

                    df_samples = df_samples.set_index(left_attribute, drop=False)
                    next_table_data = next_table_data.set_index(right_attribute, drop=False)
                    df_samples = df_samples.merge(next_table_data, how='left', right_index=True,
                                                  left_on=left_attribute)  # 10s, 15s
                    # df_samples.merge(next_table_data, how='left', right_on=right_attribute,
                    #   left_index=True) # 16s, 20s
                    # df_samples.merge(next_table_data, how='left', right_index=True,
                    #   left_index=True) # 20s, 26s
                    # df_samples.merge(next_table_data, how='left', right_on=right_attribute,
                    #   left_on=left_attribute) # 16s, 18s
                    # df_samples = df_samples.join(next_table_data, how='left') # 23s, 23s

                    # update full_join_size with table meta data
                    table_meta_data = self.table_meta_data[relationship_obj.end]

                    # e.g. orders without customers
                    # for next neighbor this is an outgoing edge
                    incoming_edge_information = self.table_meta_data[next_joined_table][relationship_obj.identifier]
                    if incoming_edge_information['length'] > 0:

                        null_value = self._get_null_value(next_joined_table,
                                                          incoming_edge_information['fk_attribute_name'])
                        wo_join_partners = \
                            next_table_data[
                                next_table_data[incoming_edge_information['fk_attribute_name']] == null_value]
                        if sample_rate * start_table_sample_rate < 1:
                            wo_join_partners = wo_join_partners.sample(
                                prob_round(len(wo_join_partners) * sample_rate * start_table_sample_rate))
                        df_samples = pd.concat([df_samples, wo_join_partners])

                joined_tables.add(next_joined_table)
                todo_relationships.remove(relationship_obj.identifier)

            if len(df_samples) > max_intermediate_size:
                df_samples = df_samples.sample(max_intermediate_size)

            # remove unnecessary multipliers and replace nans
            mul_columns = []
            del_mul_attributes = []
            for relationship_obj in self.schema_graph.relationships:

                # if multiplier in relationship: remove multiplier, nan imputation for multiplier_nn
                if relationship_obj.start in joined_tables and relationship_obj.end in joined_tables:
                    del_mul_attributes.append(relationship_obj.end + '.' + relationship_obj.multiplier_attribute_name)
                    multiplier_nn_name = relationship_obj.end + '.' + relationship_obj.multiplier_attribute_name_nn
                    df_samples[multiplier_nn_name] = df_samples[multiplier_nn_name].fillna(1)
                    mul_columns.append(multiplier_nn_name)

                # if multiplier to outside: remove multiplier_nn, nan imputation for multiplier
                if relationship_obj.end in joined_tables and not relationship_obj.start in joined_tables:
                    del_mul_attributes.append(
                        relationship_obj.end + '.' + relationship_obj.multiplier_attribute_name_nn)
                    multiplier_name = relationship_obj.end + '.' + relationship_obj.multiplier_attribute_name
                    df_samples[multiplier_name] = df_samples[multiplier_name].fillna(0)
                    mul_columns.append(multiplier_name)

            if drop_redundant_columns:
                df_samples = df_samples.drop(columns=del_mul_attributes)

            # remove unnecessary id fields (if nan is present, turn into nn field)
            del_id_columns = []
            for table in joined_tables:

                table_obj = self.schema_graph.table_dictionary[table]

                # use first id attribute to create nn attribute if required
                id_attribute = table + '.' + table_obj.primary_key[0]
                nn_attribute = table + '.' + table_obj.table_nn_attribute

                # attribute nn field is required
                if df_samples[id_attribute].isna().any():
                    df_samples = df_samples.rename(columns={id_attribute: nn_attribute})
                    df_samples.loc[df_samples[nn_attribute].notna(), nn_attribute] = 1
                    df_samples.loc[df_samples[nn_attribute].isna(), nn_attribute] = 0

                # column can be removed
                else:
                    del_id_columns.append(id_attribute)

                # remove all other id attributes
                if len(table_obj.primary_key) > 1:
                    for pk_attribute in table_obj.primary_key[1:]:
                        del_id_columns.append(table + '.' + pk_attribute)

            df_samples = df_samples.drop(columns=del_id_columns)

            # remove unnecessary fk id field
            del_fk_cols = []
            for table in joined_tables:
                table_obj = self.schema_graph.table_dictionary[table]
                for outgoing_relationship in table_obj.outgoing_relationships:
                    if outgoing_relationship.start_attr not in table_obj.keep_fk_attributes:
                        del_fk_cols.append(table + '.' + outgoing_relationship.start_attr)
            df_samples = df_samples.drop(columns=del_fk_cols)

            # Final null value imputation of other columns
            # build null value data structure
            # build data structure reflecting the meta types
            meta_types = []
            null_values = []
            for column in df_samples.columns:

                matched = False
                # does it belong to any table
                for table in joined_tables:
                    if column in self.table_meta_data[table]['relevant_attributes_full']:
                        if self.table_meta_data[table]['categorical_columns_dict'].get(column) is not None:
                            meta_types.append(MetaType.DISCRETE)
                        else:
                            meta_types.append(MetaType.REAL)

                        if column in mul_columns:
                            null_values.append(None)
                        else:
                            null_value = self._get_null_value(table, column)
                            null_values.append(null_value)
                            df_samples[column] = df_samples[column].fillna(null_value)
                        matched = True
                        break

                # should be nn attribute
                for table in joined_tables:
                    table_obj = self.schema_graph.table_dictionary[table]

                    if column == table + '.' + table_obj.table_nn_attribute:
                        meta_types.append(MetaType.DISCRETE)
                        null_values.append(0)
                        matched = True
                        break

                assert matched, "Unknown attribute"

            assert len(meta_types) == len(null_values), "Amount of null values does not match"
            assert len(meta_types) == len(df_samples.columns)

            logger.debug(f"Final join size is {len(df_samples)}.")

            return df_samples, meta_types, null_values


def prepare_sample_hdf(schema, hdf_path, max_table_data, sample_size):
    meta_data_path = hdf_path + '/meta_data.pkl'
    prep = JoinDataPreparator(meta_data_path, schema, max_table_data=max_table_data)
    new_meta_data = copy.deepcopy(prep.table_meta_data)

    def correct_meta_data(table):
        new_meta_data[table]['hdf_path'] = new_meta_data[table]['hdf_path'].replace(table, table + '_sampled')
        incoming_relationships = find_relationships(schema, table, incoming=True)
        for relationship_obj in incoming_relationships:
            new_meta_data[table][relationship_obj.identifier] = None
        outgoing_relationships = find_relationships(schema, table, incoming=False)
        for relationship_obj in outgoing_relationships:
            new_meta_data[table][relationship_obj.identifier]['length'] = 0

    # find first table and sample
    max_join_relationships, _ = create_random_join(schema, len(schema.relationships))
    start_table, _ = prep._find_start_table(max_join_relationships, 1)
    logger.debug(f"Creating sample for {start_table}")
    sampled_tables = {start_table}
    df_sample_cache = dict()
    df_full_samples, _, _, _ = prep.generate_n_samples(sample_size, single_table=start_table,
                                                       drop_redundant_columns=False)
    df_sample_cache[start_table] = df_full_samples
    df_full_samples.to_hdf(f'{hdf_path}/{start_table}_sampled.hdf', key='df', format='table')
    correct_meta_data(start_table)

    while len(sampled_tables) < len(schema.tables):
        for relationship_obj in schema.relationships:
            if (relationship_obj.start in sampled_tables and not relationship_obj.end in sampled_tables) or (
                    relationship_obj.start not in sampled_tables and relationship_obj.end in sampled_tables):
                if relationship_obj.start in sampled_tables and not relationship_obj.end in sampled_tables:
                    # outgoing edge, e.g. lineorders joined, join date
                    next_joined_table = relationship_obj.end
                    logger.debug(f"Creating sample for {next_joined_table}")
                    next_table_data = prep._get_table_data(prep.table_meta_data[next_joined_table]['hdf_path'],
                                                           next_joined_table)
                    left_attribute = relationship_obj.end + '.' + relationship_obj.end_attr
                    right_attribute = relationship_obj.start + '.' + relationship_obj.start_attr

                    df_samples = df_sample_cache[relationship_obj.start]
                    df_samples = df_samples.set_index(right_attribute, drop=False)
                    next_table_data = next_table_data.set_index(left_attribute, drop=False)
                    next_table_data = df_samples.merge(next_table_data, right_index=True, left_on=right_attribute)
                    # only keep rows with join partner
                    next_table_data = next_table_data[
                        next_table_data[relationship_obj.end + '.' + relationship_obj.multiplier_attribute_name] > 0]

                elif relationship_obj.start not in sampled_tables and relationship_obj.end in sampled_tables:
                    next_joined_table = relationship_obj.start
                    logger.debug(f"Creating sample for {next_joined_table}")
                    next_table_data = prep._get_table_data(prep.table_meta_data[next_joined_table]['hdf_path'],
                                                           next_joined_table)
                    left_attribute = relationship_obj.end + '.' + relationship_obj.end_attr
                    right_attribute = relationship_obj.start + '.' + relationship_obj.start_attr

                    df_samples = df_sample_cache[relationship_obj.end]
                    df_samples = df_samples.set_index(left_attribute, drop=False)
                    # df_samples.index.name = None
                    next_table_data = next_table_data.set_index(right_attribute, drop=False)
                    next_table_data = df_samples.merge(next_table_data, right_index=True, left_on=left_attribute)
                    # only keep rows with join partner
                    next_table_data = next_table_data[
                        next_table_data[relationship_obj.end + '.' + relationship_obj.multiplier_attribute_name] > 0]

                if len(next_table_data) > sample_size:
                    next_table_data = next_table_data.sample(sample_size)
                # only keep columns of interest
                del_cols = []
                for col in next_table_data.columns:
                    if col not in prep.table_meta_data[next_joined_table]['relevant_attributes_full']:
                        del_cols.append(col)
                next_table_data.drop(columns=del_cols, inplace=True)
                df_sample_cache[next_joined_table] = next_table_data
                next_table_data.to_hdf(f'{hdf_path}/{next_joined_table}_sampled.hdf', key='df', format='table')
                correct_meta_data(next_joined_table)
                sampled_tables.add(next_joined_table)

    # different meta data
    with open(hdf_path + '/meta_data_sampled.pkl', 'wb') as f:
        pickle.dump(new_meta_data, f, pickle.HIGHEST_PROTOCOL)
