import psycopg2
import pandas as pd

from ensemble_compilation.utils import gen_full_join_query, print_conditions


class DBConnection:

    def __init__(self, db_user="postgres", db_password="postgres", db_host="localhost", db_port="5432", db="shopdb"):
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.db = db

    def vacuum(self):
        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        old_isolation_level = connection.isolation_level
        connection.set_isolation_level(0)
        query = "VACUUM"
        cursor = connection.cursor()
        cursor.execute(query)
        connection.commit()
        connection.set_isolation_level(old_isolation_level)

    def get_dataframe(self, sql):
        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        return pd.read_sql(sql, connection)

    def submit_query(self, sql):
        """Submits query and ignores result."""

        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        cursor = connection.cursor()
        cursor.execute(sql)
        connection.commit()

    def get_result(self, sql):
        """Fetches exactly one row of result set."""

        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        cursor = connection.cursor()

        cursor.execute(sql)
        record = cursor.fetchone()
        result = record[0]

        if connection:
            cursor.close()
            connection.close()

        return result

    def get_result_set(self, sql, return_columns=False):
        """Fetches all rows of result set."""

        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        cursor = connection.cursor()

        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        if connection:
            cursor.close()
            connection.close()

        if return_columns:
            return rows, columns

        return rows


class TrueCardinalityEstimator:
    """Queries the database to return true cardinalities."""

    def __init__(self, schema_graph, db_connection):
        self.schema_graph = schema_graph
        self.db_connection = db_connection

    def true_cardinality(self, query):
        full_join_query = gen_full_join_query(self.schema_graph, query.relationship_set, query.table_set, "JOIN")

        where_cond = print_conditions(query.conditions, seperator='AND')
        if where_cond != "":
            where_cond = "WHERE " + where_cond
        sql_query = full_join_query.format("COUNT(*)", where_cond)
        cardinality = self.db_connection.get_result(sql_query)
        return sql_query, cardinality
