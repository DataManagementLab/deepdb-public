from ensemble_compilation.graph_representation import SchemaGraph, Table


def gen_flights_1B_schema(csv_path):
    schema = gen_flights_10M_schema(csv_path)

    schema.table_dictionary['flights'].sample_rate = 0.01
    schema.table_dictionary['flights'].table_size = 1000000000

    return schema


def gen_mini_flights_schema(csv_path):
    schema = gen_flights_10M_schema(csv_path)

    schema.table_dictionary['flights'].sample_rate = 0.0001
    schema.table_dictionary['flights'].table_size = 10000000

    return schema


def gen_flights_10M_schema(csv_path):
    """
    Flights schema with 1M tuples
    """

    schema = SchemaGraph()
    # YEAR_DATE,UNIQUE_CARRIER,ORIGIN,ORIGIN_STATE_ABR,DEST,DEST_STATE_ABR,DEP_DELAY,TAXI_OUT,TAXI_IN,ARR_DELAY,AIR_TIME,DISTANCE

    # tables
    # lineorder
    schema.add_table(Table('flights',
                           attributes=['year_date', 'unique_carrier', 'origin', 'origin_state_abr', 'dest',
                                       'dest_state_abr', 'dep_delay', 'taxi_out', 'taxi_in', 'arr_delay', 'air_time',
                                       'distance'],
                           csv_file_location=csv_path.format('dataset_sampled'),
                           table_size=10000000, primary_key=['f_flightno'], sample_rate=0.1,
                           fd_list=[('origin', 'origin_state_abr'), ('dest', 'dest_state_abr')]
                           ))

    return schema


def gen_flights_5M_schema(csv_path):
    schema = gen_flights_10M_schema(csv_path)

    schema.table_dictionary['flights'].sample_rate = 1
    schema.table_dictionary['flights'].table_size = 5000000
    schema.table_dictionary['flights'].csv_file_location = csv_path.format('orig_sample')

    return schema
