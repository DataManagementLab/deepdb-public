from ensemble_compilation.graph_representation import SchemaGraph, Table


def gen_mini_ssb_schema(csv_path):
    schema = gen_500gb_ssb_schema(csv_path)
    table_sizes = {'lineorder': 500,
                   'dwdate': 10,
                   'part': 10,
                   'supplier': 10,
                   'customer': 10}
    schema.table_dictionary['lineorder'].sample_rate = 0.02
    for table in schema.tables:
        table.table_size = table_sizes[table.table_name]
        schema.table_dictionary[table.table_name].table_size = table_sizes[table.table_name]

    return schema


def gen_10gb_ssb_schema(csv_path):
    schema = gen_500gb_ssb_schema(csv_path)
    table_sizes = {'lineorder': 59986052,
                   'dwdate': 2556,
                   'part': 800000,
                   'supplier': 20000,
                   'customer': 300000}
    schema.table_dictionary['lineorder'].sample_rate = 0.20
    for table in schema.tables:
        table.table_size = table_sizes[table.table_name]
        schema.table_dictionary[table.table_name].table_size = table_sizes[table.table_name]

    return schema


def gen_1000gb_ssb_schema(csv_path):
    schema = gen_500gb_ssb_schema(csv_path)
    table_sizes = {'lineorder': 5999989709,
                   'dwdate': 2556,
                   'part': 2000000,
                   'supplier': 2000000,
                   'customer': 30000000}
    schema.table_dictionary['lineorder'].sample_rate = 0.02
    for table in schema.tables:
        table.table_size = table_sizes[table.table_name]
        schema.table_dictionary[table.table_name].table_size = table_sizes[table.table_name]

    return schema

def gen_test_incr_learn_ssb_schema(csv_path, lo_size=100):
#    Loc.enter('lo_size:', lo_size)
    schema = gen_500gb_ssb_schema(csv_path)
    table_sizes = {'lineorder': lo_size,   # 11998347 complete
                   'dwdate': 2556,
                   'part': 800000,
                   'supplier': 20000,
                   'customer': 300000}
    schema.table_dictionary['lineorder'].sample_rate = 1.0
    for table in schema.tables:
        table.table_size = table_sizes[table.table_name]
        schema.table_dictionary[table.table_name].table_size = table_sizes[table.table_name]
#    Loc.leave()
    return schema


def gen_test_incr_learn_ssb_sampled_schema(csv_path, lo_size=100, sample_factor=.01):
#    Loc.enter('lo_size:', lo_size)
    schema = gen_500gb_ssb_schema(csv_path)
    table_sizes = {'lineorder': lo_size,   # 11998347 complete
                   'dwdate': 2556,
                   'part': 800000,
                   'supplier': 20000,
                   'customer': 300000}
    schema.table_dictionary['lineorder'].sample_rate = sample_factor
    sf = str(sample_factor)[2:]
    schema.table_dictionary['lineorder'].csv_file_location=csv_path.format(f"lineorder_sampled_{sf}")
    for table in schema.tables:
        table.table_size = table_sizes[table.table_name]
        schema.table_dictionary[table.table_name].table_size = table_sizes[table.table_name]
#    Loc.leave()
    return schema


def gen_500gb_ssb_schema(csv_path):
    """
    SSB schema for SF=500.
    """

    schema = SchemaGraph()

    # tables
    # lineorder
    schema.add_table(Table('lineorder',
                           attributes=['lo_orderkey', 'lo_linenumber', 'lo_custkey', 'lo_partkey', 'lo_suppkey',
                                       'lo_orderdate', 'lo_orderpriority', 'lo_shippriority', 'lo_quantity',
                                       'lo_extendedprice', 'lo_ordertotalprice', 'lo_discount', 'lo_revenue',
                                       'lo_supplycost', 'lo_tax', 'lo_commitdate', 'lo_shipmode'],
                           csv_file_location=csv_path.format('lineorder_sampled'),
                           irrelevant_attributes=['lo_commitdate'],
                           table_size=3000028242, primary_key=['lo_orderkey', 'lo_linenumber'], sample_rate=0.003333))

    # dwdate
    # dwdate.d_dayofweek -> dwdate.d_daynuminweek
    # dwdate.d_dayofweek -> dwdate.d_lastdayinweekfl
    # dwdate.d_month -> dwdate.d_monthnuminyear
    # dwdate.d_monthnuminyear -> dwdate.d_sellingseason
    # dwdate.d_daynuminyear -> dwdate.d_weeknuminyear
    schema.add_table(
        Table('dwdate',
              attributes=['d_datekey', 'd_date', 'd_dayofweek', 'd_month', 'd_year', 'd_yearmonthnum', 'd_yearmonth',
                          'd_daynuminweek', 'd_daynuminmonth', 'd_daynuminyear', 'd_monthnuminyear', 'd_weeknuminyear',
                          'd_sellingseason', 'd_lastdayinweekfl', 'd_lastdayinmonthfl', 'd_holidayfl', 'd_weekdayfl'],
              csv_file_location=csv_path.format('date'),
              table_size=2556, primary_key=["d_datekey"],
              fd_list=[('d_dayofweek', 'd_daynuminweek'), ('d_dayofweek', 'd_lastdayinweekfl'),
                       ('d_month', 'd_monthnuminyear'), ('d_monthnuminyear', 'd_sellingseason'),
                       ('d_daynuminyear', 'd_weeknuminyear')]))

    # customer
    # customer.c_city -> customer.c_nation
    # customer.c_nation -> customer.c_region
    schema.add_table(
        Table('customer',
              attributes=['c_custkey', 'c_name', 'c_address', 'c_city', 'c_nation', 'c_region', 'c_phone',
                          'c_mktsegment'],
              csv_file_location=csv_path.format('customer'),
              table_size=15000000, primary_key=["c_custkey"],
              fd_list=[('c_city', 'c_nation'), ('c_nation', 'c_region')]))

    # part
    # part.p_brand1 -> part.p_category
    # part.p_category -> part.p_mfgr
    schema.add_table(
        Table('part',
              attributes=['p_partkey', 'p_name', 'p_mfgr', 'p_category', 'p_brand1', 'p_color', 'p_type', 'p_size',
                          'p_container'],
              csv_file_location=csv_path.format('part'),
              table_size=1800000, primary_key=["p_partkey"],
              fd_list=[('p_category', 'p_mfgr'), ('p_brand1', 'p_category')]))

    # supplier
    # supplier.s_city -> supplier.s_nation
    # supplier.s_nation -> supplier.s_region
    schema.add_table(
        Table('supplier', attributes=['s_suppkey', 's_name', 's_address', 's_city', 's_nation', 's_region', 's_phone'],
              csv_file_location=csv_path.format('supplier'),
              table_size=1000000, primary_key=["s_suppkey"],
              fd_list=[('s_city', 's_nation'), ('s_nation', 's_region')]))

    # relationships
    schema.add_relationship('lineorder', 'lo_custkey', 'customer', 'c_custkey')
    schema.add_relationship('lineorder', 'lo_partkey', 'part', 'p_partkey')
    schema.add_relationship('lineorder', 'lo_suppkey', 'supplier', 's_suppkey')
    schema.add_relationship('lineorder', 'lo_orderdate', 'dwdate', 'd_datekey')
    # schema.add_relationship('lineorder', 'lo_commitdate', 'dwdate', 'd_datekey')

    return schema
