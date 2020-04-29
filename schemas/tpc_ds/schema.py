from ensemble_compilation.graph_representation import SchemaGraph, Table


def gen_10gb_tpc_ds_schema(csv_path):
    """
    TPCDS 10g schema
    """

    schema = SchemaGraph()
    schema.add_table(Table('store_sales',
                           attributes=['ss_sold_date_sk', 'ss_sold_time_sk', 'ss_item_sk', 'ss_customer_sk',
                                       'ss_cdemo_sk', 'ss_hdemo_sk', 'ss_addr_sk', 'ss_store_sk', 'ss_promo_sk',
                                       'ss_ticket_number', 'ss_quantity', 'ss_wholesale_cost', 'ss_list_price',
                                       'ss_sales_price', 'ss_ext_discount_amt', 'ss_ext_sales_price',
                                       'ss_ext_wholesale_cost', 'ss_ext_list_price', 'ss_ext_tax',
                                       'ss_coupon_amt', 'ss_net_paid', 'ss_net_paid_inc_tax', 'ss_net_profit'],
                           irrelevant_attributes=['ss_sold_time_sk', 'ss_item_sk', 'ss_customer_sk', 'ss_cdemo_sk',
                                                  'ss_hdemo_sk', 'ss_addr_sk', 'ss_promo_sk', 'ss_ticket_number',
                                                  'ss_quantity', 'ss_wholesale_cost', 'ss_list_price',
                                                  'ss_ext_discount_amt', 'ss_ext_sales_price', 'ss_ext_wholesale_cost',
                                                  'ss_ext_list_price', 'ss_ext_tax',
                                                  'ss_coupon_amt', 'ss_net_paid', 'ss_net_paid_inc_tax',
                                                  'ss_net_profit'],
                           no_compression=['ss_sold_date_sk', 'ss_store_sk', 'ss_sales_price'],
                           csv_file_location=csv_path.format('store_sales_sampled'),
                           table_size=28800991, primary_key=['ss_item_sk', 'ss_ticket_number'],
                           sample_rate=10000000 / 28800991
                           ))

    return schema


def gen_1t_tpc_ds_schema(csv_path):
    """
    TPCDS 1t schema
    """

    schema = SchemaGraph()
    schema.add_table(Table('store_sales',
                           attributes=['ss_sold_date_sk', 'ss_sold_time_sk', 'ss_item_sk', 'ss_customer_sk',
                                       'ss_cdemo_sk', 'ss_hdemo_sk', 'ss_addr_sk', 'ss_store_sk', 'ss_promo_sk',
                                       'ss_ticket_number', 'ss_quantity', 'ss_wholesale_cost', 'ss_list_price',
                                       'ss_sales_price', 'ss_ext_discount_amt', 'ss_ext_sales_price',
                                       'ss_ext_wholesale_cost', 'ss_ext_list_price', 'ss_ext_tax',
                                       'ss_coupon_amt', 'ss_net_paid', 'ss_net_paid_inc_tax', 'ss_net_profit'],
                           irrelevant_attributes=['ss_sold_time_sk', 'ss_item_sk', 'ss_customer_sk', 'ss_cdemo_sk',
                                                  'ss_hdemo_sk', 'ss_addr_sk', 'ss_promo_sk', 'ss_ticket_number',
                                                  'ss_quantity', 'ss_wholesale_cost', 'ss_list_price',
                                                  'ss_ext_discount_amt', 'ss_ext_sales_price', 'ss_ext_wholesale_cost',
                                                  'ss_ext_list_price', 'ss_ext_tax',
                                                  'ss_coupon_amt', 'ss_net_paid', 'ss_net_paid_inc_tax',
                                                  'ss_net_profit'],
                           no_compression=['ss_sold_date_sk', 'ss_store_sk', 'ss_sales_price'],
                           csv_file_location=csv_path.format('store_sales_sampled'),
                           table_size=2879987999, primary_key=['ss_item_sk', 'ss_ticket_number'],
                           sample_rate=10000000 / 2879987999
                           ))

    return schema

# suboptimal configuration
# def gen_10gb_tpc_ds_schema(csv_path):
#     """
#     TPCDS 10g schema
#     """
#
#     schema = SchemaGraph()
#     schema.add_table(Table('store_sales',
#                            attributes=['ss_sold_date_sk', 'ss_sold_time_sk', 'ss_item_sk', 'ss_customer_sk',
#                                        'ss_cdemo_sk', 'ss_hdemo_sk', 'ss_addr_sk', 'ss_store_sk', 'ss_promo_sk',
#                                        'ss_ticket_number', 'ss_quantity', 'ss_wholesale_cost', 'ss_list_price',
#                                        'ss_sales_price', 'ss_ext_discount_amt', 'ss_ext_sales_price',
#                                        'ss_ext_wholesale_cost', 'ss_ext_list_price', 'ss_ext_tax',
#                                        'ss_coupon_amt', 'ss_net_paid', 'ss_net_paid_inc_tax', 'ss_net_profit'],
#                            csv_file_location=csv_path.format('store_sales_sampled'),
#                            table_size=28800991, primary_key=['ss_item_sk', 'ss_ticket_number'], sample_rate=0.33
#                            ))
#
#     return schema
