SELECT SUM(ss_sales_price) FROM store_sales;
SELECT ss_store_sk, SUM(ss_sales_price) FROM store_sales GROUP BY ss_store_sk;
SELECT ss_store_sk, SUM(ss_sales_price) FROM store_sales WHERE ss_sold_date_sk>=2452632 GROUP BY ss_store_sk;
SELECT ss_store_sk, SUM(ss_sales_price) FROM store_sales WHERE ss_sold_date_sk>=2451642 GROUP BY ss_store_sk;
SELECT ss_store_sk, SUM(ss_sales_price) FROM store_sales WHERE ss_sold_date_sk>=2450632 GROUP BY ss_store_sk;