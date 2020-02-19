select sum(lo_extendedprice*lo_discount)
from lineorder, dwdate
where lo_orderdate = d_datekey
and d_year = 1993
and lo_discount>=1
and lo_discount<=3
and lo_quantity < 25;

select sum(lo_extendedprice*lo_discount)
 from lineorder, dwdate
 where lo_orderdate = d_datekey
 and d_yearmonthnum = 199401
 and lo_discount>=4
 and lo_discount<=6
 and  lo_quantity>=26
 and lo_quantity<=35;

select sum(lo_extendedprice*lo_discount)
from lineorder, dwdate
where lo_orderdate = d_datekey
and d_weeknuminyear = 6
and d_year = 1994
and lo_discount>=5
and lo_discount<=7
and lo_quantity>=26
and lo_quantity<=35;

select d_year, p_brand1, sum(lo_revenue)
from lineorder, dwdate, part, supplier
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_category = 'MFGR#12'
and s_region = 'AMERICA'
group by d_year, p_brand1
order by d_year, p_brand1;

select d_year, p_brand1, sum(lo_revenue)
from lineorder, dwdate, part, supplier
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_brand1 in ('MFGR#2221','MFGR#2222','MFGR#2223','MFGR#2224','MFGR#2225','MFGR#2226','MFGR#2227','MFGR#2228')
and s_region = 'ASIA'
group by d_year, p_brand1
order by d_year, p_brand1;

select d_year, p_brand1, sum(lo_revenue)
from lineorder, dwdate, part, supplier
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_brand1 = 'MFGR#2221'
and s_region = 'EUROPE'
group by d_year, p_brand1
order by d_year, p_brand1;

select c_nation, s_nation, d_year, sum(lo_revenue)
from customer, lineorder, supplier, dwdate
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_orderdate = d_datekey
and c_region = 'ASIA' and s_region = 'ASIA'
and d_year >= 1992 and d_year <= 1997
group by c_nation, s_nation, d_year
order by d_year asc;

select c_city, s_city, d_year, sum(lo_revenue)
from customer, lineorder, supplier, dwdate
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_orderdate = d_datekey
and c_nation = 'UNITED STATES'
and s_nation = 'UNITED STATES'
and d_year >= 1992 and d_year <= 1997
group by c_city, s_city, d_year
order by d_year asc;

select c_city, s_city, d_year, sum(lo_revenue)
from customer, lineorder, supplier, dwdate
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_orderdate = d_datekey
and c_city IN ('UNITED KI1', 'UNITED KI5')
and s_city IN ('UNITED KI1', 'UNITED KI5')
and d_year >= 1992 and d_year <= 1997
group by c_city, s_city, d_year
order by d_year asc;

select c_city, s_city, d_year, sum(lo_revenue)
from customer, lineorder, supplier, dwdate
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_orderdate = d_datekey
and c_city IN ('UNITED KI1', 'UNITED KI5')
and s_city IN ('UNITED KI1', 'UNITED KI5')
and d_yearmonth = 'Dec1997'
group by c_city, s_city, d_year
order by d_year asc;

select d_year, c_nation, sum(lo_revenue) - sum(lo_supplycost)
from dwdate, customer, supplier, part, lineorder
where lo_custkey = c_custkey
 and lo_suppkey = s_suppkey
 and lo_partkey = p_partkey
 and lo_orderdate = d_datekey
 and c_region = 'AMERICA'
 and s_region = 'AMERICA'
 and p_mfgr IN ('MFGR#1', 'MFGR#2')
group by d_year, c_nation
order by d_year, c_nation;

select d_year, s_nation, p_category, sum(lo_revenue) - sum(lo_supplycost)
from dwdate, customer, supplier, part, lineorder
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_partkey = p_partkey
and lo_orderdate = d_datekey
and c_region = 'AMERICA'
and s_region = 'AMERICA'
and d_year >= 1997 and d_year <= 1998
and p_mfgr IN ('MFGR#1', 'MFGR#2')
group by d_year, s_nation, p_category order by d_year, s_nation, p_category;

select d_year, s_city, p_brand1, sum(lo_revenue) - sum(lo_supplycost)
from dwdate, customer, supplier, part, lineorder
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_partkey = p_partkey
and lo_orderdate = d_datekey
and c_region = 'AMERICA'
and s_nation = 'UNITED STATES'
and d_year >= 1997 and d_year <= 1998
and p_category = 'MFGR#14'
group by d_year, s_city, p_brand1 order by d_year, s_city, p_brand1;