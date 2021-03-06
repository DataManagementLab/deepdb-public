select count(*) from title t,cast_info ci,movie_info_idx mi_idx where t.id=ci.movie_id and t.id=mi_idx.movie_id and ci.role_id < 11 and t.kind_id < 4 and t.production_year < 1993;
select count(*) from title t,cast_info ci,movie_companies mc where t.id=ci.movie_id and t.id=mc.movie_id and mc.company_type_id = 2 and ci.person_id > 3991904 and t.production_year > 1911;
select count(*) from title t,cast_info ci,movie_companies mc where t.id=ci.movie_id and t.id=mc.movie_id and mc.company_id > 124708 and mc.company_type_id = 2 and ci.person_id > 228567;
select count(*) from title t,cast_info ci,movie_companies mc where t.id=ci.movie_id and t.id=mc.movie_id and t.production_year < 1987 and mc.company_type_id = 1 and t.kind_id < 3;
select count(*) from title t,movie_info mi,movie_info_idx mi_idx where t.id=mi.movie_id and t.id=mi_idx.movie_id and mi_idx.info_type_id < 100 and t.production_year < 1965 and mi.info_type_id < 15;
select count(*) from title t,cast_info ci,movie_info mi where t.id=ci.movie_id and t.id=mi.movie_id and ci.role_id < 6 and t.production_year < 1983 and mi.info_type_id > 59;
select count(*) from title t,movie_companies mc,movie_info mi where t.id=mc.movie_id and t.id=mi.movie_id and t.production_year < 1951 and t.kind_id > 5 and mc.company_type_id < 2;
select count(*) from title t,movie_keyword mk,movie_info_idx mi_idx where t.id=mi_idx.movie_id and t.id=mk.movie_id and t.kind_id > 4 and mk.keyword_id = 32636 and mi_idx.info_type_id < 112;
select count(*) from title t,movie_companies mc,movie_info mi where t.id=mc.movie_id and t.id=mi.movie_id and t.kind_id < 6 and mi.info_type_id > 94 and mc.company_id > 178917;
select count(*) from title t,cast_info ci,movie_info_idx mi_idx where t.id=ci.movie_id and t.id=mi_idx.movie_id and t.production_year = 1986 and ci.person_id < 1817192 and mi_idx.info_type_id < 109;
select count(*) from title t,movie_info mi,movie_info_idx mi_idx where t.id=mi.movie_id and t.id=mi_idx.movie_id and t.production_year > 1920 and t.kind_id > 1 and mi_idx.info_type_id < 105;
select count(*) from title t,cast_info ci,movie_info mi where t.id=ci.movie_id and t.id=mi.movie_id and mi.info_type_id > 99 and t.kind_id = 7 and ci.role_id = 2;
select count(*) from title t,movie_keyword mk,movie_info mi where t.id=mi.movie_id and t.id=mk.movie_id and t.kind_id > 4 and mi.info_type_id < 49 and t.production_year < 2003;
select count(*) from title t,cast_info ci,movie_companies mc where t.id=ci.movie_id and t.id=mc.movie_id and ci.role_id > 2 and mc.company_id > 209874 and ci.person_id < 2570169;
select count(*) from title t,cast_info ci,movie_keyword mk where t.id=ci.movie_id and t.id=mk.movie_id and t.production_year > 1951 and mk.keyword_id > 9177 and ci.role_id = 3;
select count(*) from title t,cast_info ci,movie_companies mc where t.id=ci.movie_id and t.id=mc.movie_id and t.production_year = 1996 and ci.person_id > 2990546 and ci.role_id = 6;
select count(*) from title t,cast_info ci,movie_companies mc where t.id=ci.movie_id and t.id=mc.movie_id and ci.person_id < 2852867 and t.production_year > 1905 and mc.company_id < 92459;
select count(*) from title t,cast_info ci,movie_info mi where t.id=ci.movie_id and t.id=mi.movie_id and ci.role_id = 7 and mi.info_type_id > 67 and t.kind_id < 5;
select count(*) from title t,cast_info ci,movie_keyword mk where t.id=ci.movie_id and t.id=mk.movie_id and t.kind_id = 6 and t.production_year > 1921 and mk.keyword_id < 22497;
select count(*) from title t,movie_info mi,movie_info_idx mi_idx where t.id=mi.movie_id and t.id=mi_idx.movie_id and t.production_year < 1989 and mi_idx.info_type_id = 99 and t.kind_id < 6;
select count(*) from title t,cast_info ci,movie_companies mc where t.id=ci.movie_id and t.id=mc.movie_id and mc.company_type_id = 1 and t.production_year > 1917 and ci.person_id < 2486060;
select count(*) from title t,movie_info mi,movie_info_idx mi_idx where t.id=mi.movie_id and t.id=mi_idx.movie_id and t.production_year > 1941 and t.kind_id > 4 and mi.info_type_id < 42;
select count(*) from title t,movie_keyword mk,movie_info_idx mi_idx where t.id=mi_idx.movie_id and t.id=mk.movie_id and mi_idx.info_type_id < 107 and t.kind_id < 4 and mk.keyword_id < 5393;
select count(*) from title t,cast_info ci,movie_companies mc where t.id=ci.movie_id and t.id=mc.movie_id and ci.person_id < 2841294 and t.production_year < 2014 and mc.company_type_id = 1;
select count(*) from title t,movie_companies mc,movie_info mi where t.id=mc.movie_id and t.id=mi.movie_id and mi.info_type_id = 88 and mc.company_id > 228217 and t.production_year > 1885;
select count(*) from title t,movie_keyword mk,movie_info_idx mi_idx where t.id=mi_idx.movie_id and t.id=mk.movie_id and t.production_year = 2006 and mi_idx.info_type_id < 100 and t.kind_id > 6;
select count(*) from title t,cast_info ci,movie_keyword mk where t.id=ci.movie_id and t.id=mk.movie_id and t.kind_id > 4 and mk.keyword_id = 36203 and t.production_year > 1941;
select count(*) from title t,movie_companies mc,movie_info_idx mi_idx where t.id=mc.movie_id and t.id=mi_idx.movie_id and mi_idx.info_type_id < 100 and t.production_year < 1958 and mc.company_type_id > 1;
select count(*) from title t,cast_info ci,movie_info mi where t.id=ci.movie_id and t.id=mi.movie_id and t.kind_id > 2 and mi.info_type_id > 7 and ci.person_id < 1060262;
select count(*) from title t,cast_info ci,movie_info_idx mi_idx where t.id=ci.movie_id and t.id=mi_idx.movie_id and mi_idx.info_type_id > 99 and t.production_year < 1920 and ci.role_id = 7;
select count(*) from title t,cast_info ci,movie_info_idx mi_idx where t.id=ci.movie_id and t.id=mi_idx.movie_id and t.kind_id = 4 and t.production_year > 1981 and ci.role_id > 10;
select count(*) from title t,cast_info ci,movie_companies mc where t.id=ci.movie_id and t.id=mc.movie_id and t.production_year < 1953 and ci.role_id < 2 and ci.person_id > 980614;
select count(*) from title t,cast_info ci,movie_companies mc where t.id=ci.movie_id and t.id=mc.movie_id and mc.company_type_id < 2 and ci.role_id > 1 and ci.person_id < 2970301;
select count(*) from title t,cast_info ci,movie_info_idx mi_idx where t.id=ci.movie_id and t.id=mi_idx.movie_id and t.production_year > 1921 and ci.role_id < 11 and ci.person_id > 1601476;
select count(*) from title t,cast_info ci,movie_keyword mk where t.id=ci.movie_id and t.id=mk.movie_id and t.kind_id > 3 and ci.person_id < 949958 and t.production_year > 1896;
select count(*) from title t,movie_keyword mk,movie_info_idx mi_idx where t.id=mi_idx.movie_id and t.id=mk.movie_id and t.production_year < 2016 and t.kind_id < 4 and mk.keyword_id < 66541;
select count(*) from title t,movie_companies mc,movie_info mi where t.id=mc.movie_id and t.id=mi.movie_id and t.production_year > 1996 and mi.info_type_id < 6 and mc.company_id = 109620;
select count(*) from title t,movie_info mi,movie_info_idx mi_idx where t.id=mi.movie_id and t.id=mi_idx.movie_id and mi_idx.info_type_id < 105 and mi.info_type_id > 73 and t.production_year < 2016;
select count(*) from title t,movie_info mi,movie_info_idx mi_idx where t.id=mi.movie_id and t.id=mi_idx.movie_id and t.production_year = 2014 and t.kind_id < 4 and mi_idx.info_type_id < 101;
select count(*) from title t,movie_info mi,movie_info_idx mi_idx where t.id=mi.movie_id and t.id=mi_idx.movie_id and mi.info_type_id > 99 and mi_idx.info_type_id > 109 and t.production_year > 1962;
select count(*) from title t,cast_info ci,movie_info mi where t.id=ci.movie_id and t.id=mi.movie_id and t.production_year < 1967 and t.kind_id = 1 and mi.info_type_id < 11;
select count(*) from title t,movie_keyword mk,movie_info mi where t.id=mi.movie_id and t.id=mk.movie_id and mi.info_type_id < 35 and mk.keyword_id > 105150 and t.kind_id = 1;
select count(*) from title t,movie_companies mc,movie_info_idx mi_idx where t.id=mc.movie_id and t.id=mi_idx.movie_id and mi_idx.info_type_id > 99 and t.production_year > 2007 and mc.company_type_id = 1;
select count(*) from title t,movie_keyword mk,movie_info mi where t.id=mi.movie_id and t.id=mk.movie_id and t.kind_id < 4 and mi.info_type_id < 57 and mk.keyword_id > 6555;
select count(*) from title t,cast_info ci,movie_info mi where t.id=ci.movie_id and t.id=mi.movie_id and ci.person_id < 3232591 and t.production_year = 1992 and ci.role_id > 10;
select count(*) from title t,movie_companies mc,movie_info_idx mi_idx where t.id=mc.movie_id and t.id=mi_idx.movie_id and mc.company_id < 174086 and mc.company_type_id = 1 and mi_idx.info_type_id < 111;
select count(*) from title t,movie_companies mc,movie_keyword mk where t.id=mc.movie_id and t.id=mk.movie_id and mk.keyword_id > 45538 and t.kind_id > 6 and mc.company_id < 11944;
select count(*) from title t,movie_info mi,movie_info_idx mi_idx where t.id=mi.movie_id and t.id=mi_idx.movie_id and t.production_year = 1899 and mi_idx.info_type_id < 109 and mi.info_type_id < 62;
select count(*) from title t,cast_info ci,movie_companies mc where t.id=ci.movie_id and t.id=mc.movie_id and mc.company_type_id < 2 and mc.company_id > 47907 and t.kind_id < 2;