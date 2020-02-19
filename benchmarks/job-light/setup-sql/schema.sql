-- Table: public.aka_name

-- DROP TABLE public.aka_name;

CREATE TABLE public.aka_name
(
    id integer NOT NULL DEFAULT nextval('aka_name_id_seq'::regclass),
    person_id integer NOT NULL,
    name text COLLATE pg_catalog."default" NOT NULL,
    imdb_index character varying(12) COLLATE pg_catalog."default",
    name_pcode_cf character varying(5) COLLATE pg_catalog."default",
    name_pcode_nf character varying(5) COLLATE pg_catalog."default",
    surname_pcode character varying(5) COLLATE pg_catalog."default",
    md5sum character varying(32) COLLATE pg_catalog."default",
    CONSTRAINT aka_name_pkey PRIMARY KEY (id),
    CONSTRAINT person_id_exists FOREIGN KEY (person_id)
        REFERENCES public.name (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.aka_name
    OWNER to postgres;

-- Index: aka_name_idx_md5

-- DROP INDEX public.aka_name_idx_md5;

CREATE INDEX aka_name_idx_md5
    ON public.aka_name USING btree
    (md5sum COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Index: aka_name_idx_pcode

-- DROP INDEX public.aka_name_idx_pcode;

CREATE INDEX aka_name_idx_pcode
    ON public.aka_name USING btree
    (surname_pcode COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Index: aka_name_idx_pcodecf

-- DROP INDEX public.aka_name_idx_pcodecf;

CREATE INDEX aka_name_idx_pcodecf
    ON public.aka_name USING btree
    (name_pcode_cf COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Index: aka_name_idx_pcodenf

-- DROP INDEX public.aka_name_idx_pcodenf;

CREATE INDEX aka_name_idx_pcodenf
    ON public.aka_name USING btree
    (name_pcode_nf COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Index: aka_name_idx_person

-- DROP INDEX public.aka_name_idx_person;

CREATE INDEX aka_name_idx_person
    ON public.aka_name USING btree
    (person_id)
    TABLESPACE pg_default;

-- Table: public.aka_title

-- DROP TABLE public.aka_title;

CREATE TABLE public.aka_title
(
    id integer NOT NULL DEFAULT nextval('aka_title_id_seq'::regclass),
    movie_id integer NOT NULL,
    title text COLLATE pg_catalog."default" NOT NULL,
    imdb_index character varying(12) COLLATE pg_catalog."default",
    kind_id integer NOT NULL,
    production_year integer,
    phonetic_code character varying(5) COLLATE pg_catalog."default",
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    note text COLLATE pg_catalog."default",
    md5sum character varying(32) COLLATE pg_catalog."default",
    CONSTRAINT aka_title_pkey PRIMARY KEY (id)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.aka_title
    OWNER to postgres;

-- Index: aka_title_idx_epof

-- DROP INDEX public.aka_title_idx_epof;

CREATE INDEX aka_title_idx_epof
    ON public.aka_title USING btree
    (episode_of_id)
    TABLESPACE pg_default;

-- Index: aka_title_idx_md5

-- DROP INDEX public.aka_title_idx_md5;

CREATE INDEX aka_title_idx_md5
    ON public.aka_title USING btree
    (md5sum COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Index: aka_title_idx_movieid

-- DROP INDEX public.aka_title_idx_movieid;

CREATE INDEX aka_title_idx_movieid
    ON public.aka_title USING btree
    (movie_id)
    TABLESPACE pg_default;

-- Index: aka_title_idx_pcode

-- DROP INDEX public.aka_title_idx_pcode;

CREATE INDEX aka_title_idx_pcode
    ON public.aka_title USING btree
    (phonetic_code COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Table: public.cast_info

-- DROP TABLE public.cast_info;

CREATE TABLE public.cast_info
(
    id integer NOT NULL DEFAULT nextval('cast_info_id_seq'::regclass),
    person_id integer NOT NULL,
    movie_id integer NOT NULL,
    person_role_id integer,
    note text COLLATE pg_catalog."default",
    nr_order integer,
    role_id integer NOT NULL,
    CONSTRAINT cast_info_pkey PRIMARY KEY (id),
    CONSTRAINT movie_id_exists FOREIGN KEY (movie_id)
        REFERENCES public.title (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT person_id_exists FOREIGN KEY (person_id)
        REFERENCES public.name (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT person_role_id_exists FOREIGN KEY (person_role_id)
        REFERENCES public.char_name (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT role_id_exists FOREIGN KEY (role_id)
        REFERENCES public.role_type (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.cast_info
    OWNER to postgres;

-- Index: cast_info_idx_cid

-- DROP INDEX public.cast_info_idx_cid;

CREATE INDEX cast_info_idx_cid
    ON public.cast_info USING btree
    (person_role_id)
    TABLESPACE pg_default;

-- Index: cast_info_idx_mid

-- DROP INDEX public.cast_info_idx_mid;

CREATE INDEX cast_info_idx_mid
    ON public.cast_info USING btree
    (movie_id)
    TABLESPACE pg_default;

-- Index: cast_info_idx_pid

-- DROP INDEX public.cast_info_idx_pid;

CREATE INDEX cast_info_idx_pid
    ON public.cast_info USING btree
    (person_id)
    TABLESPACE pg_default;

-- Table: public.char_name

-- DROP TABLE public.char_name;

CREATE TABLE public.char_name
(
    id integer NOT NULL DEFAULT nextval('char_name_id_seq'::regclass),
    name text COLLATE pg_catalog."default" NOT NULL,
    imdb_index character varying(12) COLLATE pg_catalog."default",
    imdb_id integer,
    name_pcode_nf character varying(5) COLLATE pg_catalog."default",
    surname_pcode character varying(5) COLLATE pg_catalog."default",
    md5sum character varying(32) COLLATE pg_catalog."default",
    CONSTRAINT char_name_pkey PRIMARY KEY (id)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.char_name
    OWNER to postgres;

-- Index: char_name_idx_md5

-- DROP INDEX public.char_name_idx_md5;

CREATE INDEX char_name_idx_md5
    ON public.char_name USING btree
    (md5sum COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Index: char_name_idx_name

-- DROP INDEX public.char_name_idx_name;

CREATE INDEX char_name_idx_name
    ON public.char_name USING btree
    (name COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Index: char_name_idx_pcode

-- DROP INDEX public.char_name_idx_pcode;

CREATE INDEX char_name_idx_pcode
    ON public.char_name USING btree
    (surname_pcode COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Index: char_name_idx_pcodenf

-- DROP INDEX public.char_name_idx_pcodenf;

CREATE INDEX char_name_idx_pcodenf
    ON public.char_name USING btree
    (name_pcode_nf COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Table: public.comp_cast_type

-- DROP TABLE public.comp_cast_type;

CREATE TABLE public.comp_cast_type
(
    id integer NOT NULL DEFAULT nextval('comp_cast_type_id_seq'::regclass),
    kind character varying(32) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT comp_cast_type_pkey PRIMARY KEY (id),
    CONSTRAINT comp_cast_type_kind_key UNIQUE (kind)

)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.comp_cast_type
    OWNER to postgres;

-- Table: public.company_name

-- DROP TABLE public.company_name;

CREATE TABLE public.company_name
(
    id integer NOT NULL DEFAULT nextval('company_name_id_seq'::regclass),
    name text COLLATE pg_catalog."default" NOT NULL,
    country_code character varying(255) COLLATE pg_catalog."default",
    imdb_id integer,
    name_pcode_nf character varying(5) COLLATE pg_catalog."default",
    name_pcode_sf character varying(5) COLLATE pg_catalog."default",
    md5sum character varying(32) COLLATE pg_catalog."default",
    CONSTRAINT company_name_pkey PRIMARY KEY (id)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.company_name
    OWNER to postgres;

-- Index: company_name_idx_md5

-- DROP INDEX public.company_name_idx_md5;

CREATE INDEX company_name_idx_md5
    ON public.company_name USING btree
    (md5sum COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Index: company_name_idx_name

-- DROP INDEX public.company_name_idx_name;

CREATE INDEX company_name_idx_name
    ON public.company_name USING btree
    (name COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Index: company_name_idx_pcodenf

-- DROP INDEX public.company_name_idx_pcodenf;

CREATE INDEX company_name_idx_pcodenf
    ON public.company_name USING btree
    (name_pcode_nf COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Index: company_name_idx_pcodesf

-- DROP INDEX public.company_name_idx_pcodesf;

CREATE INDEX company_name_idx_pcodesf
    ON public.company_name USING btree
    (name_pcode_sf COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Table: public.company_type

-- DROP TABLE public.company_type;

CREATE TABLE public.company_type
(
    id integer NOT NULL DEFAULT nextval('company_type_id_seq'::regclass),
    kind character varying(32) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT company_type_pkey PRIMARY KEY (id),
    CONSTRAINT company_type_kind_key UNIQUE (kind)

)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.company_type
    OWNER to postgres;

-- Table: public.complete_cast

-- DROP TABLE public.complete_cast;

CREATE TABLE public.complete_cast
(
    id integer NOT NULL DEFAULT nextval('complete_cast_id_seq'::regclass),
    movie_id integer,
    subject_id integer NOT NULL,
    status_id integer NOT NULL,
    CONSTRAINT complete_cast_pkey PRIMARY KEY (id),
    CONSTRAINT movie_id_exists FOREIGN KEY (movie_id)
        REFERENCES public.title (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT status_id_exists FOREIGN KEY (status_id)
        REFERENCES public.comp_cast_type (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT subject_id_exists FOREIGN KEY (subject_id)
        REFERENCES public.comp_cast_type (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.complete_cast
    OWNER to postgres;

-- Index: complete_cast_idx_mid

-- DROP INDEX public.complete_cast_idx_mid;

CREATE INDEX complete_cast_idx_mid
    ON public.complete_cast USING btree
    (movie_id)
    TABLESPACE pg_default;

-- Table: public.info_type

-- DROP TABLE public.info_type;

CREATE TABLE public.info_type
(
    id integer NOT NULL DEFAULT nextval('info_type_id_seq'::regclass),
    info character varying(32) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT info_type_pkey PRIMARY KEY (id),
    CONSTRAINT info_type_info_key UNIQUE (info)

)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.info_type
    OWNER to postgres;

-- Table: public.keyword

-- DROP TABLE public.keyword;

CREATE TABLE public.keyword
(
    id integer NOT NULL DEFAULT nextval('keyword_id_seq'::regclass),
    keyword text COLLATE pg_catalog."default" NOT NULL,
    phonetic_code character varying(5) COLLATE pg_catalog."default",
    CONSTRAINT keyword_pkey PRIMARY KEY (id)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.keyword
    OWNER to postgres;

-- Index: keyword_idx_keyword

-- DROP INDEX public.keyword_idx_keyword;

CREATE INDEX keyword_idx_keyword
    ON public.keyword USING btree
    (keyword COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Index: keyword_idx_pcode

-- DROP INDEX public.keyword_idx_pcode;

CREATE INDEX keyword_idx_pcode
    ON public.keyword USING btree
    (phonetic_code COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Table: public.kind_type

-- DROP TABLE public.kind_type;

CREATE TABLE public.kind_type
(
    id integer NOT NULL DEFAULT nextval('kind_type_id_seq'::regclass),
    kind character varying(15) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT kind_type_pkey PRIMARY KEY (id),
    CONSTRAINT kind_type_kind_key UNIQUE (kind)

)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.kind_type
    OWNER to postgres;

-- Table: public.link_type

-- DROP TABLE public.link_type;

CREATE TABLE public.link_type
(
    id integer NOT NULL DEFAULT nextval('link_type_id_seq'::regclass),
    link character varying(32) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT link_type_pkey PRIMARY KEY (id),
    CONSTRAINT link_type_link_key UNIQUE (link)

)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.link_type
    OWNER to postgres;

-- Table: public.movie_companies

-- DROP TABLE public.movie_companies;

CREATE TABLE public.movie_companies
(
    id integer NOT NULL DEFAULT nextval('movie_companies_id_seq'::regclass),
    movie_id integer NOT NULL,
    company_id integer NOT NULL,
    company_type_id integer NOT NULL,
    note text COLLATE pg_catalog."default",
    CONSTRAINT movie_companies_pkey PRIMARY KEY (id),
    CONSTRAINT company_id_exists FOREIGN KEY (company_id)
        REFERENCES public.company_name (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT company_type_id_exists FOREIGN KEY (company_type_id)
        REFERENCES public.company_type (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT movie_id_exists FOREIGN KEY (movie_id)
        REFERENCES public.title (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.movie_companies
    OWNER to postgres;

-- Index: movie_companies_idx_cid

-- DROP INDEX public.movie_companies_idx_cid;

CREATE INDEX movie_companies_idx_cid
    ON public.movie_companies USING btree
    (company_id)
    TABLESPACE pg_default;

-- Index: movie_companies_idx_mid

-- DROP INDEX public.movie_companies_idx_mid;

CREATE INDEX movie_companies_idx_mid
    ON public.movie_companies USING btree
    (movie_id)
    TABLESPACE pg_default;

-- Table: public.movie_info

-- DROP TABLE public.movie_info;

CREATE TABLE public.movie_info
(
    id integer NOT NULL DEFAULT nextval('movie_info_id_seq'::regclass),
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text COLLATE pg_catalog."default" NOT NULL,
    note text COLLATE pg_catalog."default",
    CONSTRAINT movie_info_pkey PRIMARY KEY (id),
    CONSTRAINT info_type_id_exists FOREIGN KEY (info_type_id)
        REFERENCES public.info_type (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT movie_id_exists FOREIGN KEY (movie_id)
        REFERENCES public.title (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.movie_info
    OWNER to postgres;

-- Index: movie_info_idx_mid

-- DROP INDEX public.movie_info_idx_mid;

CREATE INDEX movie_info_idx_mid
    ON public.movie_info USING btree
    (movie_id)
    TABLESPACE pg_default;

-- Table: public.movie_info_idx

-- DROP TABLE public.movie_info_idx;

CREATE TABLE public.movie_info_idx
(
    id integer NOT NULL DEFAULT nextval('movie_info_idx_id_seq'::regclass),
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text COLLATE pg_catalog."default" NOT NULL,
    note text COLLATE pg_catalog."default",
    CONSTRAINT movie_info_idx_pkey PRIMARY KEY (id),
    CONSTRAINT info_type_id_exists FOREIGN KEY (info_type_id)
        REFERENCES public.info_type (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT movie_id_exists FOREIGN KEY (movie_id)
        REFERENCES public.title (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.movie_info_idx
    OWNER to postgres;

-- Index: movie_info_idx_idx_info

-- DROP INDEX public.movie_info_idx_idx_info;

CREATE INDEX movie_info_idx_idx_info
    ON public.movie_info_idx USING btree
    (info COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Index: movie_info_idx_idx_infotypeid

-- DROP INDEX public.movie_info_idx_idx_infotypeid;

CREATE INDEX movie_info_idx_idx_infotypeid
    ON public.movie_info_idx USING btree
    (info_type_id)
    TABLESPACE pg_default;

-- Index: movie_info_idx_idx_mid

-- DROP INDEX public.movie_info_idx_idx_mid;

CREATE INDEX movie_info_idx_idx_mid
    ON public.movie_info_idx USING btree
    (movie_id)
    TABLESPACE pg_default;

-- Table: public.movie_keyword

-- DROP TABLE public.movie_keyword;

CREATE TABLE public.movie_keyword
(
    id integer NOT NULL DEFAULT nextval('movie_keyword_id_seq'::regclass),
    movie_id integer NOT NULL,
    keyword_id integer NOT NULL,
    CONSTRAINT movie_keyword_pkey PRIMARY KEY (id),
    CONSTRAINT keyword_id_exists FOREIGN KEY (keyword_id)
        REFERENCES public.keyword (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT movie_id_exists FOREIGN KEY (movie_id)
        REFERENCES public.title (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.movie_keyword
    OWNER to postgres;

-- Index: movie_keyword_idx_keywordid

-- DROP INDEX public.movie_keyword_idx_keywordid;

CREATE INDEX movie_keyword_idx_keywordid
    ON public.movie_keyword USING btree
    (keyword_id)
    TABLESPACE pg_default;

-- Index: movie_keyword_idx_mid

-- DROP INDEX public.movie_keyword_idx_mid;

CREATE INDEX movie_keyword_idx_mid
    ON public.movie_keyword USING btree
    (movie_id)
    TABLESPACE pg_default;

-- Table: public.movie_link

-- DROP TABLE public.movie_link;

CREATE TABLE public.movie_link
(
    id integer NOT NULL DEFAULT nextval('movie_link_id_seq'::regclass),
    movie_id integer NOT NULL,
    linked_movie_id integer NOT NULL,
    link_type_id integer NOT NULL,
    CONSTRAINT movie_link_pkey PRIMARY KEY (id),
    CONSTRAINT link_type_id_exists FOREIGN KEY (link_type_id)
        REFERENCES public.link_type (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT linked_movie_id_exists FOREIGN KEY (linked_movie_id)
        REFERENCES public.title (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT movie_id_exists FOREIGN KEY (movie_id)
        REFERENCES public.title (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.movie_link
    OWNER to postgres;

-- Index: movie_link_idx_mid

-- DROP INDEX public.movie_link_idx_mid;

CREATE INDEX movie_link_idx_mid
    ON public.movie_link USING btree
    (movie_id)
    TABLESPACE pg_default;

-- Table: public.name

-- DROP TABLE public.name;

CREATE TABLE public.name
(
    id integer NOT NULL DEFAULT nextval('name_id_seq'::regclass),
    name text COLLATE pg_catalog."default" NOT NULL,
    imdb_index character varying(12) COLLATE pg_catalog."default",
    imdb_id integer,
    gender character varying(1) COLLATE pg_catalog."default",
    name_pcode_cf character varying(5) COLLATE pg_catalog."default",
    name_pcode_nf character varying(5) COLLATE pg_catalog."default",
    surname_pcode character varying(5) COLLATE pg_catalog."default",
    md5sum character varying(32) COLLATE pg_catalog."default",
    CONSTRAINT name_pkey PRIMARY KEY (id)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.name
    OWNER to postgres;

-- Index: name_idx_imdb_id

-- DROP INDEX public.name_idx_imdb_id;

CREATE INDEX name_idx_imdb_id
    ON public.name USING btree
    (imdb_id)
    TABLESPACE pg_default;

-- Index: name_idx_md5

-- DROP INDEX public.name_idx_md5;

CREATE INDEX name_idx_md5
    ON public.name USING btree
    (md5sum COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Index: name_idx_name

-- DROP INDEX public.name_idx_name;

CREATE INDEX name_idx_name
    ON public.name USING btree
    (name COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Index: name_idx_pcode

-- DROP INDEX public.name_idx_pcode;

CREATE INDEX name_idx_pcode
    ON public.name USING btree
    (surname_pcode COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Index: name_idx_pcodecf

-- DROP INDEX public.name_idx_pcodecf;

CREATE INDEX name_idx_pcodecf
    ON public.name USING btree
    (name_pcode_cf COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Index: name_idx_pcodenf

-- DROP INDEX public.name_idx_pcodenf;

CREATE INDEX name_idx_pcodenf
    ON public.name USING btree
    (name_pcode_nf COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Table: public.person_info

-- DROP TABLE public.person_info;

CREATE TABLE public.person_info
(
    id integer NOT NULL DEFAULT nextval('person_info_id_seq'::regclass),
    person_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text COLLATE pg_catalog."default" NOT NULL,
    note text COLLATE pg_catalog."default",
    CONSTRAINT person_info_pkey PRIMARY KEY (id),
    CONSTRAINT info_type_id_exists FOREIGN KEY (info_type_id)
        REFERENCES public.info_type (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT person_id_exists FOREIGN KEY (person_id)
        REFERENCES public.name (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.person_info
    OWNER to postgres;

-- Index: person_info_idx_pid

-- DROP INDEX public.person_info_idx_pid;

CREATE INDEX person_info_idx_pid
    ON public.person_info USING btree
    (person_id)
    TABLESPACE pg_default;

-- Table: public.role_type

-- DROP TABLE public.role_type;

CREATE TABLE public.role_type
(
    id integer NOT NULL DEFAULT nextval('role_type_id_seq'::regclass),
    role character varying(32) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT role_type_pkey PRIMARY KEY (id),
    CONSTRAINT role_type_role_key UNIQUE (role)

)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.role_type
    OWNER to postgres;

-- Table: public.title

-- DROP TABLE public.title;

CREATE TABLE public.title
(
    id integer NOT NULL DEFAULT nextval('title_id_seq'::regclass),
    title text COLLATE pg_catalog."default" NOT NULL,
    imdb_index character varying(12) COLLATE pg_catalog."default",
    kind_id integer NOT NULL,
    production_year integer,
    imdb_id integer,
    phonetic_code character varying(5) COLLATE pg_catalog."default",
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    series_years character varying(49) COLLATE pg_catalog."default",
    md5sum character varying(32) COLLATE pg_catalog."default",
    CONSTRAINT title_pkey PRIMARY KEY (id),
    CONSTRAINT episode_of_id_exists FOREIGN KEY (episode_of_id)
        REFERENCES public.title (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT kind_id_exists FOREIGN KEY (kind_id)
        REFERENCES public.kind_type (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.title
    OWNER to postgres;

-- Index: title_idx_episode_nr

-- DROP INDEX public.title_idx_episode_nr;

CREATE INDEX title_idx_episode_nr
    ON public.title USING btree
    (episode_nr)
    TABLESPACE pg_default;

-- Index: title_idx_epof

-- DROP INDEX public.title_idx_epof;

CREATE INDEX title_idx_epof
    ON public.title USING btree
    (episode_of_id)
    TABLESPACE pg_default;

-- Index: title_idx_imdb_id

-- DROP INDEX public.title_idx_imdb_id;

CREATE INDEX title_idx_imdb_id
    ON public.title USING btree
    (imdb_id)
    TABLESPACE pg_default;

-- Index: title_idx_md5

-- DROP INDEX public.title_idx_md5;

CREATE INDEX title_idx_md5
    ON public.title USING btree
    (md5sum COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Index: title_idx_pcode

-- DROP INDEX public.title_idx_pcode;

CREATE INDEX title_idx_pcode
    ON public.title USING btree
    (phonetic_code COLLATE pg_catalog."default")
    TABLESPACE pg_default;

-- Index: title_idx_season_nr

-- DROP INDEX public.title_idx_season_nr;

CREATE INDEX title_idx_season_nr
    ON public.title USING btree
    (season_nr)
    TABLESPACE pg_default;

-- Index: title_idx_title

-- DROP INDEX public.title_idx_title;

CREATE INDEX title_idx_title
    ON public.title USING btree
    (title COLLATE pg_catalog."default")
    TABLESPACE pg_default;

