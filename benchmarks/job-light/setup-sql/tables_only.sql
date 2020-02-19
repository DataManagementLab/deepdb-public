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
);

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
);

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
);

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
);

CREATE TABLE public.comp_cast_type
(
    id integer NOT NULL DEFAULT nextval('comp_cast_type_id_seq'::regclass),
    kind character varying(32) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT comp_cast_type_pkey PRIMARY KEY (id),
    CONSTRAINT comp_cast_type_kind_key UNIQUE (kind)

);

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
);

CREATE TABLE public.company_type
(
    id integer NOT NULL DEFAULT nextval('company_type_id_seq'::regclass),
    kind character varying(32) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT company_type_pkey PRIMARY KEY (id),
    CONSTRAINT company_type_kind_key UNIQUE (kind)

);

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
);

CREATE TABLE public.info_type
(
    id integer NOT NULL DEFAULT nextval('info_type_id_seq'::regclass),
    info character varying(32) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT info_type_pkey PRIMARY KEY (id),
    CONSTRAINT info_type_info_key UNIQUE (info)

);

CREATE TABLE public.keyword
(
    id integer NOT NULL DEFAULT nextval('keyword_id_seq'::regclass),
    keyword text COLLATE pg_catalog."default" NOT NULL,
    phonetic_code character varying(5) COLLATE pg_catalog."default",
    CONSTRAINT keyword_pkey PRIMARY KEY (id)
);

CREATE TABLE public.kind_type
(
    id integer NOT NULL DEFAULT nextval('kind_type_id_seq'::regclass),
    kind character varying(15) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT kind_type_pkey PRIMARY KEY (id),
    CONSTRAINT kind_type_kind_key UNIQUE (kind)
);

CREATE TABLE public.link_type
(
    id integer NOT NULL DEFAULT nextval('link_type_id_seq'::regclass),
    link character varying(32) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT link_type_pkey PRIMARY KEY (id),
    CONSTRAINT link_type_link_key UNIQUE (link)

);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

CREATE TABLE public.role_type
(
    id integer NOT NULL DEFAULT nextval('role_type_id_seq'::regclass),
    role character varying(32) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT role_type_pkey PRIMARY KEY (id),
    CONSTRAINT role_type_role_key UNIQUE (role)
);

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
);





