CREATE TABLE flights (
    YEAR_DATE integer,
    UNIQUE_CARRIER varchar(100),
    ORIGIN varchar(100),
    ORIGIN_STATE_ABR varchar(2),
    DEST varchar(100),
    DEST_STATE_ABR varchar(2),
    DEP_DELAY decimal,
    TAXI_OUT decimal,
    TAXI_IN decimal,
    ARR_DELAY decimal,
    AIR_TIME decimal,
    DISTANCE decimal
);