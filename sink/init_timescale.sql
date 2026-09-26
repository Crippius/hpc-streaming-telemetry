CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS metric_agg_1h (
    node_id           text             NOT NULL,
    metric            text             NOT NULL,
    stat              text             NOT NULL,
    window_start      timestamptz      NOT NULL,
    window_end        timestamptz      NOT NULL,
    avg_value         double precision,
    min_value         double precision,
    max_value         double precision,
    n_samples         integer          NOT NULL,
    anomaly_count     integer          NOT NULL,
    labeled_count     integer          NOT NULL,
    last_publish_time timestamptz,
    written_at        timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (node_id, metric, stat, window_start)
);

SELECT create_hypertable('metric_agg_1h', by_range('window_start', INTERVAL '7 days'),
                         if_not_exists => TRUE);
