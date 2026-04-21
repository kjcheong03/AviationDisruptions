WITH openmeteo_src AS (
    SELECT
        CAST(time AS TIMESTAMP) AS ts_utc,
        CAST(time AS DATE) AS date_id,
        EXTRACT(HOUR FROM CAST(time AS TIMESTAMP)) AS hour_id,
        airport_code AS airport_id,
        CAST(temperature_2m AS FLOAT64) AS temperature_c,
        CAST(precipitation AS FLOAT64) AS precipitation_mm,
        CAST(wind_speed_10m AS FLOAT64) AS wind_speed_knots,
        CAST(cloud_cover AS INT64) AS cloud_cover_total_pct,
        CAST(cloud_cover_low AS INT64) AS cloud_cover_low_pct,
        CAST(NULL AS TIMESTAMP) AS ingested_ts,
        'openmeteo' AS weather_source,
        2 AS source_priority
    FROM {{ source('raw_weather', 'openmeteo_raw') }}
),

metar_src AS (
    SELECT
        TIMESTAMP_TRUNC(SAFE_CAST(obs_time AS TIMESTAMP), HOUR) AS ts_utc,
        CAST(TIMESTAMP_TRUNC(SAFE_CAST(obs_time AS TIMESTAMP), HOUR) AS DATE) AS date_id,
        EXTRACT(HOUR FROM TIMESTAMP_TRUNC(SAFE_CAST(obs_time AS TIMESTAMP), HOUR)) AS hour_id,
        iata_code AS airport_id,
        CAST(temp_c AS FLOAT64) AS temperature_c,
        CAST(NULL AS FLOAT64) AS precipitation_mm,
        CAST(wind_speed_knots AS FLOAT64) AS wind_speed_knots,
        CAST(NULL AS INT64) AS cloud_cover_total_pct,
        CAST(NULL AS INT64) AS cloud_cover_low_pct,
        SAFE_CAST(ingested_at AS TIMESTAMP) AS ingested_ts,
        'metar' AS weather_source,
        1 AS source_priority
    FROM {{ source('raw_weather', 'metar_raw') }}
    WHERE SAFE_CAST(obs_time AS TIMESTAMP) IS NOT NULL
),

taf_src AS (
    SELECT
        TIMESTAMP_TRUNC(TIMESTAMP_SECONDS(CAST(valid_from AS INT64)), HOUR) AS ts_utc,
        CAST(TIMESTAMP_TRUNC(TIMESTAMP_SECONDS(CAST(valid_from AS INT64)), HOUR) AS DATE) AS date_id,
        EXTRACT(HOUR FROM TIMESTAMP_TRUNC(TIMESTAMP_SECONDS(CAST(valid_from AS INT64)), HOUR)) AS hour_id,
        iata_code AS airport_id,
        CAST(NULL AS FLOAT64) AS temperature_c,
        CAST(NULL AS FLOAT64) AS precipitation_mm,
        CAST(wind_speed_knots AS FLOAT64) AS wind_speed_knots,
        CAST(NULL AS INT64) AS cloud_cover_total_pct,
        CAST(NULL AS INT64) AS cloud_cover_low_pct,
        SAFE_CAST(ingested_at AS TIMESTAMP) AS ingested_ts,
        'taf' AS weather_source,
        3 AS source_priority
    FROM {{ source('raw_weather', 'taf_raw') }}
    WHERE valid_from IS NOT NULL
),

unified AS (
    SELECT * FROM metar_src
    UNION ALL
    SELECT * FROM openmeteo_src
    UNION ALL
    SELECT * FROM taf_src
),

ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY airport_id, date_id, hour_id
            ORDER BY source_priority ASC, ts_utc DESC, ingested_ts DESC
        ) AS rn
    FROM unified
    WHERE airport_id IS NOT NULL
),

cleaned AS (
    SELECT
        CONCAT(airport_id, '_', CAST(date_id AS STRING), '_', CAST(hour_id AS STRING)) AS weather_id,
        ts_utc AS time,
        ts_utc AS time_parsed,
        date_id,
        hour_id,
        airport_id,
        temperature_c,
        precipitation_mm,
        wind_speed_knots,
        cloud_cover_total_pct,
        cloud_cover_low_pct
    FROM ranked
    WHERE rn = 1
)

SELECT * FROM cleaned
