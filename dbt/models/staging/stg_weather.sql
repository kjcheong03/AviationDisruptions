WITH source AS (
    SELECT * FROM {{ ref('weather_sample') }}
),

cleaned AS (
    SELECT
        -- PK
        CONCAT(
            airport_code, '_', 
            CAST(CAST(time AS DATE) AS STRING), '_', 
            CAST(EXTRACT(HOUR FROM CAST(time AS TIMESTAMP)) AS STRING)
        ) AS weather_id,

        -- time parsing
        time,
        CAST(time AS TIMESTAMP) AS time_parsed,
        CAST(time AS DATE) AS date_id,
        EXTRACT(HOUR FROM CAST(time AS TIMESTAMP)) AS hour_id,

        -- ID the airport
        airport_code AS airport_id,

        -- weather metrics
        CAST(temperature_2m AS FLOAT64) AS temperature_c,
        CAST(precipitation AS FLOAT64) AS precipitation_mm,
        CAST(wind_speed_10m AS FLOAT64) AS wind_speed_knots,
        
        -- Cloud cover
        CAST(cloudcover AS INT64) AS cloud_cover_total_pct,
        CAST(cloudcover_low AS INT64) AS cloud_cover_low_pct

    FROM source
)

SELECT * FROM cleaned