WITH flights AS (
    SELECT * FROM {{ ref('stg_flights') }}
),

weather AS (
    SELECT * FROM {{ ref('stg_weather') }}
),

joined AS (
    SELECT 
        -- identifiers and FKs
        f.flight_id,
        f.flight_date AS date_id,
        f.scheduled_dep_hour AS hour_id,
        f.airline_code AS airline_id,
        f.flight_number,
        f.origin_airport AS origin_airport_id,
        f.dest_airport AS dest_airport_id,
        
        -- for feature eng
        f.flight_month,
        f.day_of_week,
        f.distance_miles,
        f.taxi_out_mins,
        f.taxi_in_mins,

        -- status and delay metrics (targets)
        f.is_cancelled,
        f.cancellation_code,
        f.dep_delay_minutes,
        f.arr_delay_minutes,
        f.weather_delay_mins,
        f.nas_delay_mins,
        f.carrier_delay_mins,
        f.security_delay_mins,
        f.late_aircraft_delay_mins,
        
        --  weather metrics (for prediction)
        w.temperature_c,
        w.precipitation_mm,
        w.wind_speed_knots,
        w.cloud_cover_total_pct,
        w.cloud_cover_low_pct

    FROM flights AS f
    -- ERD JOIN: connect flights to the weather at their Origin Airport, on the specific Date, during the specific Hour
    LEFT JOIN weather AS w
        ON f.origin_airport = w.airport_id
        AND f.flight_date = w.date_id
        AND f.scheduled_dep_hour = w.hour_id
)

SELECT * FROM joined