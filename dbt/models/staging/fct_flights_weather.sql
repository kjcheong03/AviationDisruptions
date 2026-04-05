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
        f.tail_number,
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
),

-- aggregate the current hour's conditions
hourly_airport_conditions AS (
    SELECT
        origin_airport_id AS airport_id,
        date_id,
        hour_id,
        AVG(dep_delay_minutes) AS current_hour_delay_avg,
        COUNT(flight_id) AS current_hour_flight_count
    FROM joined
    GROUP BY 1, 2, 3
),

-- shift the aggregated data down by one hour
shifted_hourly_conditions AS (
    SELECT
        airport_id,
        date_id,
        hour_id,
        LAG(current_hour_delay_avg, 1) OVER (
            PARTITION BY airport_id 
            ORDER BY date_id, hour_id
        ) AS prev_hour_delay_avg,
        
        LAG(current_hour_flight_count, 1) OVER (
            PARTITION BY airport_id 
            ORDER BY date_id, hour_id
        ) AS prev_hour_flight_count
    FROM hourly_airport_conditions
),

-- Time-Series Feature Engineering - to include current state of the world signals into each row
lagged_features AS (
    SELECT
        j.*,
        
        -- aircraft chain (if depart delays, high risk for subsequent departs to delay)
        LAG(j.arr_delay_minutes, 1) OVER (
            PARTITION BY j.tail_number
            ORDER BY j.date_id, j.hour_id, j.flight_id
        ) AS lag_1_tail_arr_delay_mins,

        -- congestion at original airport
        AVG(j.dep_delay_minutes) OVER (
            PARTITION BY j.origin_airport_id
            ORDER BY j.date_id, j.hour_id, j.flight_id
            ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
        ) AS rolling_6_flight_origin_delay_avg,

        COUNT(j.dep_delay_minutes) OVER (
            PARTITION BY j.origin_airport_id
            ORDER BY j.date_id, j.hour_id, j.flight_id
            ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
        ) AS rolling_6_flight_origin_count,

        -- for the airport, how congested is it based on prev hour delay avg and flight count
        d.prev_hour_delay_avg AS dest_prev_hour_delay_avg,
        d.prev_hour_flight_count AS dest_prev_hour_flight_count,

        -- weather momentum
        j.wind_speed_knots - LAG(j.wind_speed_knots, 1) OVER (
            PARTITION BY j.origin_airport_id
            ORDER BY j.date_id, j.hour_id, j.flight_id
        ) AS wind_speed_delta,

        -- weather delay risk
        MAX(CASE WHEN j.weather_delay_mins > 0 THEN 1 ELSE 0 END) OVER (
            PARTITION BY j.origin_airport_id
            ORDER BY j.date_id, j.hour_id, j.flight_id
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS rolling_3_flight_weather_delay_flag

    FROM joined AS j
    -- join against shifted timeline
    LEFT JOIN shifted_hourly_conditions AS d
        ON j.dest_airport_id = d.airport_id
        AND j.date_id = d.date_id
        AND j.hour_id = d.hour_id
)

SELECT * FROM lagged_features