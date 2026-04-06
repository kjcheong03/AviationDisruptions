with source AS (
    SELECT * FROM {{ ref('bts_sample' ) }}
),

cleaned AS (
    SELECT 
        -- produce unique flight id for EACH FLIGHT 
        CONCAT(
            CAST(FlightDate AS STRING), '_', 
            Reporting_Airline, '_', 
            CAST(Flight_Number_Reporting_Airline AS STRING), '_', 
            Origin, '_', 
            CAST(CRSDepTime AS STRING)
        ) AS flight_id,

        -- date and time
        CAST(FlightDate AS DATE) AS flight_date,
        CAST(Month AS INT64) AS flight_month,
        CAST(DayOfWeek AS INT64) AS day_of_week,
        CAST(TRUNC(CAST(CRSDepTime AS INT64) / 100) AS INT64) AS scheduled_dep_hour,

        -- identifiers and route dim
        Reporting_Airline AS airline_code,
        CAST(Flight_Number_Reporting_Airline AS INT64) AS flight_number,
        Tail_Number AS tail_number,
        Origin AS origin_airport,
        Dest AS dest_airport,
        CAST(Distance AS FLOAT64) AS distance_miles,
        
        -- status of flight
        CAST(Cancelled AS BOOLEAN) AS is_cancelled,
        CancellationCode AS cancellation_code,

        -- other metrics
        CAST(TaxiOut AS FLOAT64) AS taxi_out_mins,
        CAST(TaxiIn AS FLOAT64) AS taxi_in_mins,
        
        CASE
            WHEN DepDelayMinutes > 720 THEN 720
            WHEN DepDelayMinutes < -720 THEN -720
            ELSE COALESCE(DepDelayMinutes, 0)
        END AS dep_delay_minutes,

        CASE
            WHEN ArrDelayMinutes > 720 THEN 720
            WHEN ArrDelayMinutes < -720 THEN -720
            ELSE COALESCE(ArrDelayMinutes, 0)
        END AS arr_delay_minutes,

        -- delay data breakdown
        COALESCE(WeatherDelay, 0) AS weather_delay_mins,
        COALESCE(NASDelay, 0) AS nas_delay_mins,
        COALESCE(CarrierDelay, 0) AS carrier_delay_mins,
        COALESCE(SecurityDelay, 0) AS security_delay_mins,
        COALESCE(LateAircraftDelay, 0) AS late_aircraft_delay_mins
    
    FROM source
)

SELECT * FROM cleaned