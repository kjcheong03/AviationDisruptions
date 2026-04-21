with source_raw AS (
    SELECT -- add columns required
        flightdate AS FlightDate,
        CAST(EXTRACT(DAYOFWEEK FROM CAST(flightdate AS DATE)) AS INT64) AS DayOfWeek,
        reporting_airline AS Reporting_Airline,
        flight_number_reporting_airline AS Flight_Number_Reporting_Airline,
        tail_number AS Tail_Number,
        origin AS Origin,
        dest AS Dest,
        crsdeptime AS CRSDepTime,
        month AS Month,
        distance AS Distance,
        cancelled AS Cancelled,
        cancellationcode AS CancellationCode,
        CAST(NULL AS FLOAT64) AS TaxiOut, -- not available in current (rawnet data)
        CAST(NULL AS FLOAT64) AS TaxiIn, -- missing taxiIn handled
        depdelay AS DepDelayMinutes,
        arrdelay AS ArrDelayMinutes,
        weatherdelay AS WeatherDelay,
        nasdelay AS NASDelay,
        carrierdelay AS CarrierDelay,
        securitydelay AS SecurityDelay,
        lateaircraftdelay AS LateAircraftDelay
    FROM {{ source('raw_aviation', 'bts_raw') }}
),

source AS (
    -- Safety net: collapse exact duplicate raw rows before downstream feature engineering.
    SELECT DISTINCT *
    FROM source_raw
),

cleaned AS (
    SELECT 
        -- produce unique flight id for EACH FLIGHT 
        CONCAT(
            COALESCE(CAST(FlightDate AS STRING), 'NA'), '_', 
            COALESCE(Reporting_Airline, 'NA'), '_', 
            COALESCE(CAST(Flight_Number_Reporting_Airline AS STRING), 'NA'), '_', 
            COALESCE(Origin, 'NA'), '_', 
            COALESCE(CAST(CRSDepTime AS STRING), 'NA')
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
        CASE
            WHEN Cancelled = 1 THEN TRUE
            WHEN Cancelled = 0 THEN FALSE
            ELSE NULL
        END AS is_cancelled,
        CancellationCode AS cancellation_code,

        -- other metrics -- not available at the moment
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