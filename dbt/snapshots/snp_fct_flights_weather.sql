{% snapshot snp_fct_flights_weather %}

{{
    config(
        target_database='is3107-aviation-pipeline',
        target_schema='analytics_snapshots',
        unique_key='flight_id',
        strategy='check', 
        check_cols=[
            'dep_delay_minutes',
            'arr_delay_minutes',
            'weather_delay_mins',
            'temperature_c',
            'wind_speed_knots'
        ]
    )
}}

select * from {{ ref('fct_flights_weather') }}

{% endsnapshot %}