import pandas as pd
import os

print("Starting Parquet to CSV conversion for dbt seeds...")

# Create the seeds directory if it doesn't exist
os.makedirs('dbt/seeds', exist_ok=True)

# 1. Convert the Weather Data
try:
    weather_df = pd.read_parquet('eda/cache/weather_sample.parquet')
    
    # Format datetime perfectly for BigQuery (adds the missing :00 seconds)
    weather_df['time'] = pd.to_datetime(weather_df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    weather_df.to_csv('dbt/seeds/weather_sample.csv', index=False, float_format='%g')
    print("Weather data converted successfully!")
except Exception as e:
    print(f"Weather error: {e}")

# 2. Convert a small sample of the July BTS Data
try:
    # remove columns not required - based on EDA
    cols_to_keep = [
        'FlightDate', 'Reporting_Airline', 'Flight_Number_Reporting_Airline', 
        'Origin', 'Dest', 'CRSDepTime', 'DepTime', 'DepDelay', 'DepDelayMinutes', 
        'CRSArrTime', 'ArrTime', 'ArrDelay', 'ArrDelayMinutes', 'Cancelled', 
        'CancellationCode', 'WeatherDelay', 'NASDelay', 'CarrierDelay', 
        'SecurityDelay', 'LateAircraftDelay',
        # recommended for feature engineering
        'DayOfWeek', 'Month', 'Distance', 'TaxiOut', 'TaxiIn'
    ]
    
    # Read, filter columns, and grab 1000 rows
    bts_df = pd.read_parquet('eda/cache/bts_2024_7.parquet')[cols_to_keep].head(1000)
    
    bts_df.to_csv('dbt/seeds/bts_sample.csv', index=False, float_format='%g')
    print("BTS data stripped of junk and converted successfully!")
except Exception as e:
    print(f"BTS error: {e}")

print("Finished! You can now run `dbt seed`.")