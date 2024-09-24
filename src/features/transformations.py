import pandas as pd
from haversine import haversine

from src.utils.time import robust_hour_of_iso_date

def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_distance"] = df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ),
        axis=1,
    )
    return df

def hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    df["event_hour"] = df["event_timestamp"].apply(robust_hour_of_iso_date)
    return df

def driver_historical_completed_bookings(df: pd.DataFrame) -> pd.DataFrame:
    # Load the participant log data (make sure this path points to the correct location of participant_log.csv)
    participant_log = pd.read_csv("data/raw/participant_log.csv")
    
    # Filter the rows where the status is 'completed'
    completed_bookings = participant_log[participant_log['participant_status'] == 'COMPLETED']
    
    # Group by driver_id and count the number of completed bookings for each driver
    completed_bookings_count = completed_bookings.groupby('driver_id').size().reset_index(name='completed_bookings_count')
    
    # Merge this information back into the dataset (df)
    df = df.merge(completed_bookings_count, on='driver_id', how='left')
    
    # Fill any NaN values with 0 (for drivers with no historical completed bookings)
    df['completed_bookings_count'] = df['completed_bookings_count'].fillna(0)
    
    return df

def driver_acceptance_rate(df: pd.DataFrame) -> pd.DataFrame:
    participant_log = pd.read_csv("data/raw/participant_log.csv")
    
    # Calculate total participations and acceptances per driver
    total_events = participant_log.groupby('driver_id').size().reset_index(name='total_events')
    accepted_events = participant_log[participant_log['participant_status'] == 'ACCEPTED'].groupby('driver_id').size().reset_index(name='accepted_events')
    
    # Merge data and calculate acceptance rate
    acceptance_rate = pd.merge(total_events, accepted_events, on='driver_id', how='left')
    acceptance_rate['acceptance_rate'] = acceptance_rate['accepted_events'] / acceptance_rate['total_events']
    
    # Fill missing values with 0
    acceptance_rate['acceptance_rate'] = acceptance_rate['acceptance_rate'].fillna(0)
    
    # Merge acceptance rate back into the main dataset
    df = df.merge(acceptance_rate[['driver_id', 'acceptance_rate']], on='driver_id', how='left')
    return df
