"""
Handles all data loading and parsing for the climate analysis project.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def load_global_data(filepath: str) -> pd.DataFrame:
    """
    Parses the NASA/GISTEMP annual anomaly file (e.g., global_temps.txt).

    Args:
        filepath: The path to the global temperature data file.

    Returns:
        A pandas DataFrame with 'Year' and 'AnnualAnomaly' columns.
    """
    years = []
    annual_avg = []

    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Each year has two lines; we only need the first (anomaly data)
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) >= 14: # Year + 12 months + annual average
            years.append(int(parts[0]))
            annual_avg.append(float(parts[-1]))

    df = pd.DataFrame({
        'Year': years,
        'AnnualAnomaly': annual_avg
    })
    return df.set_index('Year')

def load_station_inventory(filepath: str) -> pd.DataFrame:
    """
    Parses a GHCN-M station inventory file.

    Args:
        filepath: The path to the station inventory file (.inv).

    Returns:
        A pandas DataFrame with station metadata.
    """
    stations = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) >= 5:
                station_id = parts[0]
                try:
                    lat = float(parts[1])
                    lon = float(parts[2])
                    elev = float(parts[3])
                    name = " ".join(parts[4:])
                    stations.append({
                        'StationID': station_id,
                        'Latitude': lat,
                        'Longitude': lon,
                        'Elevation': elev,
                        'StationName': name
                    })
                except (ValueError, IndexError):
                    continue # Skip malformed lines

    return pd.DataFrame(stations)

def load_local_station_data(filepath: str, station_id: str) -> pd.DataFrame:
    """
    Extracts, cleans, and calculates annual averages for a single GHCN-M station.

    Args:
        filepath: The path to the GHCN-M data file (.dat).
        station_id: The ID of the station to extract.

    Returns:
        A pandas DataFrame with 'Year' and 'TAVG' (annual average temperature)
        for the specified station, or an empty DataFrame if no valid data is found.
    """
    yearly_data = {}
    with open(filepath, 'r') as f:
        for line in f:
            current_id = line[0:11].strip()
            element = line[15:19].strip()

            if current_id == station_id and element == 'TAVG':
                year = int(line[11:15].strip())
                monthly_vals = []
                for i in range(12):
                    val_str = line[19 + i*8 : 24 + i*8].strip()
                    if val_str == '-9999':
                        monthly_vals.append(np.nan)
                    else:
                        monthly_vals.append(float(val_str) / 100.0) # Convert from hundredths of °C

                # Calculate annual average only if all 12 months are present
                if not np.isnan(monthly_vals).any():
                    yearly_data[year] = np.mean(monthly_vals)

    if not yearly_data:
        return pd.DataFrame()

    df = pd.DataFrame(list(yearly_data.items()), columns=['Year', 'TAVG'])
    return df.set_index('Year').sort_index()
