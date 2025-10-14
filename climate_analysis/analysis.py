"""
Handles all statistical and scientific analysis for the climate project.
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, Tuple, Optional

def get_merged_data(
    local_data: pd.DataFrame,
    global_data: pd.DataFrame,
    min_year: int = 1930,
    min_data_points: int = 50
) -> Optional[pd.DataFrame]:
    """
    Merges local and global data, filters by year, and ensures enough data points.

    Args:
        local_data: DataFrame with local station TAVG data, indexed by Year.
        global_data: DataFrame with global anomaly data, indexed by Year.
        min_year: The first year to include in the analysis (e.g., 1930).
        min_data_points: The minimum number of overlapping years required.

    Returns:
        A merged DataFrame with 'TAVG' and 'AnnualAnomaly' columns,
        or None if the station fails the quality gates.
    """
    if local_data.empty:
        return None

    # Merge the two dataframes on their common index (Year)
    merged_df = local_data.join(global_data, how='inner')

    # Filter for years after the minimum year
    merged_df = merged_df[merged_df.index >= min_year]

    # Check if we have enough data
    if len(merged_df) < min_data_points:
        return None

    return merged_df

def calculate_local_anomalies(merged_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates local temperature anomalies based on the station's own baseline.

    Args:
        merged_data: A DataFrame containing a 'TAVG' column.

    Returns:
        The DataFrame with a new 'LocalAnomaly' column.
    """
    # The baseline is the average temperature over the entire period of record
    baseline = merged_data['TAVG'].mean()
    merged_data['LocalAnomaly'] = merged_data['TAVG'] - baseline
    return merged_data

def perform_regression(merged_data: pd.DataFrame) -> Dict:
    """
    Performs linear regression between local and global anomalies.

    Args:
        merged_data: DataFrame containing 'LocalAnomaly' and 'GlobalAnomaly'
                     (renamed from 'AnnualAnomaly' for clarity).

    Returns:
        A dictionary containing regression statistics: slope, intercept,
        r_squared, and the residuals as a pandas Series.
    """
    # For clarity, let's rename the global anomaly column
    data = merged_data.rename(columns={'AnnualAnomaly': 'GlobalAnomaly'})

    global_vals = data['GlobalAnomaly'].values
    local_vals = data['LocalAnomaly'].values

    # Use numpy's polyfit for linear regression (degree 1)
    coeffs = np.polyfit(global_vals, local_vals, deg=1)
    slope, intercept = coeffs[0], coeffs[1]

    # Calculate R-squared value
    r_squared = np.corrcoef(global_vals, local_vals)[0, 1] ** 2

    # Calculate residuals
    predicted_local_vals = slope * global_vals + intercept
    residuals = local_vals - predicted_local_vals

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'residuals': pd.Series(residuals, index=data.index)
    }

def calculate_time_trends(merged_data: pd.DataFrame) -> Dict:
    """
    Calculates the linear trend over time (°C/year) for both local and global data.

    Args:
        merged_data: A DataFrame with 'LocalAnomaly' and 'AnnualAnomaly'.

    Returns:
        A dictionary with 'local_trend_slope' and 'global_trend_slope'.
    """
    years = merged_data.index.values

    # Local trend
    local_coeffs = np.polyfit(years, merged_data['LocalAnomaly'], deg=1)

    # Global trend
    global_coeffs = np.polyfit(years, merged_data['AnnualAnomaly'], deg=1)

    return {
        'local_trend_slope': local_coeffs[0],
        'global_trend_slope': global_coeffs[0]
    }

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the great-circle distance in miles between two points on Earth.

    Args:
        lat1, lon1: Latitude and longitude of point 1.
        lat2, lon2: Latitude and longitude of point 2.

    Returns:
        The distance in miles.
    """
    R = 3958.8  # Earth's radius in miles

    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(math.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c
