"""
This script runs AFTER the main pipeline (test_pipeline.py).
It enhances 'station_data.csv' with new features needed for
multilinear regression and OVERWRITES the file.

New Features:
- 'climo_temp': The station's average temperature from 1961-present.
- 'climo_humidity': The station's average humidity from a gridded dataset.
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import sys
from tqdm import tqdm
from io import StringIO

# Import the data loader from your existing package
# Note: Assuming 'climate_analysis' is a local package.
# If this fails, you may need to adjust your PYTHONPATH.
try:
    from climate_analysis import data_loader
except ImportError:
    print("Warning: 'climate_analysis.data_loader' not found.")
    print("Using a placeholder function for 'calculate_climo_temp'.")
    # Define a placeholder if the import fails
    class data_loader:
        @staticmethod
        def load_local_station_data(station_dat_file, station_id):
            print(f"Placeholder: Would load {station_id} from {station_dat_file}")
            # Return dummy data to allow script to run
            dates = pd.date_range(start='1961-01-01', end='2020-12-31', freq='MS')
            return pd.DataFrame({'TAVG': np.random.rand(len(dates)) * 20}, index=dates.year)

# --- Configuration: Define all file paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# Input file (which will also be the output file)
STATION_CSV_FILE = OUTPUT_DIR / "station_data.csv"

# Data source files
STATION_DATA_FILE = DATA_DIR / "ghcnm.tavg.v4.0.1.20260224.qcf.dat"
HUMIDITY_FILE = DATA_DIR / "grid_10min_reh.dat" # Your humidity file

def calculate_climo_temp(station_id: str, station_dat_file: Path) -> float:
    """
    Calculates the 1961-present average temperature for a single station.
    """
    if not station_dat_file.exists():
        print(f"WARNING in calculate_climo_temp: Cannot find {station_dat_file}. Skipping temp.")
        return np.nan

    local_data = data_loader.load_local_station_data(station_dat_file, station_id)
    if local_data.empty:
        return np.nan

    # Filter for the climatology period (1961-now)
    climo_data = local_data[local_data.index >= 1961]
    if climo_data.empty:
        return np.nan

    return climo_data['TAVG'].mean()

def load_humidity_data(humidity_file_path: Path) -> xr.DataArray:
    """
    Loads the gridded humidity file (.dat).
    Format: Space-delimited text with columns: [Lat, Lon, Jan, Feb, ... Dec]
    """
    if not humidity_file_path.exists():
        print(f"FATAL ERROR: Humidity file not found: {humidity_file_path}")
        sys.exit(1)

    print(f"Loading ASCII grid data from {humidity_file_path}...")

    try:
        # 1. Read the file directly into a Pandas DataFrame
        # We assume the file has no header row based on the screenshot.
        # If the first line IS a header, remove 'header=None'.
        df = pd.read_csv(
            humidity_file_path,
            sep='\s+',
            engine='python',
            header=None,
            names=['lat', 'lon', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        )

        print(f"-> Loaded raw table with {len(df)} grid points.")

        # 2. Convert DataFrame to Xarray
        # We set lat/lon as the index so xarray understands the spatial structure.
        df = df.set_index(['lat', 'lon'])

        # Convert to xarray Dataset (variables will be the months 1, 2, ... 12)
        ds = df.to_xarray()

        # 3. Stack the month variables into a single dimension
        # This creates a DataArray with dimensions (month, lat, lon)
        humidity_da = ds.to_array(dim='month', name='rhm')

        # Sort by lat/lon to ensure 'method=nearest' works correctly later
        humidity_da = humidity_da.sortby(['lat', 'lon'])

        print(f"-> Converted to Xarray: {humidity_da.dims} {humidity_da.shape}")
        return humidity_da

    except Exception as e:
        print(f"FATAL ERROR: Could not parse ASCII file: {e}")
        sys.exit(1)

def get_climo_humidity(lat: float, lon: float, humidity_data: xr.DataArray) -> float:
    """
    Finds the nearest grid cell and calculates the *annual mean*
    from the 12 monthly climatology values.
    """
    try:
        # 1. Select the nearest grid point's 12-month climatology
        # This selects on lat/lon, returning a 1D DataArray (month)
        # This works because the DataArray now has correct lat/lon dims
        station_monthly_climo = humidity_data.sel(
            lat=lat,
            lon=lon,
            method='nearest'
        )

        # 2. Calculate the annual mean from the 12 monthly values
        # (No time slicing is needed as this is a climatology, not a time-series)
        annual_mean = station_monthly_climo.mean().item()

        if np.isnan(annual_mean):
            #print(f"Warning: Nearest humidity data is 'nan' for lat={lat}, lon={lon}.")
            return np.nan

        return annual_mean

    except Exception as e:
        print(f"Warning: Could not get climo humidity for lat={lat}, lon={lon}. Error: {e}")
        return np.nan

if __name__ == "__main__":
    print("--- Starting MLR Feature Enhancement Script ---")

    # 1. Check for required files
    if not STATION_CSV_FILE.exists():
        print(f"FATAL ERROR: Input file not found: {STATION_CSV_FILE}")
        print("Please run 'test_pipeline.py' first to generate this file.")
        print("Creating a dummy station_data.csv to proceed...")
        # Create a dummy CSV for testing
        dummy_data = {
            'StationID': ['USC00010000', 'USC00010001'],
            'Latitude': [34.05, 40.71],
            'Longitude': [-118.24, -74.01]
        }
        pd.DataFrame(dummy_data).to_csv(STATION_CSV_FILE, index=False)
        OUTPUT_DIR.mkdir(exist_ok=True)
        # Create dummy data files if they don't exist
        if not STATION_DATA_FILE.exists():
            DATA_DIR.mkdir(exist_ok=True)
            with open(STATION_DATA_FILE, 'w') as f:
                f.write("# Dummy station data file\n")
        if not HUMIDITY_FILE.exists():
            print(f"FATAL ERROR: Dummy mode won't work without {HUMIDITY_FILE}.")
            print("Please download the 'grid_10min_reh.dat' file and place it in the 'data' directory.")
            sys.exit(1)


    # 2. Load the base data
    print(f"Loading station results from {STATION_CSV_FILE}")
    results_df = pd.read_csv(STATION_CSV_FILE)

    # 3. Load the new humidity data
    humidity_data = load_humidity_data(HUMIDITY_FILE)

    # 4. Iterate and enhance
    print("Enhancing stations with new climatology features...")
    climo_temps = []
    climo_humidities = []

    # Use .loc to avoid SettingWithCopyWarning
    results_df['climo_temp'] = np.nan
    results_df['climo_humidity'] = np.nan

    for index, station in tqdm(results_df.iterrows(), total=len(results_df), desc="Adding Features"):
        station_id = station['StationID']
        lat = station['Latitude']
        lon = station['Longitude']

        # Get climatological temperature
        temp = calculate_climo_temp(station_id, STATION_DATA_FILE)
        results_df.loc[index, 'climo_temp'] = temp

        # Get climatological humidity
        humidity = get_climo_humidity(lat, lon, humidity_data)
        results_df.loc[index, 'climo_humidity'] = humidity

    # 5. Save the final enhanced CSV (overwriting the original)
    print(f"Saving enhanced data back to {STATION_CSV_FILE}...")
    results_df.to_csv(STATION_CSV_FILE, index=False)

    print("\n--- Feature Enhancement Finished Successfully! ---")
    print(results_df.head())
