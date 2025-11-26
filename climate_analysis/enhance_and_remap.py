"""
This script runs AFTER the main pipeline (test_pipeline.py).
It "enhances" the station_data.csv by adding model-predicted slopes
from a NetCDF file. It then re-generates the map with this new data.
"""

import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Import the mapping function from your existing package
from climate_analysis import mapping

# --- Configuration: Define all file paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# Input files
STATION_CSV_INPUT = OUTPUT_DIR / "station_data.csv"
MODEL_DATA_FILE = Path("/home/gretar/fs06/bc3_urop/data/PatternScalingCoefficients_tas_ssp245-ssp370__r240x120.nc")

# Output files
STATION_CSV_OUTPUT = OUTPUT_DIR / "station_data.csv" # Overwrites the original
MAP_HTML_OUTPUT = OUTPUT_DIR / "world_climate_map.html" # Overwrites the map

def enhance_station_data(results_df: pd.DataFrame, model_data: xr.DataArray) -> pd.DataFrame:
    """
    Finds the nearest model slope for each station via spatial lookup.
    """
    print("Matching stations to nearest model grid points...")

    model_slopes = []

    # Check if model longitude is 0-360
    model_lons = model_data.coords['lon'].values
    model_uses_0_to_360 = np.min(model_lons) >= 0 and np.max(model_lons) > 180

    for index, station in tqdm(results_df.iterrows(), total=len(results_df), desc="Finding Model Slopes"):
        lat = station['Latitude']
        lon = station['Longitude']

        # Handle longitude conversion if model uses 0-360
        if model_uses_0_to_360 and lon < 0:
            lon = lon + 360 # Convert -180/180 to 0/360

        try:
            # Use xarray's .sel() with method='nearest' for spatial lookup
            model_val = model_data.sel(
                lat=lat,
                lon=lon,
                method='nearest'
            ).item() # .item() extracts the single float value

            model_slopes.append(model_val)
        except Exception as e:
            print(f"Warning: Could not find model data for {station['StationID']}. Error: {e}")
            model_slopes.append(np.nan)

    results_df['model_slope'] = model_slopes
    return results_df

if __name__ == "__main__":
    print("--- Starting CSV Enhancement Script ---")

    # 1. Check for required files
    if not STATION_CSV_INPUT.exists():
        print(f"FATAL ERROR: Input file not found: {STATION_CSV_INPUT}")
        print("Please run 'test_pipeline.py' first to generate this file.")
        sys.exit(1)

    if not MODEL_DATA_FILE.exists():
        print(f"FATAL ERROR: Model data file not found: {MODEL_DATA_FILE}")
        print("Please add your NetCDF file to the 'data' directory.")
        print("You can run 'create_dummy_model_data.py' to generate a test file.")
        sys.exit(1)

    # 2. Load the data
    print(f"Loading station results from {STATION_CSV_INPUT}")
    results_df = pd.read_csv(STATION_CSV_INPUT)

    print(f"Loading model data from {MODEL_DATA_FILE}")
    # We assume the variable name in the NetCDF is 'slope'
    model_data_xr = xr.open_dataset(MODEL_DATA_FILE)

    if 'slope' not in model_data_xr:
        print(f"FATAL ERROR: Variable 'slope' not found in {MODEL_DATA_FILE}.")
        sys.exit(1)

    model_data = model_data_xr['slope']

    # 3. Perform enhancement
    results_df = enhance_station_data(results_df, model_data)

    # 4. Save enhanced CSV
    print(f"Saving enhanced data back to {STATION_CSV_OUTPUT}...")
    results_df.to_csv(STATION_CSV_OUTPUT, index=False)

    # 5. Re-generate the map
    print(f"Re-generating interactive map at {MAP_HTML_OUTPUT}...")
    map_station_data = results_df.to_dict('records')

    mapping.create_station_map(map_station_data, MAP_HTML_OUTPUT)

    print("\n--- Enhancement and Remapping Finished Successfully! ---")
