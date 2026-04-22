"""
Main script to run the end-to-end climate analysis pipeline.
- Loads global and local station data.
- Processes each station, applying quality control gates.
- Performs regression analysis and generates diagnostic plots.
- Saves a summary CSV of all valid stations.
- Creates a final interactive world map.
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm  # A library for smart progress bars

# Import all our custom modules from the climate_analysis package
from climate_analysis import data_loader, analysis, plotting, mapping

# --- Configuration: Define all file paths and settings ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
PLOT_DIR = OUTPUT_DIR / "plots"

# --- UPDATED FILE NAMES ---
GLOBAL_DATA_FILE = DATA_DIR / "global_temps.txt"
STATION_INV_FILE = DATA_DIR / "ghcnm.tavg.v4.0.1.20260224.qcf.inv"
STATION_DATA_FILE = DATA_DIR / "ghcnm.tavg.v4.0.1.20260224.qcf.dat"

# Output file paths
STATION_CSV_OUTPUT = OUTPUT_DIR / "station_data.csv"
MAP_HTML_OUTPUT = OUTPUT_DIR / "maps" / "world_climate_map.html"

# Climatological data files
HUMIDITY_FILE = DATA_DIR / "grid_10min_reh.dat"

# --- Helper Functions for Climatological Features ---

def calculate_climo_temp(station_id: str, station_dat_file: Path) -> float:
    """
    Calculates the 1961-present average temperature for a single station.
    """
    if not station_dat_file.exists():
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
        print(f"WARNING: Humidity file not found: {humidity_file_path}")
        print("Climatological humidity features will be set to NaN.")
        return None

    print(f"Loading humidity grid data from {humidity_file_path}...")

    try:
        # Read the file into a Pandas DataFrame
        df = pd.read_csv(
            humidity_file_path,
            sep='\s+',
            header=None,
            names=['lat', 'lon', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            encoding='latin-1'
        )

        print(f"-> Loaded raw table with {len(df)} grid points.")

        # Convert DataFrame to Xarray
        df = df.set_index(['lat', 'lon'])
        ds = df.to_xarray()

        # Stack the month variables into a single dimension
        humidity_da = ds.to_array(dim='month', name='rhm')

        # Sort by lat/lon to ensure 'method=nearest' works correctly later
        humidity_da = humidity_da.sortby(['lat', 'lon'])

        print(f"-> Converted to Xarray: {humidity_da.dims} {humidity_da.shape}")
        return humidity_da

    except Exception as e:
        print(f"WARNING: Could not parse humidity file: {e}")
        print("Climatological humidity features will be set to NaN.")
        return None

def get_climo_humidity(lat: float, lon: float, humidity_data: xr.DataArray) -> float:
    """
    Finds the nearest grid cell and calculates the annual mean
    from the 12 monthly climatology values.
    """
    if humidity_data is None:
        return np.nan

    try:
        # Select the nearest grid point's 12-month climatology
        station_monthly_climo = humidity_data.sel(
            lat=lat,
            lon=lon,
            method='nearest'
        )

        # Calculate the annual mean from the 12 monthly values
        annual_mean = station_monthly_climo.mean().item()

        if np.isnan(annual_mean):
            return np.nan

        return annual_mean

    except Exception as e:
        return np.nan

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure output directories exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)

    print("--- Climate Analysis Pipeline Started ---")

    # 1. Load the foundational data
    print(f"Loading global data from: {GLOBAL_DATA_FILE}")
    global_data = data_loader.load_global_data(GLOBAL_DATA_FILE)

    print(f"Loading station inventory from: {STATION_INV_FILE}")
    station_inventory = data_loader.load_station_inventory(STATION_INV_FILE)

    # Load climatological humidity data
    humidity_data = load_humidity_data(HUMIDITY_FILE)

    # 2. Process each station
    print(f"Starting analysis for {len(station_inventory)} stations...")
    all_station_results = []

    # Using tqdm for a progress bar
    for index, station in tqdm(station_inventory.iterrows(), total=len(station_inventory), desc="Processing Stations"):
        station_id = station['StationID']

        # Load data for the individual station
        local_data = data_loader.load_local_station_data(STATION_DATA_FILE, station_id)

        # Merge data and apply quality gates (>= 50 years, post-1930)
        merged_data = analysis.get_merged_data(local_data, global_data)

        # If merged_data is None, it means the station failed the quality check
        if merged_data is None:
            continue

        # Perform the core analysis
        merged_data = analysis.calculate_local_anomalies(merged_data)
        regression_stats = analysis.perform_regression(merged_data)

        # Generate all three plots
        reg_plot_path = plotting.create_regression_plot(merged_data, regression_stats, station_id, PLOT_DIR)
        res_plot_path = plotting.create_residual_plot(merged_data, regression_stats, station_id, PLOT_DIR)
        ts_plot_path = plotting.create_timeseries_plot(merged_data, station_id, PLOT_DIR)

        # Calculate climatological features
        climo_temp = calculate_climo_temp(station_id, STATION_DATA_FILE)
        climo_humidity = get_climo_humidity(station['Latitude'], station['Longitude'], humidity_data)

        # Consolidate all results into one dictionary
        station_result = {
            **station.to_dict(), # Station metadata (ID, Name, Lat, Lon, Elev)
            **regression_stats, # slope, intercept, r_squared
            'regression_plot': reg_plot_path.relative_to(OUTPUT_DIR).as_posix(),
            'residual_plot': res_plot_path.relative_to(OUTPUT_DIR).as_posix(),
            'timeseries_plot': ts_plot_path.relative_to(OUTPUT_DIR).as_posix(),
            'climo_temp': climo_temp,
            'climo_humidity': climo_humidity,
        }
        # We don't need the full residuals Series in our final CSV/map data
        del station_result['residuals']

        all_station_results.append(station_result)

    print(f"\nSuccessfully processed {len(all_station_results)} stations that met the quality criteria.")

    # 3. Save results to CSV and create the map
    if not all_station_results:
        print("No stations passed the quality gates. Exiting.")
    else:
        # Save the detailed results to a CSV file
        print(f"Saving summary data to {STATION_CSV_OUTPUT}...")
        results_df = pd.DataFrame(all_station_results)
        results_df.to_csv(STATION_CSV_OUTPUT, index=False)

        # Create the final interactive map
        print(f"Creating interactive map at {MAP_HTML_OUTPUT}...")
        # Use relative paths for the map popups
        map_station_data = results_df.to_dict('records')
        for record in map_station_data:
            # The map HTML will be in output/, so plots need to be relative to that
            record['regression_plot'] = Path(record['regression_plot']).relative_to(Path('.')).as_posix()
            record['residual_plot'] = Path(record['residual_plot']).relative_to(Path('.')).as_posix()
            record['timeseries_plot'] = Path(record['timeseries_plot']).relative_to(Path('.')).as_posix()

        mapping.create_station_map(map_station_data, MAP_HTML_OUTPUT)

    print("\n--- Pipeline Finished Successfully! ---")
