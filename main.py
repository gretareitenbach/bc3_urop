"""
Main script to run the end-to-end climate analysis pipeline.
- Loads global and local station data.
- Processes each station, applying quality control gates.
- Performs regression analysis and generates diagnostic plots.
- Saves a summary CSV of all valid stations.
- Creates a final interactive world map.
"""

import pandas as pd
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
STATION_INV_FILE = DATA_DIR / "ghcnm.tavg.v4.0.1.20251007.qcf.inv"
STATION_DATA_FILE = DATA_DIR / "ghcnm.tavg.v4.0.1.20251007.qcf.dat"

# Output file paths
STATION_CSV_OUTPUT = OUTPUT_DIR / "station_data.csv"
MAP_HTML_OUTPUT = OUTPUT_DIR / "world_climate_map.html"

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

        # Consolidate all results into one dictionary
        station_result = {
            **station.to_dict(), # Station metadata (ID, Name, Lat, Lon, Elev)
            **regression_stats, # slope, intercept, r_squared
            'regression_plot': reg_plot_path.relative_to(OUTPUT_DIR).as_posix(),
            'residual_plot': res_plot_path.relative_to(OUTPUT_DIR).as_posix(),
            'timeseries_plot': ts_plot_path.relative_to(OUTPUT_DIR).as_posix(),
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
