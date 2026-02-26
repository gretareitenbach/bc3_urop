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
from tqdm import tqdm
import sys

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
MAP_HTML_OUTPUT = OUTPUT_DIR / "world_climate_map.html"

# --- Main Execution ---
if __name__ == "__main__":
    # --- NEW: Add safety checks for input files ---
    required_files = [GLOBAL_DATA_FILE, STATION_INV_FILE, STATION_DATA_FILE]
    for f in required_files:
        if not f.exists():
            print(f"FATAL ERROR: Required data file not found at: {f}")
            print("Please download the necessary files and place them in the 'data' directory.")
            sys.exit(1) # Exit the script

    # Ensure output directories exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)

    print("--- Climate Analysis Pipeline Started ---")

    # 1. Load the foundational data
    print(f"Loading global data from: {GLOBAL_DATA_FILE}")
    global_data = data_loader.load_global_data(GLOBAL_DATA_FILE)
    print(f"-> Loaded {len(global_data)} years of global data.")

    print(f"Loading station inventory from: {STATION_INV_FILE}")
    station_inventory = data_loader.load_station_inventory(STATION_INV_FILE)
    print(f"-> Loaded {len(station_inventory)} stations in inventory.")

    # --- NEW: Limit to 5 random stations for testing ---
    if len(station_inventory) > 5:
        print("\n--- RUNNING IN TEST MODE: Selecting 5 random stations. ---")
        station_inventory = station_inventory.sample(n=5, random_state=42) # random_state makes the sample reproducible
        print(f"-> Sampled stations: {station_inventory['StationID'].tolist()}")
    else:
        print("\n--- Inventory has 5 or fewer stations, running on all. ---")
    # --- END OF NEW SECTION ---


    # 2. Process each station
    print(f"\nStarting analysis...")
    all_station_results = []

    for index, station in tqdm(station_inventory.iterrows(), total=len(station_inventory), desc="Processing Stations"):
        station_id = station['StationID']

        local_data = data_loader.load_local_station_data(STATION_DATA_FILE, station_id)
        merged_data = analysis.get_merged_data(local_data, global_data)

        if merged_data is None:
            continue

        merged_data = analysis.calculate_local_anomalies(merged_data)
        regression_stats = analysis.perform_regression(merged_data)

        reg_plot_path = plotting.create_regression_plot(merged_data, regression_stats, station_id, PLOT_DIR)
        res_plot_path = plotting.create_residual_plot(merged_data, regression_stats, station_id, PLOT_DIR)
        ts_plot_path = plotting.create_timeseries_plot(merged_data, station_id, PLOT_DIR)

        station_result = {
            **station.to_dict(),
            **regression_stats,
            'regression_plot': reg_plot_path.relative_to(OUTPUT_DIR).as_posix(),
            'residual_plot': res_plot_path.relative_to(OUTPUT_DIR).as_posix(),
            'timeseries_plot': ts_plot_path.relative_to(OUTPUT_DIR).as_posix(),
        }
        del station_result['residuals']
        all_station_results.append(station_result)

    print(f"\nSuccessfully processed {len(all_station_results)} stations that met the quality criteria.")

    # 3. Save results to CSV and create the map
    if not all_station_results:
        print("No stations passed the quality gates. Exiting.")
    else:
        print(f"Saving summary data to {STATION_CSV_OUTPUT}...")
        results_df = pd.DataFrame(all_station_results)
        results_df.to_csv(STATION_CSV_OUTPUT, index=False)

        print(f"Creating interactive map at {MAP_HTML_OUTPUT}...")
        map_station_data = results_df.to_dict('records')
        mapping.create_station_map(map_station_data, MAP_HTML_OUTPUT)

    print("\n--- Pipeline Finished Successfully! ---")
