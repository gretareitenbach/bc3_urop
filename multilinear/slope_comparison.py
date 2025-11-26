import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def compare_and_plot(original_data_file, regression_results_file, output_csv_file):
    """
    1. Calculates predicted slopes using regression coefficients.
    2. Saves the comparison data to a CSV.
    3. Generates and saves an 'Actual vs Predicted' scatter plot.
    4. Generates and saves a 'Residual Map' (Lat/Lon scatter).
    """
    print(f"--- Starting Analysis ---")
    print(f"Loading data from '{original_data_file}'...")

    try:
        df_data = pd.read_csv(original_data_file)
        df_coeffs = pd.read_csv(regression_results_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # --- Step 1: Predict Slopes ---
    try:
        # Convert coefficients dataframe to dictionary
        coeffs = pd.Series(
            df_coeffs['Coefficient (Slope)'].values,
            index=df_coeffs['Factor']
        ).to_dict()
    except KeyError:
        print("Error: Regression results file missing 'Factor' or 'Coefficient (Slope)' columns.")
        return

    # Start with intercept
    intercept = coeffs.get('const', 0.0)
    df_data['Predicted_Slope'] = intercept

    # Add contribution of each feature
    predictors = ['Latitude', 'Longitude', 'Elevation', 'climo_temp', 'climo_humidity']

    for feature in predictors:
        if feature in coeffs and feature in df_data.columns:
            df_data['Predicted_Slope'] += df_data[feature] * coeffs[feature]

    # Calculate Residuals
    df_data['Actual_Slope'] = df_data['slope']
    df_data['Residual'] = df_data['Actual_Slope'] - df_data['Predicted_Slope']

    # --- Step 2: Save CSV ---
    # Filter columns for cleaner output
    output_cols = [
        'StationID', 'StationName', 'Latitude', 'Longitude', 'Elevation',
        'Actual_Slope', 'Predicted_Slope', 'Residual'
    ]
    # Only keep columns that actually exist in the dataframe
    final_cols = [c for c in output_cols if c in df_data.columns]

    df_data[final_cols].to_csv(output_csv_file, index=False)
    print(f"Success: Data saved to '{output_csv_file}'")

    # --- Step 3: Plot Actual vs Predicted ---
    plt.figure(figsize=(10, 6))
    plt.scatter(df_data['Actual_Slope'], df_data['Predicted_Slope'], alpha=0.6, color='blue')

    # Draw a perfect prediction line (y=x)
    min_val = min(df_data['Actual_Slope'].min(), df_data['Predicted_Slope'].min())
    max_val = max(df_data['Actual_Slope'].max(), df_data['Predicted_Slope'].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')

    plt.title('Actual vs. Predicted Climate Change Slope')
    plt.xlabel('Actual Slope')
    plt.ylabel('Predicted Slope (from Spatial Model)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plot1_filename = 'output/multilinear/plot_actual_vs_predicted.png'
    os.makedirs(os.path.dirname(plot1_filename), exist_ok=True)
    plt.savefig(plot1_filename)
    plt.close()
    print(f"Success: Plot saved to '{plot1_filename}'")

    # --- Step 4: Plot Residual Map ---
    plt.figure(figsize=(12, 8))

    # Scatter plot: x=Lon, y=Lat, color=Residual
    # We use 'coolwarm' colormap so Blue = Negative Residual (Overpredicted), Red = Positive (Underpredicted)
    sc = plt.scatter(
        df_data['Longitude'],
        df_data['Latitude'],
        c=df_data['Residual'],
        cmap='coolwarm',
        alpha=0.8,
        edgecolors='k',
        linewidth=0.5
    )

    # Add a colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('Residual (Actual - Predicted)')

    plt.title('Spatial Residual Map\n(Red = Model Underestimated, Blue = Model Overestimated)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.5)

    plot2_filename = 'output/multilinear/plot_residual_map.png'
    os.makedirs(os.path.dirname(plot2_filename), exist_ok=True)
    plt.savefig(plot2_filename)
    plt.close()
    print(f"Success: Plot saved to '{plot2_filename}'")
    print("--- Analysis Complete ---")

if __name__ == "__main__":
    # --- USER CONFIGURATION ---
    ORIGINAL_DATA_CSV = 'output/station_data.csv'          # Your input data
    REGRESSION_RESULTS_CSV = 'output/spatial_analysis_simple.csv' # The coefficients file
    OUTPUT_COMPARISON_CSV = 'output/slope_comparison_simple.csv'    # The output CSV

    compare_and_plot(ORIGINAL_DATA_CSV, REGRESSION_RESULTS_CSV, OUTPUT_COMPARISON_CSV)
