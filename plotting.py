"""
Handles the generation of all diagnostic plots.
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# Use the 'Agg' backend for Matplotlib. This is a non-interactive backend
# that is perfect for saving plots to files without displaying a GUI window.
# This is crucial for running a script that may generate hundreds of plots.
matplotlib.use("Agg")

def create_regression_plot(
    data: pd.DataFrame,
    stats: dict,
    station_id: str,
    output_dir: Path
) -> Path:
    """
    Creates a scatter plot of Local vs. Global anomalies with a regression line.

    Args:
        data: DataFrame containing 'LocalAnomaly' and 'AnnualAnomaly'.
        stats: Dictionary containing regression 'slope' and 'intercept'.
        station_id: The ID of the station.
        output_dir: The base directory to save the plot in.

    Returns:
        The path to the saved plot file.
    """
    global_vals = data['AnnualAnomaly']
    local_vals = data['LocalAnomaly']

    # Generate the regression line using slope and intercept
    fit_line = stats['slope'] * global_vals + stats['intercept']

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.scatter(global_vals, local_vals, s=20, alpha=0.7)
    plt.plot(global_vals, fit_line, color="red", ls="--", lw=2)

    plt.title(f"Regression Analysis for {station_id}", fontsize=14)
    plt.xlabel("Global Temperature Anomaly (°C)", fontsize=12)
    plt.ylabel("Local Temperature Anomaly (°C)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Ensure output directory exists
    station_plot_dir = output_dir / station_id
    station_plot_dir.mkdir(parents=True, exist_ok=True)

    # Save figure
    filepath = station_plot_dir / "regression.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close() # Close the plot to free up memory

    return filepath

def create_residual_plot(
    data: pd.DataFrame,
    stats: dict,
    station_id: str,
    output_dir: Path
) -> Path:
    """
    Creates a scatter plot of regression residuals vs. Global anomalies.

    Args:
        data: DataFrame containing 'AnnualAnomaly'.
        stats: Dictionary containing the 'residuals' Series.
        station_id: The ID of the station.
        output_dir: The base directory to save the plot in.

    Returns:
        The path to the saved plot file.
    """
    global_vals = data['AnnualAnomaly']
    residuals = stats['residuals']

    plt.figure(figsize=(8, 6))
    plt.scatter(global_vals, residuals, s=20, alpha=0.7, c='green')
    plt.axhline(0, color="red", ls="--", lw=2) # Line at y=0 for reference

    plt.title(f"Residuals for {station_id}", fontsize=14)
    plt.xlabel("Global Temperature Anomaly (°C)", fontsize=12)
    plt.ylabel("Residual (Local - Fit) (°C)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    station_plot_dir = output_dir / station_id
    station_plot_dir.mkdir(parents=True, exist_ok=True)

    filepath = station_plot_dir / "residual.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath

def create_timeseries_plot(
    data: pd.DataFrame,
    station_id: str,
    output_dir: Path
) -> Path:
    """
    Creates a time-series plot of local and global anomalies with trend lines.

    Args:
        data: DataFrame containing 'LocalAnomaly' and 'AnnualAnomaly', indexed by Year.
        station_id: The ID of the station.
        output_dir: The base directory to save the plot in.

    Returns:
        The path to the saved plot file.
    """
    years = data.index

    # Calculate trend lines
    local_coeffs = np.polyfit(years, data['LocalAnomaly'], 1)
    global_coeffs = np.polyfit(years, data['AnnualAnomaly'], 1)
    local_trend = np.poly1d(local_coeffs)(years)
    global_trend = np.poly1d(global_coeffs)(years)

    plt.figure(figsize=(10, 6))

    # Plot anomalies
    plt.plot(years, data['AnnualAnomaly'], label="Global Anomaly", lw=1.5, color='gray')
    plt.plot(years, data['LocalAnomaly'], label="Local Anomaly", lw=1.5, color='blue')

    # Plot trend lines
    plt.plot(years, global_trend, ls='--', color='red', label='Global Trend')
    plt.plot(years, local_trend, ls='--', color='cyan', label='Local Trend')

    plt.title(f"Local vs. Global Anomalies for {station_id}", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Temperature Anomaly (°C)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    station_plot_dir = output_dir / station_id
    station_plot_dir.mkdir(parents=True, exist_ok=True)

    filepath = station_plot_dir / "timeseries.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath
