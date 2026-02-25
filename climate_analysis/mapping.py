"""
Handles the creation of interactive Folium maps.
"""

import pandas as pd
import folium
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
from pathlib import Path
from typing import List, Dict, Any

def _get_marker_shape(station_id: str) -> str:
    """Determines marker shape based on station code prefix."""
    # if station_id.startswith("USW"):
    #     return "triangle"
    if station_id.startswith("USC"):
        return "circle"
    return "circle" # Default shape

def _get_color_from_slope(slope: float, min_slope: float, max_slope: float) -> str:
    """
    Generates a color from a red-blue colormap based on the slope value.
    Negative slopes are blue, positive are red.
    """
    # Normalize the slope to the range [0, 1] for the colormap
    norm = matplotlib.colors.Normalize(vmin=min_slope, vmax=max_slope)

    # Use a diverging colormap
    cmap = cm.get_cmap('Reds')

    rgba_color = cmap(norm(slope))
    return colors.rgb2hex(rgba_color)

def _format_optional_float(value: Any, precision: int = 4) -> str:
    """
    Formats numeric values with a fixed precision, or returns 'N/A'.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return "N/A"

def create_station_map(
    station_data: List[Dict[str, Any]],
    output_path: Path,
    center_lat: float = 37.0,
    center_lon: float = -95.0,
    zoom_start: int = 4
):
    """
    Creates an interactive Folium map of all provided climate stations.

    Args:
        station_data: A list of dictionaries, where each dict represents a station
                      and contains its metadata, stats, and plot paths.
        output_path: The file path to save the HTML map to.
        center_lat: The initial latitude for the map's center.
        center_lon: The initial longitude for the map's center.
        zoom_start: The initial zoom level of the map.
    """
    if not station_data:
        print("No station data provided. Cannot create map.")
        return

    # Create a base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles="CartoDB positron")

    # Find the min and max slopes for color normalization
    slopes = [s['slope'] for s in station_data if 'slope' in s and pd.notna(s['slope'])]
    min_s, max_s = min(slopes), max(slopes)

    # Add each station as a marker to the map
    for station in station_data:
        # Get marker color and shape
        color = _get_color_from_slope(station.get('slope', 0), min_s, max_s)
        shape = _get_marker_shape(station.get('StationID', ''))

        # Create the HTML content for the popup
        popup_html = (
            f"<b>{station.get('StationName', 'N/A')} ({station.get('StationID', 'N/A')})</b><br>"
            f"<b>Elevation:</b> {_format_optional_float(station.get('Elevation'), precision=1)} m<br>"
            f"<b>Observed Slope:</b> {_format_optional_float(station.get('slope'))} °C/°C<br>"
            f"<b>Model Slope:</b> {_format_optional_float(station.get('model_slope'))}<br>"
            f"<b>R²:</b> {_format_optional_float(station.get('r_squared'))}<br><hr>"
            "<b>Plots:</b><br>"
            f"<a href='{station.get('regression_plot', '#')}' target='_blank'>Regression</a> | "
            f"<a href='{station.get('residual_plot', '#')}' target='_blank'>Residual</a> | "
            f"<a href='{station.get('timeseries_plot', '#')}' target='_blank'>Time-series</a>"
        )
        popup = folium.Popup(popup_html, max_width=300)

        # Add the appropriate marker type to the map
        if shape == "triangle":
            folium.Marker(
                location=[station['Latitude'], station['Longitude']],
                icon=folium.Icon(icon="arrow-up", color='gray'), # A triangle-like icon
                popup=popup,
            ).add_to(m)
        else: # Circle
            folium.CircleMarker(
                location=[station['Latitude'], station['Longitude']],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=popup,
            ).add_to(m)

    # Save the map to the specified HTML file
    m.save(str(output_path))
    print(f"Map successfully saved to {output_path}")
