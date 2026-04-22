"""
Handles the creation of interactive Folium maps.
"""

import pandas as pd
import folium
import plotly.graph_objects as go
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Fixed scale so all yellow-red slope maps are directly comparable.
SHARED_SLOPE_MIN = -1.0
SHARED_SLOPE_MAX = 3.1


def get_shared_slope_scale() -> Tuple[float, float]:
    """Return the fixed min/max range for slope colormaps."""
    return SHARED_SLOPE_MIN, SHARED_SLOPE_MAX


def _derive_map_output_path(output_path: Path) -> Path:
    """Route flat map outputs to a maps subfolder."""
    if output_path.parent.name == "maps":
        return output_path
    return output_path.parent / "maps" / output_path.name


def _derive_globe_output_path(output_path: Path) -> Path:
    """Return companion globe path in a sibling globe folder."""
    map_path = _derive_map_output_path(output_path)
    return map_path.parent.parent / "globe" / f"{map_path.stem}_globe{map_path.suffix}"

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

    # Use one fixed slope scale so all maps are directly comparable.
    min_s, max_s = get_shared_slope_scale()

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

    # Save flat map under maps/ and globe under globe/.
    map_output = _derive_map_output_path(output_path)
    map_output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(map_output))
    print(f"Map successfully saved to {map_output}")

    globe_output = _derive_globe_output_path(map_output)
    create_station_globe_map(station_data=station_data, output_path=globe_output)
    print(f"Globe map successfully saved to {globe_output}")


def create_station_globe_map(
    station_data: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Create an orthographic globe map with station markers."""
    if not station_data:
        return

    df = pd.DataFrame(station_data)

    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        return

    min_s, max_s = get_shared_slope_scale()

    fig = go.Figure(
        go.Scattergeo(
            lon=df["Longitude"],
            lat=df["Latitude"],
            mode="markers",
            marker={
                "size": 5,
                "color": df.get("slope", pd.Series([0.0] * len(df))),
                "cmin": min_s,
                "cmax": max_s,
                "colorscale": "YlOrRd",
                "colorbar": {"title": "Slope"},
                "opacity": 0.85,
            },
            text=[
                (
                    f"<b>{row.get('StationName', 'N/A')} ({row.get('StationID', 'N/A')})</b><br>"
                    f"<b>Observed Slope:</b> {_format_optional_float(row.get('slope'))}<br>"
                    f"<b>Model Slope:</b> {_format_optional_float(row.get('model_slope'))}<br>"
                    f"<b>R2:</b> {_format_optional_float(row.get('r_squared'))}<br>"
                    f"<b>Elevation:</b> {_format_optional_float(row.get('Elevation'), precision=1)} m"
                )
                for _, row in df.iterrows()
            ],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Station Climate Map Globe",
        geo={
            "projection": {"type": "orthographic"},
            "showland": True,
            "landcolor": "rgb(232,236,224)",
            "showocean": True,
            "oceancolor": "rgb(194,217,236)",
            "showlakes": True,
            "lakecolor": "rgb(194,217,236)",
            "bgcolor": "white",
        },
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
