"""
Create an interactive 3D globe HTML for the 5x5 aggregated climate grid.

This script reads the output from regrid.py (output/station_data_grid_5x5.csv)
and renders each occupied grid cell as a colored polygon on an orthographic
globe using Plotly.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

try:
    from global_land_mask import globe as land_globe
except ImportError:
    land_globe = None

from climate_analysis.mapping import get_shared_slope_scale


def format_metric_name(name: str) -> str:
    """Create a readable label for hover display."""
    return name.replace("_", " ").title()


def format_hover_value(value, precision: int = 2) -> str:
    """Format hover values consistently, returning N/A for missing values."""
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{precision}f}"


def latlon_to_xyz(lat_deg: float, lon_deg: float, radius: float = 1.0) -> tuple[float, float, float]:
    """Convert latitude/longitude to Cartesian XYZ coordinates on a sphere."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return float(x), float(y), float(z)


def get_cell_fill_color(value: float, cmin: float, cmax: float, colorscale) -> str:
    """Map a metric value to a concrete RGB color for mesh cell filling."""
    if np.isclose(cmax, cmin):
        t = 0.5
    else:
        t = (value - cmin) / (cmax - cmin)
        t = float(np.clip(t, 0.0, 1.0))

    sampled = sample_colorscale(colorscale, [t])[0]
    if sampled.startswith("rgba("):
        parts = sampled[5:-1].split(",")
        r, g, b = [int(float(p.strip())) for p in parts[:3]]
        return f"rgb({r},{g},{b})"
    return sampled


def create_globe_map(
    grid_df: pd.DataFrame,
    output_html: Path,
    color_metric: str = "slope",
    projection_scale: float = 0.95,
) -> None:
    """Create an interactive orthographic globe map from aggregated grid data."""
    if grid_df.empty:
        raise ValueError("Grid dataframe is empty. Nothing to map.")

    if color_metric not in grid_df.columns:
        numeric_candidates = [
            c
            for c in grid_df.select_dtypes(include=[np.number]).columns
            if c
            not in {
                "lat_bin",
                "lon_bin",
                "lat_min",
                "lat_max",
                "lon_min",
                "lon_max",
                "station_count",
            }
        ]
        if not numeric_candidates:
            raise ValueError("No numeric metric available for globe coloring.")
        color_metric = numeric_candidates[0]

    color_values = grid_df[color_metric].astype(float)
    cmin = float(color_values.min())
    cmax = float(color_values.max())

    if color_metric == "slope":
        cmin, cmax = get_shared_slope_scale()

    if np.isclose(cmin, cmax):
        cmin -= 1.0
        cmax += 1.0

    color_scale = px.colors.sequential.YlOrRd

    hover_texts = []

    for _, row in grid_df.iterrows():
        slope_range = "N/A"
        if not pd.isna(row.get("slope_min")) and not pd.isna(row.get("slope_max")):
            slope_range = (
                f"({format_hover_value(row.get('slope_min'))} to "
                f"{format_hover_value(row.get('slope_max'))})"
            )

        slope_display = format_hover_value(row.get("slope"))
        if slope_range != "N/A":
            slope_display = f"{slope_display} {slope_range}"

        hover_texts.append(
            "<br>".join(
                [
                    f"<b>Bounds:</b> [{int(row['lat_min'])}, {int(row['lat_max'])}) lat, "
                    f"[{int(row['lon_min'])}, {int(row['lon_max'])}) lon",
                    f"<b>Station Count:</b> {int(row['station_count'])}",
                    f"<b>Slope:</b> {slope_display}",
                    f"<b>Multilinear Slope:</b> {format_hover_value(row.get('multilinear_predicted_slope'))}",
                    f"<b>Tree Slope:</b> {format_hover_value(row.get('decision_tree_predicted_slope'))}",
                    f"<b>Standard Deviation:</b> {format_hover_value(row.get('slope_std'))}",
                    f"<b>Intercept:</b> {format_hover_value(row.get('intercept'))}",
                    f"<b>R2:</b> {format_hover_value(row.get('r_squared'))}",
                    f"<b>Climo Temp:</b> {format_hover_value(row.get('climo_temp'))}",
                    f"<b>Climo Humidity:</b> {format_hover_value(row.get('climo_humidity'))}",
                    f"<b>Elevation:</b> {format_hover_value(row.get('Elevation'))}",
                ]
            )
        )

    fig = go.Figure()

    # Base globe sphere for 3D context with land/ocean coloring.
    u = np.linspace(0.0, 2.0 * np.pi, 180)
    v = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 90)
    uu, vv = np.meshgrid(u, v)
    globe_r = 1.0
    globe_x = globe_r * np.cos(vv) * np.cos(uu)
    globe_y = globe_r * np.cos(vv) * np.sin(uu)
    globe_z = globe_r * np.sin(vv)

    lat_grid = np.rad2deg(vv)
    lon_grid = ((np.rad2deg(uu) + 180.0) % 360.0) - 180.0
    if land_globe is not None:
        land_mask = land_globe.is_land(lat_grid, lon_grid)
        surfacecolor = np.where(land_mask, 1.0, 0.0)
    else:
        surfacecolor = np.zeros_like(globe_x)

    fig.add_trace(
        go.Surface(
            x=globe_x,
            y=globe_y,
            z=globe_z,
            surfacecolor=surfacecolor,
            cmin=0.0,
            cmax=1.0,
            colorscale=[[0.0, "rgb(200,220,238)"], [1.0, "rgb(228,232,216)"]],
            showscale=False,
            hoverinfo="skip",
            opacity=1.0,
            lighting={"ambient": 0.8, "diffuse": 0.3, "specular": 0.05, "roughness": 0.9},
        )
    )

    cell_r = 1.012
    for idx, (_, row) in enumerate(grid_df.iterrows()):
        lat_min = float(row["lat_min"])
        lat_max = float(row["lat_max"])
        lon_min = float(row["lon_min"])
        lon_max = float(row["lon_max"])
        metric_value = float(row[color_metric])

        corners = [
            latlon_to_xyz(lat_min, lon_min, radius=cell_r),
            latlon_to_xyz(lat_min, lon_max, radius=cell_r),
            latlon_to_xyz(lat_max, lon_max, radius=cell_r),
            latlon_to_xyz(lat_max, lon_min, radius=cell_r),
        ]
        x = [p[0] for p in corners]
        y = [p[1] for p in corners]
        z = [p[2] for p in corners]

        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color=get_cell_fill_color(metric_value, cmin, cmax, color_scale),
                opacity=0.95,
                flatshading=True,
                text=[hover_texts[idx], hover_texts[idx], hover_texts[idx], hover_texts[idx]],
                hovertemplate="%{text}<extra></extra>",
                hoverinfo="text",
                showscale=False,
            )
        )

    # Hidden marker carries the shared colorbar for all cells.
    fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode="markers",
            marker={
                "size": 0.1,
                "color": [cmin],
                "cmin": cmin,
                "cmax": cmax,
                "colorscale": color_scale,
                "opacity": 0.0,
                "showscale": True,
                "colorbar": {
                    "title": f"Average {format_metric_name(color_metric)}",
                    "x": 1.03,
                    "len": 0.75,
                },
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"World Climate Grid ({format_metric_name(color_metric)})",
        margin={"l": 10, "r": 80, "t": 50, "b": 10},
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "aspectmode": "data",
            "camera": {
                "eye": {"x": projection_scale * 1.6, "y": projection_scale * 1.6, "z": projection_scale * 1.2}
            },
            "bgcolor": "white",
        },
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for globe generation."""
    parser = argparse.ArgumentParser(
        description="Render output/station_data_grid_5x5.csv as an interactive 3D globe HTML."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("output/station_data_grid_5x5.csv"),
        help="Path to aggregated 5x5 grid CSV.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("output/world_climate_map_grid_5x5_globe.html"),
        help="Output HTML path for the interactive globe.",
    )
    parser.add_argument(
        "--color-metric",
        type=str,
        default="slope",
        help="Numeric grid metric used to color cells.",
    )
    parser.add_argument(
        "--projection-scale",
        type=float,
        default=0.95,
        help="Orthographic globe scale (higher values appear larger).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for globe map generation."""
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    grid_df = pd.read_csv(args.input_csv)
    create_globe_map(
        grid_df=grid_df,
        output_html=args.output_html,
        color_metric=args.color_metric,
        projection_scale=args.projection_scale,
    )

    print(f"Loaded occupied grid cells: {len(grid_df)}")
    print(f"Saved globe map HTML: {args.output_html}")


if __name__ == "__main__":
    main()