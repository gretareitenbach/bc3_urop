"""
Regrid station-level climate data to a 5x5 latitude/longitude grid.

This script:
- Loads station data from output/station_data.csv
- Aggregates stations into 5x5 degree grid cells
- Computes mean values for numeric metrics in occupied cell
- Creates an interactive filled-grid map with per-cell metric averages in popups
- Saves aggregated grid data to CSV
"""

import argparse
from pathlib import Path
from typing import Tuple

import folium
import numpy as np
import pandas as pd
from branca.colormap import linear


def normalize_longitude(longitude: pd.Series) -> pd.Series:
    """Normalize longitudes into the [-180, 180) range."""
    return ((longitude + 180.0) % 360.0) - 180.0


def add_grid_bins(df: pd.DataFrame, grid_size: int = 5) -> pd.DataFrame:
    """Assign each station to a 5x5 degree latitude/longitude grid cell."""
    data = df.copy()

    data["Longitude"] = normalize_longitude(data["Longitude"])

    # Handle edge values so 90 and 180 are assigned to the final valid bin.
    lat_values = data["Latitude"].clip(-90.0, 90.0)
    lon_values = data["Longitude"].clip(-180.0, 180.0)

    lat_values = np.where(lat_values == 90.0, 90.0 - 1e-9, lat_values)
    lon_values = np.where(lon_values == 180.0, 180.0 - 1e-9, lon_values)

    data["lat_bin"] = (np.floor(lat_values / grid_size) * grid_size).astype(int)
    data["lon_bin"] = (np.floor(lon_values / grid_size) * grid_size).astype(int)

    valid = (
        data["lat_bin"].between(-90, 90 - grid_size)
        & data["lon_bin"].between(-180, 180 - grid_size)
    )
    return data.loc[valid].copy()


def aggregate_grid_means(df: pd.DataFrame, grid_size: int = 5) -> pd.DataFrame:
    """Compute per-cell means for all numeric metrics."""
    binned = add_grid_bins(df, grid_size=grid_size)

    numeric_columns = binned.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [c for c in numeric_columns if c not in {"lat_bin", "lon_bin"}]

    if not numeric_columns:
        raise ValueError("No numeric columns found to aggregate.")

    grouped = binned.groupby(["lat_bin", "lon_bin"], as_index=False)
    aggregated = grouped[numeric_columns].mean()

    counts = binned.groupby(["lat_bin", "lon_bin"]).size().reset_index()
    counts.columns = ["lat_bin", "lon_bin", "station_count"]
    aggregated = aggregated.merge(counts, on=["lat_bin", "lon_bin"], how="left")

    aggregated["lat_min"] = aggregated["lat_bin"]
    aggregated["lat_max"] = aggregated["lat_bin"] + grid_size
    aggregated["lon_min"] = aggregated["lon_bin"]
    aggregated["lon_max"] = aggregated["lon_bin"] + grid_size
    aggregated["cell_id"] = (
        aggregated["lat_bin"].astype(str)
        + "_"
        + aggregated["lon_bin"].astype(str)
    )

    column_order = [
        "cell_id",
        "lat_bin",
        "lon_bin",
        "lat_min",
        "lat_max",
        "lon_min",
        "lon_max",
        "station_count",
    ]
    metric_columns = [c for c in aggregated.columns if c not in column_order]
    return aggregated[column_order + metric_columns]


def format_metric_name(name: str) -> str:
    """Create a readable label for popup display."""
    return name.replace("_", " ").title()


def create_filled_grid_map(
    grid_df: pd.DataFrame,
    output_map: Path,
    color_metric: str,
    map_center: Tuple[float, float] = (20.0, 0.0),
    zoom_start: int = 2,
) -> None:
    """Render a filled-cell world map for aggregated 5x5 grid data."""
    if grid_df.empty:
        raise ValueError("Grid dataframe is empty. Nothing to map.")

    if color_metric not in grid_df.columns:
        numeric_candidates = [
            c
            for c in grid_df.select_dtypes(include=[np.number]).columns
            if c not in {"lat_bin", "lon_bin", "lat_min", "lat_max", "lon_min", "lon_max", "station_count"}
        ]
        if not numeric_candidates:
            raise ValueError("No numeric metric available for map coloring.")
        color_metric = numeric_candidates[0]

    value_min = float(grid_df[color_metric].min())
    value_max = float(grid_df[color_metric].max())

    if np.isclose(value_min, value_max):
        colormap = linear.YlOrRd_09.scale(value_min - 1.0, value_max + 1.0)
    else:
        colormap = linear.YlOrRd_09.scale(value_min, value_max)

    colormap.caption = f"Average {format_metric_name(color_metric)}"

    world_map = folium.Map(
        location=[map_center[0], map_center[1]],
        zoom_start=zoom_start,
        tiles="CartoDB positron",
    )

    popup_metric_columns = [
        c
        for c in grid_df.select_dtypes(include=[np.number]).columns
        if c not in {"lat_bin", "lon_bin", "lat_min", "lat_max", "lon_min", "lon_max"}
    ]

    for _, row in grid_df.iterrows():
        metric_value = float(row[color_metric])
        fill_color = colormap(metric_value)

        popup_lines = [
            f"<b>Grid Cell:</b> {row['cell_id']}",
            f"<b>Bounds:</b> [{row['lat_min']}, {row['lat_max']}) lat, [{row['lon_min']}, {row['lon_max']}) lon",
        ]

        for col in popup_metric_columns:
            val = row[col]
            if pd.isna(val):
                display = "N/A"
            elif col == "station_count":
                display = str(int(val))
            else:
                display = f"{float(val):.4f}"
            popup_lines.append(f"<b>{format_metric_name(col)}:</b> {display}")

        popup_html = "<br>".join(popup_lines)

        bounds = [
            [row["lat_min"], row["lon_min"]],
            [row["lat_max"], row["lon_max"]],
        ]

        folium.Rectangle(
            bounds=bounds,
            stroke=False,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=(
                f"Cell {row['cell_id']} | {format_metric_name(color_metric)}: "
                f"{metric_value:.4f} | Stations: {int(row['station_count'])}"
            ),
        ).add_to(world_map)

    colormap.add_to(world_map)

    output_map.parent.mkdir(parents=True, exist_ok=True)
    world_map.save(str(output_map))


def run_regridding(
    input_csv: Path,
    output_csv: Path,
    output_map: Path,
    grid_size: int,
    color_metric: str,
) -> None:
    """Execute 5x5 regridding and map generation."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required = {"Latitude", "Longitude"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    grid_df = aggregate_grid_means(df, grid_size=grid_size)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(output_csv, index=False)

    create_filled_grid_map(grid_df, output_map=output_map, color_metric=color_metric)

    print(f"Loaded stations: {len(df)}")
    print(f"Occupied {grid_size}x{grid_size} cells: {len(grid_df)}")
    print(f"Aggregated grid CSV: {output_csv}")
    print(f"Filled grid map: {output_map}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Regrid station_data.csv into a 5x5 lat/lon mean grid and create a filled map."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("output/station_data.csv"),
        help="Input station CSV path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("output/station_data_grid_5x5.csv"),
        help="Output CSV path for aggregated 5x5 grid means.",
    )
    parser.add_argument(
        "--output-map",
        type=Path,
        default=Path("output/world_climate_map_grid_5x5.html"),
        help="Output HTML path for the filled 5x5 grid map.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=5,
        help="Grid size in degrees for both latitude and longitude.",
    )
    parser.add_argument(
        "--color-metric",
        type=str,
        default="slope",
        help="Numeric aggregated metric used to color grid cells.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_regridding(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        output_map=args.output_map,
        grid_size=args.grid_size,
        color_metric=args.color_metric,
    )
