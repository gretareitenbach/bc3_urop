"""Create a toggleable world climate map comparing model vs physical data.

This script builds a map with two layers:
- Model layer from output/station_data_grid_5x5.csv
- Physical layer from GFDL-CM4 NetCDF files (tas, hurs, slope)

Both layers use the same 5x5-style grid visualization and can be toggled on/off.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import folium
import numpy as np
import pandas as pd
import xarray as xr
from branca.colormap import linear


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def normalize_longitude(longitude: pd.Series) -> pd.Series:
    """Normalize longitudes into the [-180, 180) range."""
    return ((longitude + 180.0) % 360.0) - 180.0


def add_grid_bins(df: pd.DataFrame, grid_size: int = 5) -> pd.DataFrame:
    """Assign rows to latitude/longitude bins."""
    data = df.copy()
    data["Longitude"] = normalize_longitude(data["Longitude"])

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
    """Create a readable metric label."""
    return name.replace("_", " ").title()


def format_popup_value(value, precision: int = 2):
    """Format popup values consistently, returning N/A for missing values."""
    if pd.isna(value):
        return "N/A"
    return "{0:.{1}f}".format(float(value), precision)


def derive_map_output_path(output_map: Path) -> Path:
    """Route flat map outputs into a maps directory."""
    if output_map.parent.name == "maps":
        return output_map
    return output_map.parent / "maps" / output_map.name


def derive_globe_output_path(output_map: Path, label: str) -> Path:
    """Return a globe output path in a sibling globe folder with label suffix."""
    map_output = derive_map_output_path(output_map)
    return map_output.parent.parent / "globe" / f"{map_output.stem}_{label}_globe{map_output.suffix}"


def _first_time_slice(da: xr.DataArray) -> xr.DataArray:
    """Return first time slice when a time dimension exists."""
    if "time" in da.dims:
        return da.isel(time=0, drop=True)
    return da


def load_physical_grid(
    tas_nc: Path,
    hurs_nc: Path,
    slope_nc: Path,
    grid_size: int,
) -> pd.DataFrame:
    """Load physical model NetCDF data and aggregate it onto the 5x5 grid."""
    with xr.open_dataset(tas_nc) as tas_ds, xr.open_dataset(hurs_nc) as hurs_ds, xr.open_dataset(
        slope_nc
    ) as slope_ds:
        tas = _first_time_slice(tas_ds["tas"])
        hurs = _first_time_slice(hurs_ds["hurs"])
        slope = _first_time_slice(slope_ds["slope"])

        merged = xr.Dataset(
            {
                "climo_temp": tas,
                "climo_humidity": hurs,
                "slope": slope,
            }
        )

        df = merged.to_dataframe().reset_index()

    df = df.rename(columns={"lat": "Latitude", "lon": "Longitude"})
    df = df.dropna(subset=["Latitude", "Longitude", "slope"])
    df["Longitude"] = normalize_longitude(df["Longitude"])

    # Convert near-surface temperature to Celsius for consistency with station-derived outputs.
    df["climo_temp"] = df["climo_temp"] - 273.15

    aggregated = aggregate_grid_means(
        df[["Latitude", "Longitude", "climo_temp", "climo_humidity", "slope"]],
        grid_size=grid_size,
    )
    aggregated["source"] = "physical_model"
    return aggregated


def load_model_grid(model_grid_csv: Path) -> pd.DataFrame:
    """Load existing model grid data produced by regrid.py."""
    model_df = pd.read_csv(model_grid_csv)

    required = {"cell_id", "lat_min", "lat_max", "lon_min", "lon_max", "slope"}
    missing = [column for column in required if column not in model_df.columns]
    if missing:
        raise ValueError(f"Model grid CSV missing required columns: {missing}")

    if "station_count" not in model_df.columns:
        model_df["station_count"] = 0

    model_df["source"] = "model"
    return model_df


def _build_popup_lines(row: pd.Series, layer_key: str) -> Iterable[str]:
    """Create popup content for each cell according to source layer."""
    base_lines = [
        f"<b>Bounds:</b> [{row['lat_min']}, {row['lat_max']}) lat, [{row['lon_min']}, {row['lon_max']}) lon",
        f"<b>Slope:</b> {format_popup_value(row.get('slope'))}",
        f"<b>Climo Temp (C):</b> {format_popup_value(row.get('climo_temp'))}",
        f"<b>Climo Humidity (%):</b> {format_popup_value(row.get('climo_humidity'))}",
    ]

    if layer_key == "model":
        model_lines = [
            f"<b>Station Count:</b> {int(row.get('station_count', 0))}",
            f"<b>Multilinear Slope:</b> {format_popup_value(row.get('multilinear_predicted_slope'))}",
            f"<b>Tree Slope:</b> {format_popup_value(row.get('decision_tree_predicted_slope'))}",
            f"<b>R2:</b> {format_popup_value(row.get('r_squared'))}",
        ]
        return [*base_lines, *model_lines]

    physical_lines = [
        f"<b>Gridpoint Count:</b> {int(row.get('station_count', 0))}",
        "<b>Source:</b> GFDL-CM4 physical model",
    ]
    return [*base_lines, *physical_lines]


def add_grid_layer(
    world_map: folium.Map,
    grid_df: pd.DataFrame,
    layer_name: str,
    color_metric: str,
    colormap,
    show: bool,
    layer_key: str,
) -> None:
    """Add one rectangle layer for a dataset."""
    layer = folium.FeatureGroup(name=layer_name, show=show)

    for _, row in grid_df.iterrows():
        value = float(row[color_metric])
        popup_html = "<br>".join(_build_popup_lines(row, layer_key=layer_key))

        folium.Rectangle(
            bounds=[[row["lat_min"], row["lon_min"]], [row["lat_max"], row["lon_max"]]],
            stroke=False,
            fill=True,
            fill_color=colormap(value),
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=(
                f"[{row['lat_min']}, {row['lat_max']}), [{row['lon_min']}, {row['lon_max']})"
                f" | {format_metric_name(color_metric)}: {value:.3f}"
            ),
        ).add_to(layer)

    layer.add_to(world_map)


def create_toggle_map(
    model_grid_df: pd.DataFrame,
    physical_grid_df: pd.DataFrame,
    output_map: Path,
    color_metric: str,
) -> None:
    """Create one map with layer toggles for model and physical datasets."""
    for name, grid_df in (("model", model_grid_df), ("physical", physical_grid_df)):
        if grid_df.empty:
            raise ValueError(f"{name} grid dataframe is empty.")
        if color_metric not in grid_df.columns:
            raise ValueError(f"{name} grid dataframe does not contain '{color_metric}'.")

    combined = pd.concat(
        [model_grid_df[[color_metric]], physical_grid_df[[color_metric]]],
        ignore_index=True,
    )
    value_min = float(combined[color_metric].min())
    value_max = float(combined[color_metric].max())

    if np.isclose(value_min, value_max):
        colormap = linear.YlOrRd_09.scale(value_min - 1.0, value_max + 1.0)
    else:
        colormap = linear.YlOrRd_09.scale(value_min, value_max)
    colormap.caption = f"Average {format_metric_name(color_metric)}"

    world_map = folium.Map(
        location=[20.0, 0.0],
        zoom_start=2,
        tiles="CartoDB positron",
    )

    add_grid_layer(
        world_map=world_map,
        grid_df=model_grid_df,
        layer_name="Model Data (station/grid)",
        color_metric=color_metric,
        colormap=colormap,
        show=True,
        layer_key="model",
    )
    add_grid_layer(
        world_map=world_map,
        grid_df=physical_grid_df,
        layer_name="Physical Model Data (GFDL-CM4)",
        color_metric=color_metric,
        colormap=colormap,
        show=False,
        layer_key="physical",
    )

    colormap.add_to(world_map)
    folium.LayerControl(collapsed=False).add_to(world_map)

    output_map = derive_map_output_path(output_map)
    output_map.parent.mkdir(parents=True, exist_ok=True)
    world_map.save(str(output_map))

    # Also emit globe maps for each source in the same output folder.
    from mapping_3d import create_globe_map

    model_globe_output = derive_globe_output_path(output_map, label="model")
    physical_globe_output = derive_globe_output_path(output_map, label="physical")
    model_globe_output.parent.mkdir(parents=True, exist_ok=True)
    physical_globe_output.parent.mkdir(parents=True, exist_ok=True)

    create_globe_map(
        grid_df=model_grid_df,
        output_html=model_globe_output,
        color_metric=color_metric,
    )
    create_globe_map(
        grid_df=physical_grid_df,
        output_html=physical_globe_output,
        color_metric=color_metric,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a toggleable world climate map from model and physical datasets."
    )
    parser.add_argument(
        "--model-grid-csv",
        type=Path,
        default=Path("output/station_data_grid_5x5.csv"),
        help="Existing model grid CSV (generated by regrid.py).",
    )
    parser.add_argument(
        "--tas-nc",
        type=Path,
        default=Path("data/piControl_Climatology_tas_GFDL-CM4_MAVG_r180x90.nc"),
        help="Physical model climatological tas NetCDF path.",
    )
    parser.add_argument(
        "--hurs-nc",
        type=Path,
        default=Path("data/piControl_Climatology_hurs_GFDL-CM4_MAVG_Amon_r180x90_AnnualAverages.nc"),
        help="Physical model climatological hurs NetCDF path.",
    )
    parser.add_argument(
        "--slope-nc",
        type=Path,
        default=Path("data/PatternScalingCoefficients_tas_ssp245-ssp585_MAVG_r180x90_AnnualAverages.nc"),
        help="Physical model slope NetCDF path.",
    )
    parser.add_argument(
        "--physical-grid-csv",
        type=Path,
        default=Path("output/physical_model_grid_5x5.csv"),
        help="Output CSV path for aggregated physical model 5x5 grid.",
    )
    parser.add_argument(
        "--output-map",
        type=Path,
        default=Path("output/maps/world_climate_map_grid_5x5_toggle.html"),
        help="Output HTML path for toggleable map.",
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
        help="Numeric metric used to color map cells.",
    )
    return parser.parse_args()


def main() -> None:
    """Build physical grid output and combined toggle map."""
    args = parse_args()

    model_grid_df = load_model_grid(args.model_grid_csv)
    physical_grid_df = load_physical_grid(
        tas_nc=args.tas_nc,
        hurs_nc=args.hurs_nc,
        slope_nc=args.slope_nc,
        grid_size=args.grid_size,
    )

    args.physical_grid_csv.parent.mkdir(parents=True, exist_ok=True)
    physical_grid_df.to_csv(args.physical_grid_csv, index=False)

    args.output_map = derive_map_output_path(args.output_map)

    create_toggle_map(
        model_grid_df=model_grid_df,
        physical_grid_df=physical_grid_df,
        output_map=args.output_map,
        color_metric=args.color_metric,
    )

    print(f"Model grid CSV: {args.model_grid_csv}")
    print(f"Physical grid CSV: {args.physical_grid_csv}")
    print(f"Toggle map: {args.output_map}")
    print(f"Model globe map: {derive_globe_output_path(args.output_map, label='model')}")
    print(f"Physical globe map: {derive_globe_output_path(args.output_map, label='physical')}")


if __name__ == "__main__":
    main()
