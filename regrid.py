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
import warnings
from pathlib import Path
from typing import Optional, Tuple

import folium
import numpy as np
import pandas as pd
from branca.colormap import linear

from climate_analysis.mapping import get_shared_slope_scale


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

    if "slope" in binned.columns:
        slope_stats = (
            binned.groupby(["lat_bin", "lon_bin"])["slope"]
            .agg([
                ("slope_min", "min"),
                ("slope_max", "max"),
                ("slope_std", lambda values: values.std(ddof=0)),
            ])
            .reset_index()
        )
        aggregated = aggregated.merge(slope_stats, on=["lat_bin", "lon_bin"], how="left")

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


def format_popup_value(value, precision=2):
    """Format popup values consistently, returning N/A for missing values."""
    if pd.isna(value):
        return "N/A"
    return "{0:.{1}f}".format(float(value), precision)


def _format_tooltip_value(value, precision=2):
    """Format values for compact tooltip display."""
    if pd.isna(value):
        return "N/A"
    return "{0:.{1}f}".format(float(value), precision)


def _derive_globe_output_path(output_map: Path) -> Path:
    """Return companion globe output path in a sibling globe directory."""
    if output_map.parent.name == "maps":
        globe_dir = output_map.parent.parent / "globe"
    else:
        globe_dir = output_map.parent / "globe"
    return globe_dir / f"{output_map.stem}_globe{output_map.suffix}"


def _derive_map_output_path(output_map: Path) -> Path:
    """Route flat map outputs into a maps directory."""
    if output_map.parent.name == "maps":
        return output_map
    return output_map.parent / "maps" / output_map.name


def _load_model_predictions(predictions_csv: Path, column_name: str) -> Optional[pd.DataFrame]:
    """Load a model prediction CSV and return cell_id + renamed predicted slope."""
    if not predictions_csv.exists():
        warnings.warn(f"Model predictions file not found: {predictions_csv}")
        return None

    predictions_df = pd.read_csv(predictions_csv)
    required_cols = {"cell_id", "Predicted_Slope"}
    missing = [c for c in required_cols if c not in predictions_df.columns]
    if missing:
        warnings.warn(
            f"Model predictions file is missing required columns {missing}: {predictions_csv}"
        )
        return None

    model_df = predictions_df[["cell_id", "Predicted_Slope"]].drop_duplicates(subset=["cell_id"])
    return model_df.rename(columns={"Predicted_Slope": column_name})


def add_model_slopes_to_grid(
    grid_df: pd.DataFrame,
    multilinear_predictions_csv: Path,
    decision_tree_predictions_csv: Path,
) -> pd.DataFrame:
    """Attach multilinear and decision-tree predicted slopes to each grid cell by cell_id."""
    enriched = grid_df.copy()

    multilinear_df = _load_model_predictions(
        multilinear_predictions_csv,
        column_name="multilinear_predicted_slope",
    )
    if multilinear_df is not None:
        enriched = enriched.merge(multilinear_df, on="cell_id", how="left")

    decision_tree_df = _load_model_predictions(
        decision_tree_predictions_csv,
        column_name="decision_tree_predicted_slope",
    )
    if decision_tree_df is not None:
        enriched = enriched.merge(decision_tree_df, on="cell_id", how="left")

    return enriched


def add_station_count_slider(
    world_map: folium.Map,
    layer_thresholds: list,
    min_count: int,
    max_count: int,
) -> None:
    """Add a slider control that filters cells by minimum station count."""
    map_name = world_map.get_name()
    slider_id = "station-count-slider-{0}".format(map_name)
    value_id = "station-count-value-{0}".format(map_name)
    control_id = "station-count-filter-{0}".format(map_name)

    slider_html = """
    <div id="{control_id}" style="position: fixed; bottom: 40px; left: 40px;
         z-index: 9999; background-color: rgba(255, 255, 255, 0.95);
         padding: 12px 14px; border: 1px solid #bdbdbd; border-radius: 6px;
         box-shadow: 0 1px 6px rgba(0, 0, 0, 0.18); min-width: 250px;">
        <div style="font-size: 13px; font-weight: 600; margin-bottom: 8px;">
            Minimum stations per cell: <span id="{value_id}">{min_count}</span>
        </div>
        <input id="{slider_id}" type="range" min="{min_count}" max="{max_count}"
               step="1" value="{min_count}" style="width: 100%;">
        <div style="display: flex; justify-content: space-between; font-size: 11px; margin-top: 6px; color: #555;">
            <span>{min_count}</span>
            <span>{max_count}</span>
        </div>
    </div>
    """.format(
        control_id=control_id,
        slider_id=slider_id,
        value_id=value_id,
        min_count=min_count,
        max_count=max_count,
    )
    world_map.get_root().html.add_child(folium.Element(slider_html))

    layer_config_js = ",\n".join(layer_thresholds)
    slider_script = """
    setTimeout(function() {{
        var mapRef = window["{map_name}"];
        var slider = document.getElementById("{slider_id}");
        var valueLabel = document.getElementById("{value_id}");
        var control = document.getElementById("{control_id}");
        var layerConfigs = [
            {layer_config_js}
        ];

        function applyStationCountFilter() {{
            var threshold = parseInt(slider.value, 10);
            valueLabel.textContent = threshold;

            for (var i = 0; i < layerConfigs.length; i++) {{
                var layerConfig = layerConfigs[i];
                if (layerConfig.stationCount >= threshold) {{
                    if (!mapRef.hasLayer(layerConfig.layer)) {{
                        layerConfig.layer.addTo(mapRef);
                    }}
                }} else if (mapRef.hasLayer(layerConfig.layer)) {{
                    mapRef.removeLayer(layerConfig.layer);
                }}
            }}
        }}

        if (control && window.L && window.L.DomEvent) {{
            L.DomEvent.disableClickPropagation(control);
            L.DomEvent.disableScrollPropagation(control);
        }}

        if (!mapRef || !slider || !valueLabel) {{
            return;
        }}

        slider.addEventListener("input", applyStationCountFilter);
        applyStationCountFilter();
    }}, 0);
    """.format(
        map_name=map_name,
        slider_id=slider_id,
        value_id=value_id,
        control_id=control_id,
        layer_config_js=layer_config_js,
    )
    world_map.get_root().script.add_child(folium.Element(slider_script))


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

    if color_metric == "slope":
        value_min, value_max = get_shared_slope_scale()

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

    layer_thresholds = []

    for _, row in grid_df.iterrows():
        metric_value = float(row[color_metric])
        fill_color = colormap(metric_value)

        slope_range = "N/A"
        if not pd.isna(row.get("slope_min")) and not pd.isna(row.get("slope_max")):
            slope_range = "({0} to {1})".format(
                format_popup_value(row.get("slope_min")),
                format_popup_value(row.get("slope_max")),
            )

        slope_display = format_popup_value(row.get("slope"))
        if slope_range != "N/A":
            slope_display = "{0} {1}".format(slope_display, slope_range)

        popup_lines = [
            f"<b>Bounds:</b> [{row['lat_min']}, {row['lat_max']}) lat, [{row['lon_min']}, {row['lon_max']}) lon",
            f"<b>Station Count:</b> {int(row['station_count'])}",
            f"<b>Slope:</b> {slope_display}",
            f"<b>Multilinear Slope:</b> {format_popup_value(row.get('multilinear_predicted_slope'))}",
            f"<b>Tree Slope:</b> {format_popup_value(row.get('decision_tree_predicted_slope'))}",
            f"<b>Standard Deviation:</b> {format_popup_value(row.get('slope_std'))}",
            f"<b>Intercept:</b> {format_popup_value(row.get('intercept'))}",
            f"<b>R2:</b> {format_popup_value(row.get('r_squared'))}",
            f"<b>Climo Temp:</b> {format_popup_value(row.get('climo_temp'))}",
            f"<b>Climo Humidity:</b> {format_popup_value(row.get('climo_humidity'))}",
            f"<b>Elevation:</b> {format_popup_value(row.get('Elevation'))}",
        ]

        popup_html = "<br>".join(popup_lines)

        bounds = [
            [row["lat_min"], row["lon_min"]],
            [row["lat_max"], row["lon_max"]],
        ]

        rectangle = folium.Rectangle(
            bounds=bounds,
            stroke=False,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=(
                f"[{row['lat_min']}, {row['lat_max']}), [{row['lon_min']}, {row['lon_max']}) | {format_metric_name(color_metric)}: "
                f"{metric_value:.2f} | Multi: {_format_tooltip_value(row.get('multilinear_predicted_slope'))} "
                f"| Tree: {_format_tooltip_value(row.get('decision_tree_predicted_slope'))} "
                f"| Stations: {int(row['station_count'])}"
            ),
        )
        rectangle.add_to(world_map)
        layer_thresholds.append(
            "{{layer: {layer_name}, stationCount: {station_count}}}".format(
                layer_name=rectangle.get_name(),
                station_count=int(row["station_count"]),
            )
        )

    colormap.add_to(world_map)
    add_station_count_slider(
        world_map,
        layer_thresholds=layer_thresholds,
        min_count=int(grid_df["station_count"].min()),
        max_count=int(grid_df["station_count"].max()),
    )

    output_map = _derive_map_output_path(output_map)
    output_map.parent.mkdir(parents=True, exist_ok=True)
    world_map.save(str(output_map))

    # Emit a matching globe HTML next to the flat map.
    from mapping_3d import create_globe_map

    globe_output_map = _derive_globe_output_path(output_map)
    globe_output_map.parent.mkdir(parents=True, exist_ok=True)
    create_globe_map(
        grid_df=grid_df,
        output_html=globe_output_map,
        color_metric=color_metric,
    )


def run_regridding(
    input_csv: Path,
    output_csv: Path,
    output_map: Path,
    grid_size: int,
    color_metric: str,
    multilinear_predictions_csv: Path,
    decision_tree_predictions_csv: Path,
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
    grid_df = add_model_slopes_to_grid(
        grid_df,
        multilinear_predictions_csv=multilinear_predictions_csv,
        decision_tree_predictions_csv=decision_tree_predictions_csv,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(output_csv, index=False)

    output_map = _derive_map_output_path(output_map)
    create_filled_grid_map(grid_df, output_map=output_map, color_metric=color_metric)
    globe_output_map = _derive_globe_output_path(output_map)

    print(f"Loaded stations: {len(df)}")
    print(f"Occupied {grid_size}x{grid_size} cells: {len(grid_df)}")
    print(f"Aggregated grid CSV: {output_csv}")
    print(f"Filled grid map: {output_map}")
    print(f"Globe grid map: {globe_output_map}")


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
        default=Path("output/maps/world_climate_map_grid_5x5.html"),
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
    parser.add_argument(
        "--multilinear-predictions-csv",
        type=Path,
        default=Path("multilinear/notebook_outputs/multilinear/predictions.csv"),
        help="Path to multilinear notebook predictions.csv (cell_id + Predicted_Slope).",
    )
    parser.add_argument(
        "--decision-tree-predictions-csv",
        type=Path,
        default=Path("multilinear/notebook_outputs/decision_tree/predictions.csv"),
        help="Path to decision-tree notebook predictions.csv (cell_id + Predicted_Slope).",
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
        multilinear_predictions_csv=args.multilinear_predictions_csv,
        decision_tree_predictions_csv=args.decision_tree_predictions_csv,
    )
