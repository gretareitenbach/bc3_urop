"""Create interactive folium maps for variability outputs (mean/std).

This script reads `*_variability.nc` files created by
`compute_variability.py` and generates interactive HTML maps with rectangles
for each grid cell colored by the statistic value.

Usage:
    python -m variability.map_variability --in-dir output/variability --out-dir output/variability/maps
"""

import argparse
from pathlib import Path
import xarray as xr
import folium
from branca.colormap import linear
from climate_analysis.mapping import get_shared_slope_scale
import numpy as np


def infer_lat_lon_edges(ds, lat_name=None, lon_name=None):
    # Try common coordinate names
    lat_keys = [lat_name] if lat_name else [k for k in ds.coords if k.lower().startswith("lat")]
    lon_keys = [lon_name] if lon_name else [k for k in ds.coords if k.lower().startswith("lon")]
    if not lat_keys or not lon_keys:
        raise ValueError("Could not find lat/lon coordinates in dataset")
    lat = ds.coords[lat_keys[0]]
    lon = ds.coords[lon_keys[0]]

    # If lat/lon are 1D arrays of cell centers, compute edges
    if lat.ndim == 1 and lon.ndim == 1:
        lat_vals = np.asarray(lat)
        lon_vals = np.asarray(lon)
        # edges: midpoint between values; extend ends
        lat_edges = np.concatenate(([lat_vals[0] - (lat_vals[1] - lat_vals[0]) / 2],
                                    (lat_vals[:-1] + lat_vals[1:]) / 2,
                                    [lat_vals[-1] + (lat_vals[-1] - lat_vals[-2]) / 2]))
        lon_edges = np.concatenate(([lon_vals[0] - (lon_vals[1] - lon_vals[0]) / 2],
                                    (lon_vals[:-1] + lon_vals[1:]) / 2,
                                    [lon_vals[-1] + (lon_vals[-1] - lon_vals[-2]) / 2]))
        return lat_edges, lon_edges

    # If lat/lon are 2D, try to compute bounds by looking at adjacent coordinates
    if lat.ndim == 2 and lon.ndim == 2:
        # assume regular grid and use first row/col
        lat_vals = np.asarray(lat[:, 0])
        lon_vals = np.asarray(lon[0, :])
        return infer_lat_lon_edges(xr.Dataset(coords={"lat": lat_vals, "lon": lon_vals}))

    raise ValueError("Unsupported lat/lon coordinate shapes")


def map_from_da(da, out_path: Path, title: str = None, ds_all: xr.Dataset = None):
    """Map a 2D DataArray `da` with optional `ds_all` containing mean/std/min/max for popups.

    If `ds_all` is provided, the popup for each cell will include the average (`mean`),
    standard deviation (`std`), minimum (`min`) and maximum (`max`) values for that cell.
    """
    ds = da.to_dataset(name=da.name)
    # get lat/lon
    lat_name = next((k for k in ds.coords if k.lower().startswith("lat")), None)
    lon_name = next((k for k in ds.coords if k.lower().startswith("lon")), None)
    lat = ds.coords[lat_name]
    lon = ds.coords[lon_name]

    lat_edges, lon_edges = infer_lat_lon_edges(ds, lat_name=lat_name, lon_name=lon_name)

    # determine colormap scale
    if ds_all is not None and da.name == "mean":
        # use shared slope scale for mean maps so they match other repo visuals
        try:
            vmin, vmax = get_shared_slope_scale()
        except Exception:
            vmin = float(np.nanmin(da.values))
            vmax = float(np.nanmax(da.values))
    elif ds_all is not None and da.name == "std":
        # stds are non-negative; fix lower bound at 0 and use dataset max
        vmin = 0.0
        try:
            vmax = float(np.nanmax(ds_all["std"].values))
        except Exception:
            vmax = float(np.nanmax(da.values))
    else:
        vmin = float(np.nanmin(da.values))
        vmax = float(np.nanmax(da.values))

    if np.isclose(vmin, vmax):
        vmin -= 1e-6
        vmax += 1e-6

    colormap = linear.YlOrRd_09.scale(vmin, vmax)
    colormap.caption = title or da.name

    m = folium.Map(location=[0, 0], zoom_start=2, tiles="CartoDB positron")

    # prepare stats arrays if ds_all provided
    stats = {}
    if ds_all is not None:
        for sname in ("mean", "std", "min", "max"):
            if sname in ds_all:
                stats[sname] = np.asarray(ds_all[sname].squeeze())

    # If da is 2D with dims (lat, lon)
    arr = np.asarray(da)
    nlat, nlon = arr.shape

    for i in range(nlat):
        for j in range(nlon):
            val = arr[i, j]
            if np.isnan(val):
                continue
            lat_min = lat_edges[i]
            lat_max = lat_edges[i + 1]
            lon_min = lon_edges[j]
            lon_max = lon_edges[j + 1]
            color = colormap(val)
            bounds = [[float(lat_min), float(lon_min)], [float(lat_max), float(lon_max)]]

            # build popup: prefer values from stats (mean/std/min/max) if available
            mean_v = stats.get("mean", None)
            std_v = stats.get("std", None)
            min_v = stats.get("min", None)
            max_v = stats.get("max", None)

            def fmt(arr, ii, jj):
                try:
                    return f"{float(arr[ii, jj]):.6f}"
                except Exception:
                    return "N/A"

            popup_lines = []
            if mean_v is not None:
                popup_lines.append(f"<b>Mean:</b> {fmt(mean_v, i, j)}")
            if std_v is not None:
                popup_lines.append(f"<b>Std:</b> {fmt(std_v, i, j)}")
            if max_v is not None and min_v is not None:
                popup_lines.append(f"<b>Min:</b> {fmt(min_v, i, j)}")
                popup_lines.append(f"<b>Max:</b> {fmt(max_v, i, j)}")

            popup_html = "<br>".join(popup_lines)
            popup = folium.Popup(popup_html, max_width=320)
            folium.Rectangle(bounds=bounds, stroke=False, fill=True, fill_color=color, fill_opacity=0.8, popup=popup).add_to(m)

    colormap.add_to(m)
    m.save(out_path)


def main(args):
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*_variability.nc"))
    if not files:
        print(f"No variability NetCDF files found in {in_dir}")
        return

    for f in files:
        ds = xr.open_dataset(f)
        base = f.stem.replace("_variability", "")
        # Expect data variables like 'mean' and 'std'
        for var in ["mean", "std"]:
            if var in ds:
                da = ds[var]
                # if dataset has extra dims, attempt to reduce to 2D (lat, lon)
                if "realization" in da.dims:
                    da = da.isel(realization=0)
                # reduce any time or other singleton dims
                da2 = da.squeeze()
                # ensure 2D ordering lat x lon
                if da2.ndim != 2:
                    print(f"Skipping variable {var} in {f} — shape {da2.shape} not 2D")
                    continue
                out_path = out_dir / f"{base}_{var}.html"
                print(f"Creating map for {base} {var} -> {out_path}")
                # pass the full dataset so popups can include mean/std/min/max
                map_from_da(da2, out_path, title=f"{base} — {var}", ds_all=ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    main(args)
