"""Compute variability statistics across realizations.

Usage (example):
    python -m variability.compute_variability \
        --input-dir /d3/pgiani/BC3/ITMDT/MPI-ESM1-2-LR/GRETA_PATTERNS \
        --out-dir output/variability --glob "*PatternScalingCoefficients*_AnnualAverages.nc"

The script groups files by their base name (filename with realization removed),
loads the primary data variable in each file with xarray, stacks realizations
along a new `realization` axis, and writes per-grid statistics: mean, std,
min, max, range to NetCDF files under `out_dir/<group>_variability.nc`.
"""

import argparse
import logging
from pathlib import Path
from collections import defaultdict

import xarray as xr
import numpy as np

from .utils import find_nc_files, extract_realization, base_name_without_realization


logger = logging.getLogger("variability")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def pick_data_variable(ds: xr.Dataset) -> str:
    """Choose the first non-coordinate data variable from a dataset."""
    # prefer 'pattern' or 'tas' if present
    for prefer in ("pattern", "tas", "coeff", "coefficients"):
        if prefer in ds.data_vars:
            return prefer
    # otherwise pick first
    if ds.data_vars:
        return list(ds.data_vars)[0]
    raise ValueError("No data variables found in dataset")


def group_files(files):
    groups = defaultdict(list)
    for f in files:
        base = base_name_without_realization(f)
        groups[base].append(f)
    return groups


def load_and_stack(file_list):
    datasets = []
    realization_codes = []
    for f in file_list:
        ds = xr.open_dataset(f)
        var = pick_data_variable(ds)
        data = ds[var]
        datasets.append(data.expand_dims({"realization": [extract_realization(f.name)]}))
        realization_codes.append(extract_realization(f.name))
    # concat along realization
    stacked = xr.concat(datasets, dim="realization")
    return stacked


def compute_stats(stacked):
    mean = stacked.mean(dim="realization", skipna=True)
    std = stacked.std(dim="realization", skipna=True)
    mn = stacked.min(dim="realization", skipna=True)
    mx = stacked.max(dim="realization", skipna=True)
    rng = mx - mn
    return {"mean": mean, "std": std, "min": mn, "max": mx, "range": rng}


def write_output(stats_dict, out_path: Path, attrs=None):
    ds_out = xr.Dataset()
    for name, da in stats_dict.items():
        ds_out[name] = da
    if attrs:
        ds_out.attrs.update(attrs)
    ds_out.to_netcdf(out_path)


def main(args):
    input_dir = args.input_dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_nc_files(input_dir, args.glob)
    logger.info(f"Found {len(files)} files matching pattern in {input_dir}")

    groups = group_files(files)
    logger.info(f"Identified {len(groups)} groups (pattern sets) by base filename")

    for base, flist in groups.items():
        if len(flist) < 2:
            logger.info(f"Skipping {base} — only {len(flist)} realization(s)")
            continue
        logger.info(f"Processing group {base} with {len(flist)} realizations")
        try:
            stacked = load_and_stack(flist)
            stats = compute_stats(stacked)
            safe_base = base.replace(".nc", "").replace("/", "_")
            out_path = out_dir / f"{safe_base}_variability.nc"
            write_output(stats, out_path, attrs={"source_group": base, "n_realizations": len(flist)})
            logger.info(f"Wrote variability NetCDF: {out_path}")
        except Exception as e:
            logger.error(f"Failed processing {base}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute variability across realizations")
    parser.add_argument("--input-dir", required=True, help="Directory with NetCDF pattern files")
    parser.add_argument("--out-dir", required=True, help="Output directory for variability results")
    parser.add_argument("--glob", default="*.nc", help="Glob pattern to match files (rglob) — default *.nc")
    args = parser.parse_args()
    main(args)
