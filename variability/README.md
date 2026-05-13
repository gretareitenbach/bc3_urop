# Variability analysis

This folder contains a small utility to compute statistics (mean, std, min, max, range)
across different model realizations of pattern-scaling NetCDF files.

Usage example:

```bash
python -m variability.compute_variability \
  --input-dir /d3/pgiani/BC3/ITMDT/MPI-ESM1-2-LR/GRETA_PATTERNS \
  --out-dir output/variability --glob "*PatternScalingCoefficients*_AnnualAverages.nc"
```

Outputs: for each group of files that share the same filename after removing the
realization token (e.g. `_r35i1p1f1`) a NetCDF file will be written containing
variables `mean`, `std`, `min`, `max`, and `range` on the same grid.

Dependencies: `xarray`, `numpy` (and an engine that can read NetCDF such as `netcdf4` or `h5netcdf`).
