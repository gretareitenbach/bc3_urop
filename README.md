# BC3 UROP — Climate Analysis and Regridding

A compact pipeline to analyze how local station temperature anomalies scale with global annual anomalies, produce station-level diagnostics, and aggregate results onto a 5×5° grid for mapping and spatial analysis.

Key features:
- Station-level regressions of local anomaly vs global anomaly
- Automated diagnostic plots and interactive HTML maps
- 5×5° regridding of station outputs with aggregated metrics
- Small utilities and notebooks for exploratory and multilinear analyses

## Quick Start

Create and activate a Python virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the full station-level pipeline:

```bash
python main.py
```

For a faster test run on a small sample:

```bash
python test_pipeline.py
```

Regrid station outputs to a 5×5° grid (run after `main.py`):

```bash
python regrid.py
# or with a custom color metric:
python regrid.py --color-metric climo_humidity
```

## Files & Layout

Top-level overview:

```text
.
├── climate_analysis/        # core analysis, data loading, mapping
├── data/                    # input datasets (not tracked here)
├── multilinear/             # notebooks and analysis helpers
├── output/                  # generated CSVs, maps, and plots
├── main.py                  # pipeline entry point for station-level analysis
├── regrid.py                # creates 5x5° aggregated grid outputs
├── test_pipeline.py         # small-sample test runner
└── requirements.txt
```

## Required Input Data
Place the following (large) files in `data/` before running the full pipeline:
- `global_temps.txt` — global annual anomaly series
- `ghcnm.tavg.v4.0.1.20260224.qcf.inv` — station inventory (coords & metadata)
- `ghcnm.tavg.v4.0.1.20260224.qcf.dat` — station monthly/annual temperature data
- `grid_10min_reh.dat` — gridded relative-humidity climatology (used for `climo_humidity`)

## Outputs
Typical outputs written to `output/`:
- `station_data.csv` — station-level summary with regression results and features
- `station_data_grid_5x5.csv` — aggregated 5×5° grid summary
- `world_climate_map.html`, `world_climate_map_grid_5x5.html` — interactive maps
- `plots/<StationID>/` — diagnostic plots per station (regression, residuals, timeseries)

## Notebooks
Explore interactive analyses in `multilinear/` and `notebooks/` for model experiments and plotting.

## Contributing / Development
- Use `test_pipeline.py` for quick iteration.
- Follow the existing code style; keep changes focused and document new scripts.

## Contact
Open an issue or contact the repository owner for dataset access, questions, or to report bugs.

---
This README was updated to clarify usage and provide a concise quick-start. If you'd like a badge, license, or more detailed API docs, tell me what to add.
