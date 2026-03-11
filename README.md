# Climate Analysis and Regridding

This repository analyzes how local station temperature anomalies scale with global temperature anomalies, builds station-level diagnostic outputs, and provides regridded spatial summaries for downstream modeling.

The main workflow:
- loads global annual anomalies and GHCN-M station data
- filters stations to those with enough post-1930 overlap
- computes local anomalies relative to each station's own baseline
- fits a linear relationship between local and global anomalies
- saves station-level summary data and diagnostic plots
- generates interactive maps from both station-level and 5x5 regridded data

## Repository Layout

```text
bc3_urop/
├── climate_analysis/
│   ├── analysis.py
│   ├── data_loader.py
│   ├── mapping.py
│   └── plotting.py
├── data/
│   ├── global_temps.txt
│   ├── ghcnm.tavg.v4.0.1.20260224.qcf.inv
│   ├── ghcnm.tavg.v4.0.1.20260224.qcf.dat
│   ├── grid_10min_reh.dat
│   └── PatternScalingCoefficients_tas_ssp245-ssp370__r240x120.nc
├── multilinear/
│   ├── spatial_analysis.py
│   ├── slope_comparison.py
│   ├── multilinear_analysis.ipynb
│   └── decision_tree_analysis.ipynb
├── output/
├── main.py
├── regrid.py
├── test_pipeline.py
└── requirements.txt
```

## Data Inputs

The pipeline expects these files in `data/`:
- `global_temps.txt`: global annual anomaly data
- `ghcnm.tavg.v4.0.1.20260224.qcf.inv`: station inventory with coordinates and metadata
- `ghcnm.tavg.v4.0.1.20260224.qcf.dat`: station monthly temperature data
- `grid_10min_reh.dat`: gridded relative humidity climatology used to add `climo_humidity`

## Environment Setup

This repo uses standard Python scientific libraries listed in `requirements.txt`.

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main Pipeline

Run the full station-level analysis with:

```bash
python main.py
```

What `main.py` does:
- parses the global anomaly series and station inventory
- loads annual means for each station from the GHCN-M `.dat` file
- keeps only stations with data from 1930 onward and at least 50 overlapping years
- computes local temperature anomalies relative to the station mean
- regresses local anomalies on global anomalies
- derives climatological temperature and humidity features
- writes station-level outputs and a world map

### Main Outputs

After running `main.py`, the main outputs are:
- `output/station_data.csv`: station-level summary table
- `output/world_climate_map.html`: interactive station map
- `output/plots/<StationID>/regression.png`: regression plot
- `output/plots/<StationID>/residual.png`: residual plot
- `output/plots/<StationID>/timeseries.png`: anomaly time series plot

### Station-Level Fields

`output/station_data.csv` includes fields such as:
- station metadata: `StationID`, `StationName`, `Latitude`, `Longitude`, `Elevation`
- regression results: `slope`, `intercept`, `r_squared`
- diagnostic plot paths
- climatological features: `climo_temp`, `climo_humidity`

## Sample Run for Faster Testing

To run a smaller sample of 100 stations instead of the full inventory:

```bash
python test_pipeline.py
```

This writes its outputs under:
- `output/test_100/station_data.csv`
- `output/test_100/world_climate_map.html`
- `output/test_100/plots/`

## 5x5 Regridding Workflow

The repo also includes a spatial regridding script that aggregates station-level outputs onto a 5x5 latitude/longitude grid.

Run it after generating `output/station_data.csv`:

```bash
python regrid.py
```

You can also choose a different coloring metric for the map:

```bash
python regrid.py --color-metric climo_humidity
```

### Regridding Behavior

`regrid.py`:
- bins stations into 5x5 degree latitude/longitude cells
- computes the mean of all numeric station fields in each occupied cell
- computes `slope_min`, `slope_max`, and `slope_std` for each cell
- skips empty cells entirely
- creates a filled-cell interactive map
- adds a slider to filter visible cells by minimum station count

### Regridding Outputs

- `output/station_data_grid_5x5.csv`: aggregated grid-cell summary table
- `output/world_climate_map_grid_5x5.html`: interactive filled-grid map

The regridded map includes:
- cell popups with aggregated climate metrics
- hover text showing grid-cell bounds
- a station-count slider to filter cells by occupancy threshold

## Multilinear Analysis

The `multilinear/` directory contains simple regression and comparison utilities built on top of `output/station_data.csv`.

### Spatial Regression

```bash
python multilinear/spatial_analysis.py
```

This fits an OLS model for station `slope` using:
- `climo_temp`
- `climo_humidity`

It writes:
- `output/spatial_analysis_simple.csv`

### Predicted vs Actual Slope Comparison

```bash
python multilinear/slope_comparison.py
```

This script:
- applies regression coefficients to generate predicted slopes
- compares predicted and actual station slopes
- saves a comparison CSV
- produces diagnostic plots

Outputs include:
- `output/slope_comparison_simple.csv`
- `output/multilinear/plot_actual_vs_predicted.png`
- `output/multilinear/plot_residual_map.png`

## Notebooks

The repository also contains exploratory notebooks in `multilinear/`:
- `multilinear_analysis.ipynb`
- `decision_tree_analysis.ipynb`

These are useful for interactive model experimentation beyond the main scripted pipeline.

## Notes

- The main regression is station-local anomaly vs global anomaly, not a direct time-trend regression.
- Local anomalies are computed relative to each station's own mean over its retained period of record.
- Humidity is assigned from the nearest humidity grid cell and averaged across its 12 monthly climatology values.
- The scripts write outputs into `output/` and will create missing subdirectories automatically.
