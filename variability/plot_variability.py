"""Plot variability NetCDF outputs (mean and std) and save PNGs.

Usage:
    python -m variability.plot_variability --in-dir output/variability --out-dir output/variability/plots
"""

import argparse
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np


def plot_ds(ds: xr.Dataset, var_name: str, out_path: Path, title: str = None):
    da = ds[var_name]
    plt.figure(figsize=(10, 5))
    try:
        im = da.plot(cmap="RdBu_r")
        plt.title(title or f"{var_name}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    except Exception:
        # fallback: convert to numpy grid
        data = np.asarray(da)
        plt.imshow(data, origin="lower", cmap="RdBu_r")
        plt.colorbar()
        plt.title(title or f"{var_name}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()


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
        if "mean" in ds:
            plot_ds(ds, "mean", out_dir / f"{base}_mean.png", title=f"Mean — {base}")
            print(f"Saved mean plot for {base}")
        if "std" in ds:
            plot_ds(ds, "std", out_dir / f"{base}_std.png", title=f"Std Dev — {base}")
            print(f"Saved std plot for {base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    main(args)
