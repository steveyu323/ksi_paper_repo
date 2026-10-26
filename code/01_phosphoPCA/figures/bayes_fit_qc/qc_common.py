"""
Shared paths, style and loaders for the Bayesian-fit QC scripts.

The fits being checked are decoupling_out/251125_pymc_{paabd}.nc, produced by
5_enrichment_analysis.ipynb (hierarchical NB, 4 chains x 1000 draws after
1000 tuning steps, NUTS, target_accept 0.9).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))      # the pipeline package root
import config  # noqa: E402

NC_DIR = config.DECOUPLING_DIR          # stage 05 posteriors
OUT = HERE / "out"

PAABD_ORDER = ["sh2", "fha", "1433", "ww"]
PAABD_LABELS = {"sh2": "SH2", "fha": "FHA", "1433": "14-3-3", "ww": "WW"}
PAABD_COLORS = {"sh2": "#2E86AB", "fha": "#E84855", "1433": "#3BB273", "ww": "#F4A261"}
NULL_COLOR = "#9E9E9E"
INK = "#222222"

# Hit calls, as in stage 06 / decoupling_supp_figures.py
LFSR_THRESH = config.LFSR_THRESH
BINDING_THRESH = config.BINDING_THRESH

# Conventional diagnostic thresholds (Vehtari et al. 2021)
RHAT_THRESH = 1.01
ESS_THRESH = 400

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Liberation Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def nc_path(paabd: str) -> Path:
    return NC_DIR / f"251125_pymc_{paabd}.nc"


def open_group(paabd: str, group: str) -> xr.Dataset:
    return xr.open_dataset(nc_path(paabd), group=group, engine="h5netcdf")


def observed_pairs(paabd: str):
    """(substrate index, kinase index) of every pair that has data.

    b is defined on the full substrate x kinase grid, but a cell with no
    observations is pure prior, so all b diagnostics are restricted to pairs
    that appear in the likelihood.
    """
    cd = open_group(paabd, "constant_data")
    pairs = np.unique(np.c_[cd.sub_i.values, cd.kin_i.values], axis=0)
    return pairs[:, 0], pairs[:, 1]


def save_fig(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote out/{name}.pdf/.png")
