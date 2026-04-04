"""Shared utilities for Siege Culture analysis pipeline."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Project paths ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_CSV = DATA_RAW / "csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
NETWORKS_DIR = DATA_PROCESSED / "networks"

# ── /pol/ platform paths ──────────────────────────────────────────────
POL_RAW = PROJECT_ROOT / "data" / "pol"
POL_ARCHIVE = POL_RAW / "pol.csv.tar.gz"

for _d in (DATA_PROCESSED, FIGURES_DIR, RESULTS_DIR, NETWORKS_DIR, POL_RAW):
    _d.mkdir(parents=True, exist_ok=True)

# ── Zeiger constants ──────────────────────────────────────────────────
ZEIGER_MEMBER_ID = 2170

# ── Plot styling ──────────────────────────────────────────────────────
PLOT_STYLE = "seaborn-v0_8-whitegrid"
CB_PALETTE = sns.color_palette("colorblind")
FIGSIZE_WIDE = (12, 6)
FIGSIZE_SQUARE = (8, 8)
DPI = 300


def setup_plot_style():
    """Apply consistent plot styling across the project."""
    plt.style.use(PLOT_STYLE)
    sns.set_palette("colorblind")
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.figsize": FIGSIZE_WIDE,
    })


def save_figure(fig, name: str):
    """Save figure as both PNG and PDF."""
    for ext in ("png", "pdf"):
        path = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"  Saved figure: {name}")


def load_parquet(name: str):
    """Load a parquet file from the processed directory."""
    import polars as pl
    path = DATA_PROCESSED / name
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pl.read_parquet(path)
