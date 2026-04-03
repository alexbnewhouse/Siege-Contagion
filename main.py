#!/usr/bin/env python3
"""Siege Culture Diffusion Analysis – Pipeline Orchestrator.

Runs all pipeline stages in order.  Each stage is idempotent and writes
its outputs to data/processed/ or results/.

Usage:
    uv run python main.py              # run all stages
    uv run python main.py --from 4     # resume from stage 04
    uv run python main.py --only 2 3   # run only stages 02 and 03
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

STAGES: list[tuple[str, str]] = [
    ("00_ingest",              "Data ingestion (CSV → Parquet)"),
    ("01_preprocess",          "Preprocessing & treatment dates"),
    ("02_siege_lexicon",       "Dictionary-based siege scoring"),
    ("03_siege_embeddings",    "Embedding-based siege scoring"),
    ("04_its_analysis",        "H1 – Interrupted time series"),
    ("05_network_construction","Network construction"),
    ("06_contagion_model",     "H2 – Social contagion model"),
    ("07_granger_causality",   "H3 – Granger causality"),
    ("08_cohort_analysis",     "H4 – Cohort-stratified analysis"),
    ("09_dm_pipeline",         "H5 – Private-to-public pipeline"),
    ("10_reputation_diffusion","H6 – Reputation-mediated diffusion"),
    ("11_summary_report",      "Summary report generation"),
]


def run_stage(module_name: str, description: str) -> None:
    print(f"\n{'━' * 60}")
    print(f"  STAGE {module_name}  –  {description}")
    print(f"{'━' * 60}")
    t0 = time.perf_counter()
    mod = importlib.import_module(module_name)
    mod.main()
    elapsed = time.perf_counter() - t0
    print(f"\n  ⏱  {module_name} finished in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Run the Siege Contagion pipeline.")
    parser.add_argument(
        "--from", dest="from_stage", type=int, default=0,
        help="Resume from stage number (e.g. --from 4 starts at 04_its_analysis)",
    )
    parser.add_argument(
        "--only", nargs="*", type=int, default=None,
        help="Run only the listed stage numbers (e.g. --only 2 3)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SIEGE CULTURE DIFFUSION ANALYSIS PIPELINE")
    print("=" * 60)

    pipeline_start = time.perf_counter()
    for module_name, description in STAGES:
        stage_num = int(module_name[:2])
        if args.only is not None and stage_num not in args.only:
            continue
        if stage_num < args.from_stage:
            continue
        try:
            run_stage(module_name, description)
        except Exception as exc:
            print(f"\n  ✗ STAGE {module_name} FAILED: {exc}")
            raise

    elapsed = time.perf_counter() - pipeline_start
    print(f"\n{'=' * 60}")
    print(f"  ✓ Pipeline complete  ({elapsed:.1f}s total)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
