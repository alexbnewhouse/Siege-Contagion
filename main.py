#!/usr/bin/env python3
"""Siege Culture Diffusion Analysis – Pipeline Orchestrator.

Runs all pipeline stages in order.  Each stage is idempotent and writes
its outputs to data/processed/ or results/.

Usage:
    uv run python main.py                    # run all IM stages
    uv run python main.py --from 4           # resume from stage 04
    uv run python main.py --only 2 3         # run only stages 02 and 03
    uv run python main.py --platform pol     # run /pol/ pipeline
    uv run python main.py --platform both    # run IM + /pol/ + cross-platform
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# ── Iron March stages ─────────────────────────────────────────────────
IM_STAGES: list[tuple[str, str]] = [
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
    ("12_reputation_reinforcement", "H7 – Reputation reinforcement"),
    ("13_thread_escalation",   "H8 – Within-thread escalation"),
    ("14_thread_exposure",     "H9 – Thread exposure → adoption"),
    ("15_semantic_convergence","H10 – Semantic convergence"),
    ("16_subforum_diffusion",  "H11 – Subforum diffusion geography"),
    ("11_summary_report",      "Summary report generation"),
]

# ── /pol/ platform stages ─────────────────────────────────────────────
POL_STAGES: list[tuple[str, str]] = [
    ("00b_ingest_pol",         "/pol/ data ingestion (tar.gz → Parquet)"),
    ("01b_preprocess_pol",     "/pol/ preprocessing & HTML cleaning"),
    ("02b_siege_lexicon_pol",  "/pol/ dictionary-based siege scoring"),
    ("03b_siege_embeddings_pol", "/pol/ embedding-based siege scoring"),
    ("20_pol_thread_escalation", "H8-pol – /pol/ within-thread escalation"),
    ("21_pol_semantic_convergence", "H10-pol – /pol/ semantic convergence"),
]

# ── Cross-platform stages ────────────────────────────────────────────
CROSS_STAGES: list[tuple[str, str]] = [
    ("17_cross_platform_its",  "H12 – Cross-platform ITS"),
    ("18_cross_platform_granger", "H13 – Cross-platform Granger causality"),
    ("19_cross_platform_bridges", "H14 – Cross-platform content bridges"),
    ("22_shutdown_its",        "H15 – Post-shutdown migration ITS"),
    ("23_vocab_adoption_lags", "H16 – Vocabulary adoption lag curves"),
    ("24_transfer_entropy",    "H17 – Transfer entropy"),
    ("25_country_correlation", "H18 – Country-level correlation"),
    ("26_dose_response",       "H19 – Dose-response at multiple lags"),
    ("27_subtheme_diffusion",  "H20 – Sub-theme disaggregation"),
    ("28_domain_diffusion",    "H21 – URL domain diffusion"),
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
    parser.add_argument(
        "--platform", choices=["im", "pol", "both"], default="im",
        help="Which platform pipeline to run: im (default), pol, or both",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SIEGE CULTURE DIFFUSION ANALYSIS PIPELINE")
    print("=" * 60)

    # Select which stage lists to run
    if args.platform == "im":
        stages = IM_STAGES
    elif args.platform == "pol":
        stages = POL_STAGES
    elif args.platform == "both":
        stages = IM_STAGES + POL_STAGES + CROSS_STAGES

    pipeline_start = time.perf_counter()
    for module_name, description in stages:
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
