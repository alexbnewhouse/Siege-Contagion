"""Tests for utils.py shared utilities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import (
    PROJECT_ROOT, DATA_RAW, DATA_PROCESSED, FIGURES_DIR, RESULTS_DIR,
    ZEIGER_MEMBER_ID, setup_plot_style,
)


class TestProjectPaths:
    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists()

    def test_data_processed_exists(self):
        assert DATA_PROCESSED.exists()

    def test_figures_dir_exists(self):
        assert FIGURES_DIR.exists()

    def test_results_dir_exists(self):
        assert RESULTS_DIR.exists()


class TestConstants:
    def test_zeiger_id(self):
        assert ZEIGER_MEMBER_ID == 2170


class TestPlotSetup:
    def test_setup_does_not_raise(self):
        setup_plot_style()
