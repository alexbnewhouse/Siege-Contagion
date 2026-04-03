"""Tests for 11_summary_report.py – report generation."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_mod = importlib.import_module("11_summary_report")
format_p = _mod.format_p
sig_stars = _mod.sig_stars


class TestFormatP:
    def test_small_p(self):
        assert format_p(0.0001) == "p < 0.001"

    def test_medium_p(self):
        assert format_p(0.043) == "p = 0.043"

    def test_large_p(self):
        assert format_p(0.5) == "p = 0.500"


class TestSigStars:
    def test_three_stars(self):
        assert sig_stars(0.0001) == "***"

    def test_two_stars(self):
        assert sig_stars(0.005) == "**"

    def test_one_star(self):
        assert sig_stars(0.04) == "*"

    def test_dagger(self):
        assert sig_stars(0.08) == "†"

    def test_no_stars(self):
        assert sig_stars(0.5) == ""
