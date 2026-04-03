"""Tests for 06_contagion_model.py – network exposure computation."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_mod = importlib.import_module("06_contagion_model")
compute_network_exposure = _mod.compute_network_exposure


class TestComputeNetworkExposure:
    def test_basic_exposure(self):
        user_scores = {
            (1, "2015-06"): 5.0,
            (2, "2015-06"): 3.0,
            (3, "2015-06"): 1.0,
        }
        neighbors = {1: {2, 3}, 2: {1}, 3: {1}}
        users = {1, 2, 3}
        months = ["2015-06", "2015-07"]

        exposure = compute_network_exposure(user_scores, neighbors, users, months)
        # User 1's exposure in 2015-07 = mean(score_2_june, score_3_june) = (3+1)/2 = 2.0
        assert abs(exposure[(1, "2015-07")] - 2.0) < 0.01
        # User 2's exposure in 2015-07 = mean(score_1_june) = 5.0
        assert abs(exposure[(2, "2015-07")] - 5.0) < 0.01

    def test_no_neighbors(self):
        user_scores = {(1, "2015-06"): 5.0}
        neighbors = {}
        users = {1}
        months = ["2015-06", "2015-07"]

        exposure = compute_network_exposure(user_scores, neighbors, users, months)
        assert exposure[(1, "2015-07")] == 0.0

    def test_first_month_skipped(self):
        user_scores = {(1, "2015-06"): 5.0}
        neighbors = {1: {2}}
        users = {1}
        months = ["2015-06"]

        exposure = compute_network_exposure(user_scores, neighbors, users, months)
        # Only one month, so no exposure computed (first month is skipped)
        assert len(exposure) == 0
