"""Tests for the five new exegesis-theory hypothesis modules (H7–H11)."""

import importlib
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── H7: Reputation Reinforcement ─────────────────────────────────────

_mod12 = importlib.import_module("12_reputation_reinforcement")
build_post_reputation_df = _mod12.build_post_reputation_df


class TestReputationReinforcement:
    """Tests for 12_reputation_reinforcement.py."""

    def test_build_post_reputation_df_returns_rep_count(self, tmp_path, monkeypatch):
        """build_post_reputation_df should join rep counts to forum posts."""
        # Create minimal forum_posts parquet
        fp = pl.DataFrame({
            "pid": [1, 2, 3],
            "author_id": [10, 20, 30],
            "post_date": [None, None, None],
            "post": ["a", "b", "c"],
            "topic_id": [100, 100, 200],
            "word_count": [5, 10, 15],
            "siege_binary": [1, 0, 1],
            "siege_keyword_score": [0.5, 0.0, 0.3],
            "siege_keyword_density": [0.1, 0.0, 0.02],
            "siege_keyword_count": [1, 0, 1],
            "siege_similarity": [0.5, 0.1, 0.3],
            "siege_keyword_context_score": [0.0, 0.0, 0.0],
            "siege_keyword_score_adjusted": [0.5, 0.0, 0.3],
            "text": ["hello", "world", "test"],
            "text_full": ["hello", "world", "test"],
        })

        rep = pl.DataFrame({
            "id": [0, 0, 0, 0],
            "member_id": [99, 98, 97, 96],
            "app": ["forums", "forums", "forums", "forums"],
            "type": ["pid", "pid", "pid", "status_id"],
            "type_id": [1, 1, 3, 999],
            "rep_date": [None, None, None, None],
            "member_received": [10, 10, 30, 50],
            "rep_class": ["x", "x", "x", "x"],
            "item_id": [100, 100, 200, 0],
            "class_type_id_hash": ["a", "b", "c", "d"],
        })

        fp.write_parquet(tmp_path / "forum_posts.parquet")
        rep.write_parquet(tmp_path / "core_reputation_index.parquet")

        import utils
        monkeypatch.setattr(utils, "DATA_PROCESSED", tmp_path)
        # Reload module to pick up patched path
        importlib.reload(_mod12)

        df = _mod12.build_post_reputation_df()
        assert "rep_count" in df.columns
        # pid 1 got 2 reps, pid 2 got 0, pid 3 got 1
        counts = df.sort("pid")["rep_count"].to_list()
        assert counts == [2, 0, 1]

    def test_rep_count_fills_null_with_zero(self, tmp_path, monkeypatch):
        """Posts without any reputation events should have rep_count = 0."""
        fp = pl.DataFrame({
            "pid": [1],
            "author_id": [10],
            "post_date": [None],
            "post": ["a"],
            "topic_id": [100],
            "word_count": [5],
            "siege_binary": [0],
            "siege_keyword_score": [0.0],
            "siege_keyword_density": [0.0],
            "siege_keyword_count": [0],
            "siege_similarity": [0.0],
            "siege_keyword_context_score": [0.0],
            "siege_keyword_score_adjusted": [0.0],
            "text": ["hi"],
            "text_full": ["hi"],
        })
        rep = pl.DataFrame({
            "id": pl.Series([], dtype=pl.Int64),
            "member_id": pl.Series([], dtype=pl.Int64),
            "app": pl.Series([], dtype=pl.Utf8),
            "type": pl.Series([], dtype=pl.Utf8),
            "type_id": pl.Series([], dtype=pl.Int64),
            "rep_date": pl.Series([], dtype=pl.Utf8),
            "member_received": pl.Series([], dtype=pl.Int64),
            "rep_class": pl.Series([], dtype=pl.Utf8),
            "item_id": pl.Series([], dtype=pl.Int64),
            "class_type_id_hash": pl.Series([], dtype=pl.Utf8),
        })
        fp.write_parquet(tmp_path / "forum_posts.parquet")
        rep.write_parquet(tmp_path / "core_reputation_index.parquet")

        import utils
        monkeypatch.setattr(utils, "DATA_PROCESSED", tmp_path)
        importlib.reload(_mod12)
        df = _mod12.build_post_reputation_df()
        assert df["rep_count"].to_list() == [0]


# ── H8: Within-Thread Escalation ──────────────────────────────────────

_mod13 = importlib.import_module("13_thread_escalation")
build_thread_position_df = _mod13.build_thread_position_df


class TestThreadEscalation:
    """Tests for 13_thread_escalation.py."""

    def test_build_thread_position_adds_position(self, tmp_path, monkeypatch):
        """Siege threads should get position_norm column."""
        from datetime import datetime
        fp = pl.DataFrame({
            "pid": list(range(1, 7)),
            "author_id": [10, 20, 30, 10, 20, 30],
            "post_date": [
                datetime(2016, 1, 1), datetime(2016, 1, 2), datetime(2016, 1, 3),
                datetime(2016, 2, 1), datetime(2016, 2, 2), datetime(2016, 2, 3),
            ],
            "post": ["a"] * 6,
            "topic_id": [100, 100, 100, 200, 200, 200],
            "word_count": [10] * 6,
            "siege_binary": [1, 0, 0, 0, 0, 0],  # Only thread 100 is a Siege thread
            "siege_keyword_score": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            "siege_keyword_density": [0.05] * 6,
            "siege_keyword_count": [1, 0, 0, 0, 0, 0],
            "siege_similarity": [0.3] * 6,
            "siege_keyword_context_score": [0.0] * 6,
            "siege_keyword_score_adjusted": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            "text": ["siege"] * 6,
            "text_full": ["siege"] * 6,
        })
        fp.write_parquet(tmp_path / "forum_posts.parquet")

        import utils
        monkeypatch.setattr(utils, "DATA_PROCESSED", tmp_path)
        # Override ZEIGER_MEMBER_ID to not filter anyone
        monkeypatch.setattr(utils, "ZEIGER_MEMBER_ID", 99999)
        importlib.reload(_mod13)

        df = _mod13.build_thread_position_df()
        assert "position_norm" in df.columns
        # Only thread 100 should be present (it's a Siege thread with 3 posts)
        assert df["topic_id"].unique().to_list() == [100]
        # Position should be [0, 0.5, 1.0]
        positions = sorted(df["position_norm"].to_list())
        np.testing.assert_allclose(positions, [0.0, 0.5, 1.0], atol=1e-6)

    def test_excludes_short_threads(self, tmp_path, monkeypatch):
        """Threads with < 3 posts should be excluded."""
        from datetime import datetime
        fp = pl.DataFrame({
            "pid": [1, 2],
            "author_id": [10, 20],
            "post_date": [datetime(2016, 1, 1), datetime(2016, 1, 2)],
            "post": ["a", "b"],
            "topic_id": [100, 100],
            "word_count": [10, 10],
            "siege_binary": [1, 0],
            "siege_keyword_score": [0.5, 0.0],
            "siege_keyword_density": [0.05, 0.0],
            "siege_keyword_count": [1, 0],
            "siege_similarity": [0.3, 0.1],
            "siege_keyword_context_score": [0.0, 0.0],
            "siege_keyword_score_adjusted": [0.5, 0.0],
            "text": ["siege", "hello"],
            "text_full": ["siege", "hello"],
        })
        fp.write_parquet(tmp_path / "forum_posts.parquet")

        import utils
        monkeypatch.setattr(utils, "DATA_PROCESSED", tmp_path)
        monkeypatch.setattr(utils, "ZEIGER_MEMBER_ID", 99999)
        importlib.reload(_mod13)

        df = _mod13.build_thread_position_df()
        assert df.height == 0


# ── H9: Thread Exposure ───────────────────────────────────────────────

class TestThreadExposure:
    """Tests for 14_thread_exposure.py."""

    def test_module_imports(self):
        """Module should import without error."""
        mod = importlib.import_module("14_thread_exposure")
        assert hasattr(mod, "main")


# ── H10: Semantic Convergence ─────────────────────────────────────────

class TestSemanticConvergence:
    """Tests for 15_semantic_convergence.py."""

    def test_module_imports(self):
        """Module should import without error."""
        mod = importlib.import_module("15_semantic_convergence")
        assert hasattr(mod, "main")


# ── H11: Subforum Diffusion ──────────────────────────────────────────

class TestSubforumDiffusion:
    """Tests for 16_subforum_diffusion.py."""

    def test_module_imports(self):
        """Module should import without error."""
        mod = importlib.import_module("16_subforum_diffusion")
        assert hasattr(mod, "main")

    def test_herfindahl_calculation(self):
        """Herfindahl index should be correct for simple case."""
        shares = np.array([0.5, 0.3, 0.2])
        hhi = float(np.sum(shares ** 2))
        assert abs(hhi - 0.38) < 0.001

    def test_herfindahl_monopoly(self):
        """HHI of a single-subforum concentration should be 1.0."""
        shares = np.array([1.0])
        hhi = float(np.sum(shares ** 2))
        assert abs(hhi - 1.0) < 0.001

    def test_herfindahl_even_split(self):
        """HHI of perfectly even split across 4 should be 0.25."""
        shares = np.array([0.25, 0.25, 0.25, 0.25])
        hhi = float(np.sum(shares ** 2))
        assert abs(hhi - 0.0625 * 4) < 0.001
