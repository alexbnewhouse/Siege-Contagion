"""Tests for 00_ingest.py – data ingestion utilities."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl

_mod = importlib.import_module("00_ingest")
_safe_int_cols = _mod._safe_int_cols
_parse_unix_ts = _mod._parse_unix_ts


class TestSafeIntCols:
    def test_converts_float_to_int(self):
        df = pl.DataFrame({"id": [1.0, 2.0, 3.0]})
        result = _safe_int_cols(df, ["id"])
        assert result["id"].dtype == pl.Int64

    def test_handles_nan(self):
        df = pl.DataFrame({"id": [1.0, float("nan"), 3.0]})
        result = _safe_int_cols(df, ["id"])
        assert result["id"].dtype == pl.Int64

    def test_ignores_missing_columns(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = _safe_int_cols(df, ["nonexistent"])
        assert result.shape == df.shape

    def test_leaves_int_unchanged(self):
        df = pl.DataFrame({"id": [1, 2, 3]})
        result = _safe_int_cols(df, ["id"])
        assert result["id"].to_list() == [1, 2, 3]


class TestParseUnixTs:
    def test_converts_timestamp(self):
        df = pl.DataFrame({"ts": [1420070400]})  # 2015-01-01 00:00:00 UTC
        result = _parse_unix_ts(df, ["ts"])
        assert result["ts"].dtype == pl.Datetime
        dt = result["ts"][0]
        assert dt.year == 2015
        assert dt.month == 1
        assert dt.day == 1

    def test_handles_float_timestamps(self):
        df = pl.DataFrame({"ts": [1420070400.0]})
        result = _parse_unix_ts(df, ["ts"])
        assert result["ts"].dtype == pl.Datetime

    def test_handles_iso_string_dates(self):
        df = pl.DataFrame({"ts": ["2015-01-01T12:00:00", "2016-06-15T08:30:00"]})
        result = _parse_unix_ts(df, ["ts"])
        assert result["ts"].dtype == pl.Datetime
        assert result["ts"][0].year == 2015

    def test_handles_date_only_strings(self):
        df = pl.DataFrame({"ts": ["2015-01-01", "2016-06-15"]})
        result = _parse_unix_ts(df, ["ts"])
        assert result["ts"].dtype == pl.Datetime
        assert result["ts"][0].year == 2015

    def test_ignores_missing_columns(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = _parse_unix_ts(df, ["nonexistent"])
        assert result.shape == df.shape
