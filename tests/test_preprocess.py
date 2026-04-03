"""Tests for 01_preprocess.py – HTML stripping and text cleaning."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_mod = importlib.import_module("01_preprocess")
strip_html = _mod.strip_html
strip_html_full = _mod.strip_html_full


class TestStripHtml:
    def test_plain_text_unchanged(self):
        assert strip_html("Hello world") == "Hello world"

    def test_removes_tags(self):
        result = strip_html("<p>Hello</p>")
        assert "Hello" in result
        assert "<p>" not in result

    def test_removes_blockquotes(self):
        html = '<blockquote>Quoted text</blockquote>My original text'
        result = strip_html(html)
        assert "Quoted text" not in result
        assert "My original text" in result

    def test_removes_ipsquote_divs(self):
        html = '<div data-ipsquote="true">Quoted</div>Original'
        result = strip_html(html)
        assert "Quoted" not in result
        assert "Original" in result

    def test_handles_none(self):
        assert strip_html(None) == ""

    def test_handles_empty_string(self):
        assert strip_html("") == ""

    def test_handles_whitespace_only(self):
        assert strip_html("   ") == ""

    def test_normalises_whitespace(self):
        result = strip_html("Hello   world")
        assert result == "Hello world"

    def test_nested_html(self):
        html = "<div><p><b>Bold text</b> normal</p></div>"
        result = strip_html(html)
        assert "Bold text" in result
        assert "normal" in result

    def test_preserves_text_content_with_entities(self):
        html = "<p>Rock &amp; Roll</p>"
        result = strip_html(html)
        assert "Rock" in result
        assert "Roll" in result


class TestStripHtmlFull:
    def test_includes_quoted_text(self):
        html = '<blockquote>Quoted text</blockquote>My original text'
        result = strip_html_full(html)
        assert "Quoted text" in result
        assert "My original text" in result

    def test_handles_none(self):
        assert strip_html_full(None) == ""
