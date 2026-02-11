"""Tests for skills/patch_edit.py – especially fuzzy matching."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from schemas.patch_edit_ir import PatchEdit
from skills.patch_edit import (
    EditApplyResult,
    StructuredEditError,
    _fuzzy_find,
    _normalize_ws,
    _replace_block,
    _insert_around_anchor,
    apply_structured_edits,
)


# ---------------------------------------------------------------------------
# _normalize_ws
# ---------------------------------------------------------------------------

class TestNormalizeWs:
    def test_collapse_spaces(self):
        assert _normalize_ws("a   b") == "a b"

    def test_collapse_tabs(self):
        assert _normalize_ws("a\t\tb") == "a b"

    def test_strip_trailing(self):
        assert _normalize_ws("a  \nb  ") == "a\nb"

    def test_crlf(self):
        assert _normalize_ws("a\r\nb") == "a\nb"

    def test_mixed(self):
        assert _normalize_ws("  for (int j  = 0;  j < n; j++)  ") == " for (int j = 0; j < n; j++)"


# ---------------------------------------------------------------------------
# _fuzzy_find
# ---------------------------------------------------------------------------

class TestFuzzyFind:
    def test_exact_match(self):
        text = "hello world"
        idx, length = _fuzzy_find(text, "world", "f.cpp")
        assert text[idx:idx+length] == "world"

    def test_trailing_whitespace_mismatch(self):
        text = "  int x = 0;  \n  int y = 1;\n"
        target = "  int x = 0;\n  int y = 1;\n"
        idx, length = _fuzzy_find(text, target, "f.cpp")
        assert "int x" in text[idx:idx+length]
        assert "int y" in text[idx:idx+length]

    def test_tab_vs_spaces(self):
        text = "\tfor (int i = 0; i < n; i++)"
        target = "    for (int i = 0; i < n; i++)"
        idx, length = _fuzzy_find(text, target, "f.cpp")
        assert "for" in text[idx:idx+length]

    def test_extra_internal_spaces(self):
        text = "double  *xi  =  x[i];"
        target = "double *xi = x[i];"
        idx, length = _fuzzy_find(text, target, "f.cpp")
        assert text[idx:idx+length] == text  # whole string matches

    def test_line_by_line_stripped(self):
        text = "    if (flag) {\n      compute();\n    }\n"
        target = "if (flag) {\n  compute();\n}"
        idx, length = _fuzzy_find(text, target, "f.cpp")
        assert "if (flag)" in text[idx:idx+length]

    def test_not_found(self):
        with pytest.raises(StructuredEditError, match="not_found"):
            _fuzzy_find("hello", "goodbye", "f.cpp")

    def test_not_unique_exact(self):
        with pytest.raises(StructuredEditError, match="not_unique"):
            _fuzzy_find("ab ab", "ab", "f.cpp")

    def test_not_unique_fuzzy(self):
        text = "int x = 0;\nint x = 0;\n"
        target = "int  x  =  0;"
        with pytest.raises(StructuredEditError, match="not_unique"):
            _fuzzy_find(text, target, "f.cpp")


# ---------------------------------------------------------------------------
# _replace_block with fuzzy matching
# ---------------------------------------------------------------------------

class TestReplaceBlock:
    def test_exact(self):
        result = _replace_block("hello world", "world", "earth", "f.cpp")
        assert result == "hello earth"

    def test_whitespace_tolerance(self):
        text = "  double  *xi  =  x[i];"
        old = "double *xi = x[i];"
        new = "const auto &xi = x[i];"
        result = _replace_block(text, old, new, "f.cpp")
        assert "const auto &xi" in result
        # fuzzy match replaces the matched region; leading space before match preserved
        assert result.startswith(" ")


# ---------------------------------------------------------------------------
# _insert_around_anchor with fuzzy matching
# ---------------------------------------------------------------------------

class TestInsertAroundAnchor:
    def test_insert_before(self):
        text = "  for (int j = 0; j < n; j++) {\n"
        anchor = "for (int j = 0; j < n; j++)"
        new = "// cached\n"
        result = _insert_around_anchor(text, anchor, new, before=True, path="f.cpp")
        assert result.index("// cached") < result.index("for")

    def test_insert_after_fuzzy(self):
        text = "  for (int  j = 0;  j < n;  j++) {\n    body;\n"
        anchor = "for (int j = 0; j < n; j++) {"
        new = "\n    // injected"
        result = _insert_around_anchor(text, anchor, new, before=False, path="f.cpp")
        assert "// injected" in result


# ---------------------------------------------------------------------------
# apply_structured_edits integration
# ---------------------------------------------------------------------------

class TestApplyStructuredEdits:
    def test_replace_with_whitespace_diff(self, tmp_path: Path):
        src = tmp_path / "pair.cpp"
        src.write_text("  double  *xi  =  x[i];\n  use(xi);\n")
        edit = PatchEdit(
            file="pair.cpp",
            op="replace",
            anchor="double *xi = x[i];",
            old_text="double *xi = x[i];",
            new_text="const auto &xi = x[i];",
        )
        result = apply_structured_edits(tmp_path, [edit], ["pair.cpp"])
        assert isinstance(result, EditApplyResult)
        assert "const auto &xi" in result.patch_diff
        assert "pair.cpp" in result.touched_files
