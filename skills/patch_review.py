from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Dict, List, Tuple


def review_patch_diff(
    patch_text: str,
    repo_root: Path,
    patch_rules: Dict[str, object],
) -> Tuple[bool, List[str], Dict[str, object]]:
    reasons: List[str] = []
    info: Dict[str, object] = {"files": [], "lines_changed": 0}
    if not patch_text or not patch_text.strip():
        return False, ["empty patch diff"], info
    if "GIT binary patch" in patch_text:
        return False, ["binary patch not allowed"], info

    allowed_globs = patch_rules.get("allowed_globs") if isinstance(patch_rules, dict) else None
    forbidden_patterns = patch_rules.get("forbidden_patterns") if isinstance(patch_rules, dict) else None
    max_lines_changed = _safe_int(patch_rules.get("max_lines_changed"), 0)
    max_files_changed = _safe_int(patch_rules.get("max_files_changed"), 0)

    allowed_globs = allowed_globs if isinstance(allowed_globs, list) else []
    forbidden_patterns = forbidden_patterns if isinstance(forbidden_patterns, list) else []
    compiled_forbidden = [_safe_regex(pat) for pat in forbidden_patterns if isinstance(pat, str)]
    compiled_forbidden = [pat for pat in compiled_forbidden if pat]

    patch_root = patch_rules.get("patch_root") if isinstance(patch_rules, dict) else None

    files = _extract_files(patch_text)
    info["files"] = files
    if files and allowed_globs:
        for path in files:
            # Try matching directly and with patch_root prefix
            if not _matches_any(path, allowed_globs):
                if patch_root and _matches_any(f"{patch_root}/{path}", allowed_globs):
                    continue  # matches with prefix
                reasons.append(f"file not allowed: {path}")
                break
    if max_files_changed and len(files) > max_files_changed:
        reasons.append(f"too many files changed: {len(files)}>{max_files_changed}")

    lines_changed = 0
    meaningful_lines = 0
    for line in patch_text.splitlines():
        if line.startswith(("+++ ", "--- ", "@@")):
            continue
        if line.startswith("+") or line.startswith("-"):
            lines_changed += 1
            stripped = line[1:].strip()
            # Count lines that are not purely whitespace, comments, or #include
            if stripped and not _is_trivial_line(stripped):
                meaningful_lines += 1
            for pattern in compiled_forbidden:
                if pattern.search(line):
                    reasons.append(f"forbidden pattern: {pattern.pattern}")
                    break
    info["lines_changed"] = lines_changed
    info["meaningful_lines"] = meaningful_lines
    if max_lines_changed and lines_changed > max_lines_changed:
        reasons.append(f"too many lines changed: {lines_changed}>{max_lines_changed}")
    if lines_changed > 0 and meaningful_lines == 0:
        reasons.append("trivial patch: no meaningful code changes")

    if reasons:
        return False, reasons, info
    return True, [], info


def _extract_files(patch_text: str) -> List[str]:
    files: List[str] = []
    for line in patch_text.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            path = line[4:].split("\t", 1)[0].strip()
            if path == "/dev/null":
                continue
            if path.startswith(("a/", "b/")):
                path = path[2:]
            if path and path not in files:
                files.append(path)
    return files


def _matches_any(path: str, patterns: List[str]) -> bool:
    posix = path.replace("\\", "/")
    for pattern in patterns:
        if fnmatch.fnmatch(posix, pattern):
            return True
    return False


def _safe_regex(pattern: str) -> re.Pattern[str] | None:
    try:
        return re.compile(pattern)
    except re.error:
        return None


_TRIVIAL_LINE_RE = re.compile(
    r"^("
    r"\s*$"                       # empty / whitespace
    r"|//.*"                       # C++ line comment
    r"|/\*.*\*/$"                  # single-line C block comment
    r"|\*.*"                       # mid-block comment
    r"|#\s*include\b.*"            # #include
    r"|#\s*pragma\b.*"             # #pragma
    r"|\}\s*$"                     # lone closing brace
    r"|\{\s*$"                     # lone opening brace
    r")$"
)


def _is_trivial_line(stripped: str) -> bool:
    """Return True if the line is trivial (whitespace, comment, include, brace)."""
    return bool(_TRIVIAL_LINE_RE.match(stripped))


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
