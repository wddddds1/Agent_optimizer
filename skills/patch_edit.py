from __future__ import annotations

import difflib
import re
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from schemas.patch_edit_ir import PatchEdit


@dataclass
class EditApplyResult:
    patch_diff: str
    touched_files: List[str]


class StructuredEditError(RuntimeError):
    pass


def apply_structured_edits(
    repo_root: Path,
    edits: List[PatchEdit],
    allowed_files: List[str],
) -> EditApplyResult:
    if not edits:
        raise StructuredEditError("no edits provided")
    normalized_allowed = {str(path) for path in allowed_files}
    originals: Dict[str, str] = {}
    updated: Dict[str, str] = {}
    touched: List[str] = []
    for edit in edits:
        path = edit.file.strip()
        if not path:
            raise StructuredEditError("edit file missing")
        if normalized_allowed and path not in normalized_allowed:
            raise StructuredEditError(f"file not allowed: {path}")
        file_path = (repo_root / path).resolve()
        try:
            file_path.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise StructuredEditError(f"file escapes repo: {path}") from exc
        if not file_path.exists():
            raise StructuredEditError(f"file not found: {path}")
        if path not in originals:
            originals[path] = file_path.read_text(encoding="utf-8")
            updated[path] = originals[path]
        updated[path] = _apply_edit_to_text(updated[path], edit, path)
        if path not in touched:
            touched.append(path)
    patch_chunks: List[str] = []
    for path in touched:
        before = originals[path].splitlines()
        after = updated[path].splitlines()
        if before == after:
            continue
        diff = difflib.unified_diff(
            before,
            after,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
        patch_chunks.append("\n".join(diff))
    patch_text = "\n".join(chunk for chunk in patch_chunks if chunk)
    if patch_text:
        patch_text += "\n"
    if not patch_text.strip():
        raise StructuredEditError("no changes after applying edits")
    return EditApplyResult(patch_diff=patch_text, touched_files=touched)


def _apply_edit_to_text(text: str, edit: PatchEdit, path: str) -> str:
    anchor = edit.anchor
    if not anchor.strip():
        raise StructuredEditError(f"anchor_missing: {path}")
    if edit.op in {"replace", "delete"}:
        if not edit.old_text:
            raise StructuredEditError(f"old_text_missing: {path}")
        if anchor not in edit.old_text:
            raise StructuredEditError(f"anchor_not_in_old_text: {path}")
        return _replace_block(
            text,
            edit.old_text,
            "" if edit.op == "delete" else edit.new_text,
            path,
            anchor=anchor,
        )
    if edit.op in {"insert_before", "insert_after"}:
        if not edit.new_text:
            raise StructuredEditError(f"new_text_missing: {path}")
        return _insert_around_anchor(
            text,
            anchor,
            edit.new_text,
            before=edit.op == "insert_before",
            path=path,
        )
    raise StructuredEditError(f"unsupported_op:{edit.op}:{path}")


def _normalize_ws(text: str) -> str:
    """Collapse horizontal whitespace, strip trailing per line, normalize newlines."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(re.sub(r"[ \t]+", " ", line).rstrip() for line in lines)


def _fuzzy_find(
    text: str, target: str, path: str, label: str = "target"
) -> Tuple[int, int]:
    """Find *target* inside *text* with fallback to whitespace-normalized matching.

    Returns ``(start_index_in_original_text, matched_length_in_original_text)``.
    Raises :class:`StructuredEditError` when zero or multiple matches are found.
    """
    # --- exact match ---
    exact = [m.start() for m in re.finditer(re.escape(target), text)]
    if len(exact) == 1:
        return exact[0], len(target)
    if len(exact) > 1:
        raise StructuredEditError(
            f"{label}_not_unique:{path}:b64:{_encode_anchor(target)}"
        )

    # --- whitespace-normalized match ---
    norm_text = _normalize_ws(text)
    norm_target = _normalize_ws(target)
    norm_matches = [m.start() for m in re.finditer(re.escape(norm_target), norm_text)]
    if len(norm_matches) > 1:
        raise StructuredEditError(
            f"{label}_not_unique:{path}:b64:{_encode_anchor(target)}"
        )
    if len(norm_matches) == 1:
        # Map normalized position back to original text position.
        # Walk both texts in parallel: for each char consumed in norm_text,
        # advance in the original text, skipping extra whitespace.
        norm_pos = norm_matches[0]
        norm_len = len(norm_target)
        orig_start = _map_norm_to_orig(text, norm_text, norm_pos)
        orig_end = _map_norm_to_orig(text, norm_text, norm_pos + norm_len)
        return orig_start, orig_end - orig_start

    # --- line-by-line stripped match ---
    target_lines = [l.strip() for l in target.splitlines() if l.strip()]
    if target_lines:
        text_lines = text.splitlines(keepends=True)
        stripped_text_lines = [l.strip() for l in text_lines]
        for i in range(len(stripped_text_lines) - len(target_lines) + 1):
            if stripped_text_lines[i : i + len(target_lines)] == target_lines:
                # check uniqueness
                second = False
                for j in range(i + 1, len(stripped_text_lines) - len(target_lines) + 1):
                    if stripped_text_lines[j : j + len(target_lines)] == target_lines:
                        second = True
                        break
                if second:
                    raise StructuredEditError(
                        f"{label}_not_unique:{path}:b64:{_encode_anchor(target)}"
                    )
                start = sum(len(l) for l in text_lines[:i])
                end = sum(len(l) for l in text_lines[: i + len(target_lines)])
                return start, end - start
    raise StructuredEditError(
        f"{label}_not_found:{path}:b64:{_encode_anchor(target)}"
    )


def _map_norm_to_orig(original: str, normalized: str, norm_pos: int) -> int:
    """Map a character position in *normalized* back to the corresponding
    position in *original*."""
    oi = 0  # original index
    ni = 0  # normalized index
    while ni < norm_pos and oi < len(original):
        oc = original[oi]
        nc = normalized[ni] if ni < len(normalized) else ""
        if oc == nc:
            oi += 1
            ni += 1
        elif oc in (" ", "\t", "\r"):
            # extra whitespace in original that was collapsed
            oi += 1
        else:
            oi += 1
            ni += 1
    return oi


def _replace_block(
    text: str,
    old_text: str,
    new_text: str,
    path: str,
    anchor: str = "",
) -> str:
    exact = [m.start() for m in re.finditer(re.escape(old_text), text)]
    if len(exact) == 1:
        idx = exact[0]
        return text[:idx] + new_text + text[idx + len(old_text):]
    if len(exact) > 1 and anchor.strip():
        anchor_pos = [m.start() for m in re.finditer(re.escape(anchor), text)]
        if anchor_pos:
            candidates = [
                idx for idx in exact
                if any(idx <= pos < idx + len(old_text) for pos in anchor_pos)
            ]
            if len(candidates) == 1:
                idx = candidates[0]
                return text[:idx] + new_text + text[idx + len(old_text):]
    idx, length = _fuzzy_find(text, old_text, path, label="old_text")
    return text[:idx] + new_text + text[idx + length:]


def _insert_around_anchor(
    text: str,
    anchor: str,
    new_text: str,
    before: bool,
    path: str,
) -> str:
    idx, length = _fuzzy_find(text, anchor, path, label="anchor")
    if before:
        return text[:idx] + new_text + text[idx:]
    return text[:idx + length] + new_text + text[idx + length:]


def _encode_anchor(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")
