from __future__ import annotations

from pathlib import Path
import base64
import difflib
import re
from typing import Dict, List, Optional

from pydantic import ValidationError

from orchestrator.errors import LLMUnavailableError
from orchestrator.llm_client import LLMClient
from schemas.action_ir import ActionIR
from schemas.patch_edit_ir import PatchEdit, PatchEditProposal
from schemas.patch_proposal_ir import PatchProposal
from schemas.profile_report import ProfileReport
from skills.applications import validate_patch_edits as app_validate_patch_edits
from skills.patch_edit import StructuredEditError, apply_structured_edits
from skills.profile_payload import build_profile_payload


class CodePatchAgent:
    def __init__(self, llm_client: Optional[LLMClient]) -> None:
        self.llm_client = llm_client
        self.last_trace: Optional[Dict[str, object]] = None

    def propose(
        self,
        action: ActionIR,
        profile: ProfileReport,
        patch_rules: Dict[str, object],
        allowed_files: List[str],
        code_snippets: List[Dict[str, object]],
        repo_root: Path,
        feedback: Optional[str] = None,
        backend_variant: Optional[str] = None,
        reference_template: Optional[Dict[str, str]] = None,
        navigation_hints: Optional[List[Dict[str, object]]] = None,
        app: Optional[str] = None,
    ) -> Optional[PatchProposal]:
        if not self.llm_client or not self.llm_client.config.enabled:
            return None
        prompt = _load_prompt("code_patch")
        uses_dbl3 = backend_variant == "openmp_backend"
        base_payload = {
            "action": {
                "action_id": action.action_id,
                "family": action.family,
                "parameters": action.parameters,
                "expected_effect": action.expected_effect,
                "risk_level": action.risk_level,
            },
            "profile": build_profile_payload(profile),
            "patch_rules": patch_rules,
            "allowed_files": allowed_files,
            "code_snippets": code_snippets,
            "feedback": feedback,
            "backend_hint": "omp_dbl3" if uses_dbl3 else "",
        }
        if reference_template:
            base_payload["reference_template"] = reference_template
        payload = dict(base_payload)
        for attempt in range(2):
            data = self.llm_client.request_json(prompt, payload)
            self.last_trace = {"payload": payload, "response": data}
            if data:
                try:
                    edit_proposal = PatchEditProposal(**data)
                    break
                except ValidationError:
                    if self.llm_client.config.strict_availability and attempt > 0:
                        raise LLMUnavailableError("CodePatchAgent returned invalid PatchEditProposal JSON")
                    edit_proposal = None
            if attempt == 0:
                hint = "Previous response invalid or empty; output one JSON object only."
                if feedback:
                    payload["feedback"] = f"{feedback}\n{hint}"
                else:
                    payload["feedback"] = hint
                continue
            if self.llm_client.config.strict_availability:
                raise LLMUnavailableError("CodePatchAgent returned empty response after retry")
            return None
        if edit_proposal is None:
            if self.llm_client.config.strict_availability:
                raise LLMUnavailableError("CodePatchAgent failed to produce a valid edit proposal")
            return None
        patch_proposal = PatchProposal(
            status=edit_proposal.status,
            patch_diff="",
            touched_files=edit_proposal.touched_files,
            rationale=edit_proposal.rationale,
            assumptions=edit_proposal.assumptions,
            confidence=edit_proposal.confidence,
            missing_fields=edit_proposal.missing_fields,
        )
        if edit_proposal.status != "OK":
            return patch_proposal
        # Anchor validation: for complex transforms (replace), require longer anchor.
        # For simple inserts (neighbor_prefetch), a single unique line is often enough.
        short_anchor = False
        for edit in edit_proposal.edits:
            if edit.op not in ("insert_before", "insert_after", "replace"):
                continue
            anchor = (edit.anchor or "").strip()
            if not anchor:
                continue
            lines = [line for line in anchor.splitlines() if line.strip()]
            total_len = sum(len(line.strip()) for line in lines)
            # For replace ops, require more context; for inserts, single line may be ok
            if edit.op == "replace":
                if len(lines) < 2 or total_len < 40:
                    short_anchor = True
                    break
            else:  # insert_before/insert_after
                if len(lines) < 1 or total_len < 20:
                    short_anchor = True
                    break
        if short_anchor:
            # Do not fail fast on short anchors. Try applying edits first and rely on
            # structured-edit disambiguation/retry to recover concrete anchors from
            # current source. This avoids dropping otherwise valid patches too early.
            assumptions = list(patch_proposal.assumptions or [])
            assumptions.append(
                "anchor_was_short; will attempt apply/disambiguation before requesting more context"
            )
            patch_proposal.assumptions = assumptions
        patch_family = None
        if action.parameters:
            patch_family = action.parameters.get("patch_family")
        # App-specific structural validation (delegated to plugin)
        if app:
            validation_result = app_validate_patch_edits(
                app,
                edit_proposal,
                patch_family=patch_family,
                uses_dbl3=uses_dbl3,
                code_snippets=code_snippets,
            )
            if validation_result:
                patch_proposal.status = validation_result["status"]
                patch_proposal.missing_fields = validation_result["missing_fields"]
                return patch_proposal
        try:
            result = apply_structured_edits(repo_root, edit_proposal.edits, allowed_files)
        except StructuredEditError as exc:
            message = str(exc)
            adjusted = try_disambiguate_edits(
                edit_proposal.edits, code_snippets, repo_root, message
            )
            if adjusted:
                try:
                    result = apply_structured_edits(
                        repo_root, edit_proposal.edits, allowed_files
                    )
                except StructuredEditError as exc2:
                    patch_proposal.status = "NEED_MORE_CONTEXT"
                    patch_proposal.missing_fields = [f"edit_apply_failed: {exc2}"]
                    return patch_proposal
            else:
                patch_proposal.status = "NEED_MORE_CONTEXT"
                patch_proposal.missing_fields = [f"edit_apply_failed: {message}"]
                return patch_proposal
        patch_proposal.patch_diff = result.patch_diff
        patch_proposal.touched_files = result.touched_files

        # AST post-validation: check structural correctness
        try:
            from skills.code_structure import is_available as _ts_avail, validate_patch_structure
            if _ts_avail():
                ast_warnings: List[str] = []
                for tf in result.touched_files:
                    full_path = repo_root / tf
                    if not full_path.is_file():
                        continue
                    patched_text = full_path.read_text(encoding="utf-8", errors="replace")
                    validation = validate_patch_structure(str(full_path), patched_text)
                    if not validation.valid:
                        ast_warnings.extend(
                            [f"[AST] {tf}: {e}" for e in validation.errors]
                        )
                    ast_warnings.extend(
                        [f"[AST warning] {tf}: {w}" for w in validation.warnings]
                    )
                if ast_warnings:
                    patch_proposal.rationale = (
                        (patch_proposal.rationale or "")
                        + "\n\nAST validation notes:\n"
                        + "\n".join(ast_warnings[:5])
                    )
        except Exception:
            pass  # AST validation is best-effort

        return patch_proposal


def _load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "agents" / f"{name}.md"
    return path.read_text(encoding="utf-8")


def try_disambiguate_edits(
    edits: List[PatchEdit],
    code_snippets: List[Dict[str, object]],
    repo_root: Path,
    message: str,
) -> bool:
    match = re.search(r"^(anchor|old_text)_not_unique:([^:]+):b64:(.+)$", message)
    if not match:
        match = re.search(r"^(anchor|old_text)_not_found:([^:]+):b64:(.+)$", message)
    if not match:
        return False
    label, path, b64 = match.group(1), match.group(2), match.group(3)
    try:
        anchor = base64.b64decode(b64).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False
    file_path = repo_root / path
    if not file_path.exists():
        return False
    lines = file_path.read_text(encoding="utf-8").splitlines()
    snippet_ranges = []
    for snippet in code_snippets or []:
        if snippet.get("path") != path:
            continue
        start = snippet.get("start_line")
        end = snippet.get("end_line")
        if isinstance(start, int) and isinstance(end, int):
            snippet_ranges.append((start - 1, end - 1))
    anchor_lines = [line for line in anchor.splitlines() if line.strip()]
    if not anchor_lines:
        return False
    matches = []
    for i in range(len(lines) - len(anchor_lines) + 1):
        if lines[i : i + len(anchor_lines)] == anchor_lines:
            matches.append(i)
    if not matches:
        trimmed_anchor_lines = [line.rstrip() for line in anchor_lines]
        for i in range(len(lines) - len(trimmed_anchor_lines) + 1):
            candidate = [line.rstrip() for line in lines[i : i + len(trimmed_anchor_lines)]]
            if candidate == trimmed_anchor_lines:
                matches.append(i)
    if not matches and label == "old_text":
        block_len = max(1, len(anchor_lines))
        needle = "\n".join(anchor_lines)
        best_idx = None
        best_score = 0.0
        for i in range(len(lines) - block_len + 1):
            candidate_text = "\n".join(lines[i : i + block_len])
            score = difflib.SequenceMatcher(None, needle, candidate_text).ratio()
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is not None and best_score >= 0.88:
            matches.append(best_idx)
    # Final fallback for old_text_not_found:
    # use the edit's anchor to localize search and recover a near-by block.
    if not matches and label == "old_text":
        recovered = _recover_old_text_by_anchor(
            edits=edits,
            path=path,
            old_text=anchor,
            file_lines=lines,
        )
        if recovered:
            return True
    if not matches:
        return False
    preferred = None
    if snippet_ranges:
        def _distance_to_ranges(idx: int) -> int:
            best = 10**9
            for start, end in snippet_ranges:
                if start <= idx <= end:
                    return 0
                if idx < start:
                    best = min(best, start - idx)
                else:
                    best = min(best, idx - end)
            return best
        preferred = min(matches, key=_distance_to_ranges)
    if preferred is None:
        preferred = matches[0]
    full_text = "\n".join(lines)
    for extra in range(1, 41):
        start = max(0, preferred - extra)
        end = min(len(lines), preferred + len(anchor_lines) + extra)
        candidate = "\n".join(lines[start:end])
        if len(re.findall(re.escape(candidate), full_text)) == 1:
            for edit in edits:
                if label == "anchor" and edit.anchor == anchor:
                    edit.anchor = candidate
                if label == "old_text" and edit.old_text == anchor:
                    edit.old_text = candidate
                    if edit.anchor and edit.anchor not in candidate:
                        first_line = next(
                            (line for line in candidate.splitlines() if line.strip()),
                            "",
                        )
                        if first_line:
                            edit.anchor = first_line
            return True
    return False


def _recover_old_text_by_anchor(
    edits: List[PatchEdit],
    path: str,
    old_text: str,
    file_lines: List[str],
) -> bool:
    old_lines = [line for line in old_text.splitlines() if line.strip()]
    if not old_lines:
        return False
    old_norm = "\n".join(line.strip() for line in old_lines)
    old_len = len(old_lines)
    text = "\n".join(file_lines)

    def _find_line_sequence(seq_lines: List[str]) -> List[int]:
        if not seq_lines:
            return []
        stripped = [line.rstrip() for line in file_lines]
        needle = [line.rstrip() for line in seq_lines]
        hits: List[int] = []
        for i in range(len(stripped) - len(needle) + 1):
            if stripped[i : i + len(needle)] == needle:
                hits.append(i)
        return hits

    for edit in edits:
        if edit.file != path or edit.old_text != old_text:
            continue
        anchor_text = (edit.anchor or "").strip()
        if not anchor_text:
            continue
        anchor_lines = [line for line in anchor_text.splitlines() if line.strip()]
        if not anchor_lines:
            continue
        anchor_hits = _find_line_sequence(anchor_lines)
        if len(anchor_hits) != 1:
            continue
        anchor_start = anchor_hits[0]
        # Search windows near the anchor with slightly variable span.
        best_block = ""
        best_score = 0.0
        for delta in range(-4, 7):
            span = max(1, old_len + delta)
            for shift in range(-6, 7):
                start = max(0, anchor_start + shift)
                end = min(len(file_lines), start + span)
                if end <= start:
                    continue
                block_lines = file_lines[start:end]
                block = "\n".join(block_lines)
                norm = "\n".join(line.strip() for line in block_lines if line.strip())
                if not norm:
                    continue
                score = difflib.SequenceMatcher(None, old_norm, norm).ratio()
                if score > best_score:
                    best_score = score
                    best_block = block
        # Conservative threshold: enough to recover minor drift without
        # applying unrelated edits.
        if best_block and best_score >= 0.62:
            if len(re.findall(re.escape(best_block), text)) == 1:
                edit.old_text = best_block
                if edit.anchor and edit.anchor not in best_block:
                    first_line = next(
                        (line for line in best_block.splitlines() if line.strip()),
                        "",
                    )
                    if first_line:
                        edit.anchor = first_line
                return True
    return False
