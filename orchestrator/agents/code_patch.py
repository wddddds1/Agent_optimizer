from __future__ import annotations

from pathlib import Path
import base64
import re
from typing import Dict, List, Optional

from pydantic import ValidationError

from orchestrator.llm_client import LLMClient
from schemas.action_ir import ActionIR
from schemas.patch_edit_ir import PatchEdit, PatchEditProposal
from schemas.patch_proposal_ir import PatchProposal
from schemas.profile_report import ProfileReport
from skills.patch_edit import StructuredEditError, apply_structured_edits


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
            "profile": {
                "timing_breakdown": profile.timing_breakdown,
                "system_metrics": profile.system_metrics,
                "notes": profile.notes,
            },
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
                    edit_proposal = None
            if attempt == 0:
                hint = "Previous response invalid or empty; output one JSON object only."
                if feedback:
                    payload["feedback"] = f"{feedback}\n{hint}"
                else:
                    payload["feedback"] = hint
                continue
            return None
        if edit_proposal is None:
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
            patch_proposal.status = "NEED_MORE_CONTEXT"
            patch_proposal.missing_fields = [
                "anchor too short/ambiguous; include 2-3 exact lines around the target "
                "to make it unique in the file"
            ]
            return patch_proposal
        patch_family = None
        if action.parameters:
            patch_family = action.parameters.get("patch_family")
        if patch_family == "param_table_pack":
            added_text = "\n".join(edit.new_text or "" for edit in edit_proposal.edits)
            has_malloc = "malloc" in added_text or "free" in added_text
            has_include = "#include <cstdlib>" in added_text or "#include <stdlib.h>" in added_text
            if has_malloc and not has_include:
                include_anchor = None
                include_file = None
                for snippet in code_snippets or []:
                    snippet_text = snippet.get("snippet") or ""
                    for line in snippet_text.splitlines():
                        if line.strip().startswith("#include"):
                            include_anchor = line
                            include_file = snippet.get("path")
                            break
                    if include_anchor:
                        break
                if include_anchor and include_file:
                    edit_proposal.edits.insert(
                        0,
                        PatchEdit(
                            file=include_file,
                            op="insert_before",
                            anchor=include_anchor,
                            new_text="#include <cstdlib>\n",
                        ),
                    )
                    added_text = "#include <cstdlib>\n" + added_text
                    has_include = True
            if has_malloc and not has_include:
                patch_proposal.status = "NEED_MORE_CONTEXT"
                patch_proposal.missing_fields = [
                    "param_table_pack requires adding `#include <cstdlib>` "
                    "and using std::malloc/std::free"
                ]
                return patch_proposal
            if has_malloc and "std::malloc" not in added_text and "std::free" not in added_text:
                patch_proposal.status = "NEED_MORE_CONTEXT"
                patch_proposal.missing_fields = [
                    "param_table_pack must use std::malloc/std::free (not raw malloc/free) with <cstdlib>"
                ]
                return patch_proposal
            # Ensure allocation happens BEFORE the outer loop, not inside it.
            needs_outer_anchor = any(
                "tabsix" in (edit.new_text or "") or "fast_alpha_t" in (edit.new_text or "")
                for edit in edit_proposal.edits
            )
            if needs_outer_anchor:
                anchor_ctx = "\n".join(
                    (edit.anchor or "") + "\n" + (edit.old_text or "")
                    for edit in edit_proposal.edits
                )
                if "for (int ii" not in anchor_ctx and "for (ii" not in anchor_ctx:
                    patch_proposal.status = "NEED_MORE_CONTEXT"
                    patch_proposal.missing_fields = [
                        "param_table_pack must insert allocation BEFORE the outer `for (ii ...)` loop; "
                        "use an anchor that includes the loop header"
                    ]
                    return patch_proposal
            if "free(" in added_text:
                anchor_ctx = "\n".join(
                    (edit.anchor or "") + "\n" + (edit.old_text or "")
                    for edit in edit_proposal.edits
                )
                if "f[i].z" not in anchor_ctx and "fztmp" not in anchor_ctx:
                    patch_proposal.status = "NEED_MORE_CONTEXT"
                    patch_proposal.missing_fields = [
                        "param_table_pack must place free(tabsix) AFTER the outer loop; "
                        "anchor near `f[i].z += fztmp;`"
                    ]
                    return patch_proposal
        if patch_family == "special_pair_split":
            has_replace_block = any(
                edit.op == "replace"
                and edit.old_text
                and "factor_lj = special_lj" in edit.old_text
                for edit in edit_proposal.edits
            )
            if not has_replace_block:
                patch_proposal.status = "NEED_MORE_CONTEXT"
                patch_proposal.missing_fields = [
                    "special_pair_split must REPLACE the original block starting at "
                    "`factor_lj = special_lj[sbmask(j)];` to avoid duplicate loops"
                ]
                return patch_proposal
            for edit in edit_proposal.edits:
                if edit.op != "replace" or not edit.old_text:
                    continue
                if "factor_lj = special_lj" in edit.old_text:
                    if "delx = xtmp - x[j].x" not in edit.old_text or "if (rsq <" not in edit.old_text:
                        patch_proposal.status = "NEED_MORE_CONTEXT"
                        patch_proposal.missing_fields = [
                            "special_pair_split replacement must include the full original inner-loop body "
                            "(delx/dely/delz, rsq, and the rsq<cutsq block) to avoid leaving duplicate code"
                        ]
                        return patch_proposal
            added_text = "\n".join(edit.new_text or "" for edit in edit_proposal.edits)
            if "x[j][" in added_text or "x[i][" in added_text:
                patch_proposal.status = "NEED_MORE_CONTEXT"
                patch_proposal.missing_fields = [
                    "special_pair_split must use dbl3_t access (x[j].x/x[j].y/x[j].z), "
                    "not x[j][0] indexing"
                ]
                return patch_proposal
        if patch_family in {"cache_local_pointers", "cache_local_pointers_multi"}:
            # Generic structural check: patch must both declare a cached
            # local variable AND replace at least one original access.
            has_cache_decl = any(
                re.search(
                    r"(?:const\s+)?(?:auto|double|int|float)\s*[*&]?\s*\w+\s*=",
                    edit.new_text or "",
                )
                for edit in edit_proposal.edits
                if edit.op in ("insert_before", "insert_after")
            )
            has_use_replacement = any(
                edit.op == "replace" and edit.old_text and edit.new_text
                and edit.new_text != edit.old_text
                for edit in edit_proposal.edits
            )
            if not has_cache_decl or not has_use_replacement:
                patch_proposal.status = "NEED_MORE_CONTEXT"
                patch_proposal.missing_fields = [
                    "cache_local_pointers requires both a local cache declaration "
                    "AND replacement of original array accesses"
                ]
                return patch_proposal
            # dbl3_t backend check: reject `double *xi = x[i]` pattern
            # (should use `const auto &xi = x[i]` for dbl3_t arrays)
            if uses_dbl3:
                added_text = "\n".join(
                    edit.new_text or "" for edit in edit_proposal.edits
                )
                if re.search(
                    r"\bconst?\s+double\s*\*+\s*\w+\s*=\s*\w+\[",
                    added_text,
                ):
                    patch_proposal.status = "NEED_MORE_CONTEXT"
                    patch_proposal.missing_fields = [
                        "OMP backend uses dbl3_t; cache arrays as refs "
                        "(e.g., `const auto &xj = x[j];`), not double*"
                    ]
                    return patch_proposal
        try:
            result = apply_structured_edits(repo_root, edit_proposal.edits, allowed_files)
        except StructuredEditError as exc:
            message = str(exc)
            adjusted = _try_disambiguate_edits(
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
        return patch_proposal


def _load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "agents" / f"{name}.md"
    return path.read_text(encoding="utf-8")


def _try_disambiguate_edits(
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
        return False
    preferred = None
    if snippet_ranges:
        for idx in matches:
            if any(start <= idx <= end for start, end in snippet_ranges):
                preferred = idx
                break
    if preferred is None:
        preferred = matches[0]
    full_text = "\n".join(lines)
    for extra in range(1, 5):
        start = max(0, preferred - extra)
        end = min(len(lines), preferred + len(anchor_lines) + extra)
        candidate = "\n".join(lines[start:end])
        if len(re.findall(re.escape(candidate), full_text)) == 1:
            for edit in edits:
                if label == "anchor" and edit.anchor == anchor:
                    edit.anchor = candidate
                if label == "old_text" and edit.old_text == anchor:
                    edit.old_text = candidate
            return True
    return False
