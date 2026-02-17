from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import ValidationError

from orchestrator.llm_client import LLMClient
from schemas.action_ir import ActionIR
from schemas.patch_edit_ir import PatchEditProposal
from schemas.patch_proposal_ir import PatchProposal
from schemas.profile_report import ProfileReport
from skills.patch_edit import StructuredEditError, apply_structured_edits
from skills.profile_payload import build_profile_payload


class PatchDebugAgent:
    def __init__(self, llm_client: Optional[LLMClient]) -> None:
        self.llm_client = llm_client
        self.last_trace: Optional[Dict[str, object]] = None

    def repair(
        self,
        action: ActionIR,
        profile: ProfileReport,
        patch_rules: Dict[str, object],
        allowed_files: List[str],
        code_snippets: List[Dict[str, object]],
        repo_root: Path,
        patch_diff: str,
        build_log: str,
        feedback: Optional[str] = None,
    ) -> Optional[PatchProposal]:
        if not self.llm_client or not self.llm_client.config.enabled:
            return None
        prompt = _load_prompt("patch_debug")
        payload = {
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
            "patch_diff": patch_diff,
            "build_log": build_log,
            "feedback": feedback,
        }
        data = self.llm_client.request_json(prompt, payload)
        self.last_trace = {"payload": payload, "response": data}
        if data is None:
            return PatchProposal(
                status="NEED_MORE_CONTEXT",
                patch_diff="",
                touched_files=[],
                rationale="PatchDebugAgent returned no JSON object.",
                assumptions=[],
                confidence=0.0,
                missing_fields=["debug_agent_no_json"],
            )
        if not data:
            return PatchProposal(
                status="NEED_MORE_CONTEXT",
                patch_diff="",
                touched_files=[],
                rationale="PatchDebugAgent returned an empty JSON object.",
                assumptions=[],
                confidence=0.0,
                missing_fields=["debug_agent_empty_json"],
            )
        try:
            edit_proposal = PatchEditProposal(**data)
        except ValidationError as exc:
            first_error = exc.errors()[0] if exc.errors() else {}
            loc = ".".join(str(part) for part in first_error.get("loc", [])) or "unknown"
            msg = str(first_error.get("msg") or "validation_error")
            return PatchProposal(
                status="NEED_MORE_CONTEXT",
                patch_diff="",
                touched_files=[],
                rationale="PatchDebugAgent response failed schema validation.",
                assumptions=[],
                confidence=0.0,
                missing_fields=[f"debug_agent_schema_error:{loc}:{msg}"],
            )
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
        try:
            result = apply_structured_edits(repo_root, edit_proposal.edits, allowed_files)
        except StructuredEditError as exc:
            patch_proposal.status = "NEED_MORE_CONTEXT"
            patch_proposal.missing_fields = [f"edit_apply_failed: {exc}"]
            return patch_proposal
        patch_proposal.patch_diff = result.patch_diff
        patch_proposal.touched_files = result.touched_files
        return patch_proposal


def _load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "agents" / f"{name}.md"
    return path.read_text(encoding="utf-8")
