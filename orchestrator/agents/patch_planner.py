"""PatchPlannerAgent -- replaces the CodeSurvey -> Idea -> ActionSynth chain.

One LLM call that takes (profile + code_snippets + experience) and directly
produces a ranked list of concrete patch actions with embedded code context.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import ValidationError

from orchestrator.llm_client import LLMClient
from schemas.patch_plan_ir import PatchPlan
from schemas.profile_report import ProfileReport


class PatchPlannerAgent:
    def __init__(self, llm_client: Optional[LLMClient]) -> None:
        self.llm_client = llm_client
        self.last_trace: Optional[Dict[str, object]] = None

    def plan(
        self,
        profile: ProfileReport,
        code_snippets: List[Dict[str, object]],
        patch_families: Dict[str, object],
        allowed_files: List[str],
        experience_hints: List[Dict[str, object]],
        backend_variant: Optional[str] = None,
        max_actions: int = 5,
        existing_action_ids: Optional[List[str]] = None,
    ) -> Optional[PatchPlan]:
        if not self.llm_client or not self.llm_client.config.enabled:
            return None
        prompt = _load_prompt("patch_planner")
        payload: Dict[str, object] = {
            "profile": {
                "timing_breakdown": profile.timing_breakdown,
                "system_metrics": profile.system_metrics,
                "notes": profile.notes,
            },
            "code_snippets": code_snippets,
            "patch_families": patch_families,
            "allowed_files": allowed_files,
            "experience_hints": experience_hints,
            "backend_variant": backend_variant or "",
            "max_actions": max_actions,
            "existing_action_ids": existing_action_ids or [],
        }
        data = self.llm_client.request_json(prompt, payload)
        self.last_trace = {"payload": payload, "response": data}
        if not data:
            return None
        try:
            result = PatchPlan(**data)
        except ValidationError:
            # Retry once with a format hint.
            payload["feedback"] = (
                "Previous response was invalid JSON. "
                "Output a single JSON object matching the PatchPlan schema."
            )
            data = self.llm_client.request_json(prompt, payload)
            self.last_trace = {"payload": payload, "response": data}
            if not data:
                return None
            try:
                result = PatchPlan(**data)
            except ValidationError:
                return None
        return result


def _load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "agents" / f"{name}.md"
    return path.read_text(encoding="utf-8")
