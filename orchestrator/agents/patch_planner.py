"""PatchPlannerAgent -- replaces the CodeSurvey -> Idea -> ActionSynth chain.

One LLM call that takes (profile + code_snippets + experience) and directly
produces a ranked list of concrete patch actions with embedded code context.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import ValidationError

from orchestrator.errors import LLMUnavailableError
from orchestrator.llm_client import LLMClient
from schemas.action_ir import ActionIR, VerificationPlan
from schemas.opportunity_graph import OpportunityMechanism, SelectedOpportunities
from schemas.patch_plan_ir import PatchPlan
from schemas.profile_report import ProfileReport
from skills.profile_payload import build_profile_payload


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
            "profile": build_profile_payload(profile),
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
            if self.llm_client.config.strict_availability:
                raise LLMUnavailableError("PatchPlannerAgent returned empty response")
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
                if self.llm_client.config.strict_availability:
                    raise LLMUnavailableError("PatchPlannerAgent retry returned empty response")
                return None
            try:
                result = PatchPlan(**data)
            except ValidationError:
                if self.llm_client.config.strict_availability:
                    raise LLMUnavailableError("PatchPlannerAgent returned invalid PatchPlan JSON")
                return None
        if result.status != "OK" and self.llm_client.config.strict_availability:
            raise LLMUnavailableError(f"PatchPlannerAgent returned non-OK status: {result.status}")
        return result

    def plan_from_opportunity_selection(
        self,
        selection: SelectedOpportunities,
        patch_families: Optional[Dict[str, object]] = None,
        existing_action_ids: Optional[List[str]] = None,
    ) -> List[ActionIR]:
        existing = set(existing_action_ids or [])
        actions: List[ActionIR] = []
        for item in selection.selected:
            node = item.opportunity
            patch_family = self._resolve_family(
                mechanism=node.mechanism,
                family_hint=node.family_hint,
                patch_families=patch_families or {},
            )
            action_id = f"graph.source_patch.{node.mechanism.value}.{node.opportunity_id}"
            if action_id in existing:
                action_id = f"{action_id}.alt"
            expected = self._expected_effects(node.mechanism)
            action = ActionIR(
                action_id=action_id,
                family="source_patch",
                description=f"{node.title} [{node.mechanism.value}]",
                applies_to=["source_patch"],
                parameters={
                    "patch_family": patch_family,
                    "graph_mechanism": node.mechanism.value,
                    "origin": "opportunity_graph",
                    "opportunity_id": node.opportunity_id,
                    "deep_analysis_id": node.opportunity_id,
                    "target_file": node.hotspot.file,
                    "target_anchor": node.hotspot.function,
                    "target_line_range": node.hotspot.line_range,
                    "target_files": list(node.target_files or [node.hotspot.file]),
                    "target_functions": list(node.target_functions or [node.hotspot.function]),
                    "hypothesis": node.hypothesis,
                    "evidence_ids": list(node.evidence_ids),
                    "validation_plan": node.validation_plan.model_dump(),
                    "expected_gain_p50": node.expected_gain.p50,
                    "expected_gain_p90": node.expected_gain.p90,
                    "success_prob": node.success_prob,
                    "implementation_cost": node.implementation_cost,
                    "composability": node.composability.score,
                    "depends_on": list(node.composability.depends_on),
                    "conflicts_with": list(node.composability.conflicts_with),
                },
                expected_effect=expected,
                risk_level="medium",
                verification_plan=VerificationPlan(gates=["runtime", "correctness", "variance"]),
            )
            actions.append(action)
        self.last_trace = {
            "selection": selection.model_dump(),
            "actions": [a.model_dump() for a in actions],
        }
        return actions

    def _resolve_family(
        self,
        mechanism: OpportunityMechanism,
        family_hint: str,
        patch_families: Dict[str, object],
    ) -> str:
        known = {
            str(item.get("id"))
            for item in (patch_families or {}).get("families", [])
            if isinstance(item, dict) and item.get("id")
        }
        if family_hint and family_hint in known:
            return family_hint
        return f"source_patch:{mechanism.value}"

    def _expected_effects(self, mechanism: OpportunityMechanism) -> List[str]:
        if mechanism in {OpportunityMechanism.DATA_LAYOUT, OpportunityMechanism.MEMORY_PATH, OpportunityMechanism.ALLOCATION}:
            return ["mem_locality"]
        if mechanism in {OpportunityMechanism.SYNC}:
            return ["imbalance_reduce"]
        if mechanism in {OpportunityMechanism.IO}:
            return ["io_reduce"]
        return ["compute_opt"]


def _load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "agents" / f"{name}.md"
    return path.read_text(encoding="utf-8")
