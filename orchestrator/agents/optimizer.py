from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from schemas.action_ir import ActionIR
from schemas.candidate_ir import CandidateList
from schemas.plan_ir import PlanIR
from schemas.profile_report import ProfileReport
from orchestrator.router import RuleContext, filter_actions
from orchestrator.llm_client import LLMClient


class OptimizerAgent:
    def __init__(self, llm_client: Optional[LLMClient]) -> None:
        self.llm_client = llm_client
        self.last_llm_trace: Optional[Dict[str, object]] = None
        self.last_ranking_mode: str = "heuristic"

    def propose(
        self,
        actions: List[ActionIR],
        ctx: RuleContext,
        plan: PlanIR,
        policy: Dict[str, object],
        profile: ProfileReport,
        exclude_action_ids: List[str],
    ) -> List[CandidateList]:
        candidate_lists: List[CandidateList] = []
        family_actions: Dict[str, List[ActionIR]] = {}
        for family in plan.chosen_families:
            family_pool = [a for a in actions if a.family == family]
            filtered = filter_actions(family_pool, ctx, [family], policy)
            filtered = [a for a in filtered if a.action_id not in exclude_action_ids]
            if filtered:
                family_actions[family] = filtered

        if family_actions and self.llm_client and self.llm_client.config.enabled:
            llm_lists = self._try_llm(plan, policy, profile, family_actions, exclude_action_ids)
            if llm_lists:
                return llm_lists

        for family, filtered in family_actions.items():
            candidates = filtered[: plan.max_candidates]
            candidate_lists.append(
                CandidateList(
                    family=family,
                    candidates=candidates,
                    assumptions=[],
                    confidence=0.6,
                )
            )
        return candidate_lists

    def _try_llm(
        self,
        plan: PlanIR,
        policy: Dict[str, object],
        profile: ProfileReport,
        family_actions: Dict[str, List[ActionIR]],
        exclude_action_ids: List[str],
    ) -> Optional[List[CandidateList]]:
        if not self.llm_client or not self.llm_client.config.enabled:
            return None
        prompt = _load_prompt("optimizer")
        payload = {
            "plan": plan.model_dump(),
            "profile": {
                "timing_breakdown": profile.timing_breakdown,
                "system_metrics": profile.system_metrics,
            },
            "policy": policy,
            "exclude_action_ids": exclude_action_ids,
            "actions_by_family": {
                family: [_action_payload(a) for a in actions]
                for family, actions in family_actions.items()
            },
        }
        data = self.llm_client.request_json(prompt, payload)
        self.last_llm_trace = {"payload": payload, "response": data}
        if not data:
            return None
        candidate_lists = _build_llm_candidates(
            data, plan, family_actions, exclude_action_ids
        )
        return candidate_lists or None


def _action_payload(action: ActionIR) -> Dict[str, object]:
    return {
        "action_id": action.action_id,
        "family": action.family,
        "description": action.description,
        "applies_to": action.applies_to,
        "parameters": action.parameters,
        "preconditions": action.preconditions,
        "constraints": action.constraints,
        "expected_effect": action.expected_effect,
        "risk_level": action.risk_level,
    }


def _build_llm_candidates(
    data: Dict[str, object],
    plan: PlanIR,
    family_actions: Dict[str, List[ActionIR]],
    exclude_action_ids: List[str],
) -> List[CandidateList]:
    raw_candidates = data.get("candidates")
    if not isinstance(raw_candidates, list):
        return []
    actions_by_id: Dict[str, ActionIR] = {}
    for actions in family_actions.values():
        for action in actions:
            actions_by_id[action.action_id] = action
    candidate_lists: List[CandidateList] = []
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        family = str(item.get("family", "")).strip()
        if family not in family_actions:
            continue
        action_ids = item.get("action_ids", [])
        if not isinstance(action_ids, list):
            continue
        assumptions = item.get("assumptions", [])
        if not isinstance(assumptions, list):
            assumptions = []
        confidence = item.get("confidence", 0.6)
        try:
            confidence_val = float(confidence)
        except (TypeError, ValueError):
            confidence_val = 0.6
        candidates: List[ActionIR] = []
        for action_id in action_ids:
            action = actions_by_id.get(str(action_id))
            if not action:
                continue
            if action.action_id in exclude_action_ids:
                continue
            if action.family != family:
                continue
            if action in candidates:
                continue
            candidates.append(action)
            if len(candidates) >= plan.max_candidates:
                break
        if candidates:
            candidate_lists.append(
                CandidateList(
                    family=family,
                    candidates=candidates,
                    assumptions=[str(a) for a in assumptions],
                    confidence=confidence_val,
                )
            )
    return candidate_lists


def _load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "agents" / f"{name}.md"
    return path.read_text(encoding="utf-8")
