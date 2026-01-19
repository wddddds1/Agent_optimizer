from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from schemas.action_ir import ActionIR
from pydantic import ValidationError

from schemas.candidate_ir import CandidateList
from schemas.optimizer_output_ir import OptimizerOutput
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
        system_caps: Optional[Dict[str, object]] = None,
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
            llm_lists = self._try_llm(
                plan,
                policy,
                profile,
                family_actions,
                exclude_action_ids,
                system_caps,
                ctx,
            )
            if llm_lists:
                return _dedupe_candidate_lists(llm_lists, plan)

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
        return _dedupe_candidate_lists(candidate_lists, plan)

    def _try_llm(
        self,
        plan: PlanIR,
        policy: Dict[str, object],
        profile: ProfileReport,
        family_actions: Dict[str, List[ActionIR]],
        exclude_action_ids: List[str],
        system_caps: Optional[Dict[str, object]],
        ctx: RuleContext,
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
            "system_caps": system_caps or {},
            "current_env": ctx.job.env or {},
            "actions_by_family": {
                family: [_action_payload(a) for a in actions]
                for family, actions in family_actions.items()
            },
        }
        data = self.llm_client.request_json(prompt, payload)
        self.last_llm_trace = {"payload": payload, "response": data}
        if not data:
            return None
        try:
            output = OptimizerOutput(**data)
        except ValidationError:
            return None
        if output.status != "OK":
            return None
        candidate_lists = _build_llm_candidates(
            output, plan, family_actions, exclude_action_ids
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
    output: OptimizerOutput,
    plan: PlanIR,
    family_actions: Dict[str, List[ActionIR]],
    exclude_action_ids: List[str],
) -> List[CandidateList]:
    actions_by_id: Dict[str, ActionIR] = {}
    for actions in family_actions.values():
        for action in actions:
            actions_by_id[action.action_id] = action
    candidate_lists: List[CandidateList] = []
    for group in output.candidates:
        family = group.family
        if family not in family_actions:
            continue
        action_ids = group.action_ids
        assumptions = group.assumptions
        confidence_val = float(group.confidence)
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


def _dedupe_candidate_lists(candidate_lists: List[CandidateList], plan: PlanIR) -> List[CandidateList]:
    deduped: List[CandidateList] = []
    for candidate_list in candidate_lists:
        seen: set[str] = set()
        candidates: List[ActionIR] = []
        for action in candidate_list.candidates:
            signature = _action_signature(action)
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(action)
            if len(candidates) >= plan.max_candidates:
                break
        if not candidates and candidate_list.candidates:
            candidates = [candidate_list.candidates[0]]
        if candidates:
            deduped.append(
                CandidateList(
                    family=candidate_list.family,
                    candidates=candidates,
                    assumptions=candidate_list.assumptions,
                    confidence=candidate_list.confidence,
                )
            )
    return deduped


def _action_signature(action: ActionIR) -> str:
    params = _normalized_parameters(action.parameters)
    applies_to = ",".join(sorted(action.applies_to))
    return f"{action.family}|{applies_to}|{params}"


def _normalized_parameters(params: Dict[str, object]) -> str:
    try:
        import json
        payload = params or {}
        return json.dumps(payload, sort_keys=True, ensure_ascii=True)
    except Exception:
        return str(params)


def _load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "agents" / f"{name}.md"
    return path.read_text(encoding="utf-8")
