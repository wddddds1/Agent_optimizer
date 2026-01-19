from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import ValidationError

from schemas.action_ir import ActionIR
from schemas.candidate_ir import CandidateList
from schemas.ranking_ir import RankedAction, RankedActions, Rejection
from schemas.profile_report import ProfileReport
from schemas.ranker_output_ir import RankerOutput
from orchestrator.llm_client import LLMClient
from orchestrator.router import RuleContext, filter_actions


class RouterRankerAgent:
    def __init__(self, llm_client: Optional[LLMClient]) -> None:
        self.llm_client = llm_client

    def rank(
        self,
        candidate_lists: List[CandidateList],
        ctx: RuleContext,
        policy: Dict[str, object],
        profile: ProfileReport,
    ) -> RankedActions:
        actions = _flatten_candidates(candidate_lists)
        filtered = filter_actions(actions, ctx, None, policy)
        rejected = [a for a in actions if a not in filtered]
        if not filtered:
            return RankedActions(
                ranked=[],
                rejected=[Rejection(action_id=a.action_id, reason="policy_filtered") for a in rejected],
                scoring_notes="no candidates after policy filtering",
            )
        ranked_actions, notes, llm_rejections = _rank_actions(filtered, profile, self.llm_client)
        rejection_map: Dict[str, Rejection] = {
            rejection.action_id: rejection for rejection in llm_rejections
        }
        for action in rejected:
            if action.action_id not in rejection_map:
                rejection_map[action.action_id] = Rejection(
                    action_id=action.action_id, reason="policy_filtered"
                )
        return RankedActions(
            ranked=ranked_actions,
            rejected=list(rejection_map.values()),
            scoring_notes=notes,
        )


def _flatten_candidates(candidate_lists: List[CandidateList]) -> List[ActionIR]:
    actions: List[ActionIR] = []
    for candidate_list in candidate_lists:
        actions.extend(candidate_list.candidates)
    return actions


def _rank_actions(
    actions: List[ActionIR],
    profile: ProfileReport,
    llm_client: Optional[LLMClient],
) -> Tuple[List[RankedAction], str, List[Rejection]]:
    if llm_client and llm_client.config.enabled:
        prompt = _load_prompt("ranker")
        payload = {
            "profile": {
                "timing_breakdown": profile.timing_breakdown,
                "system_metrics": profile.system_metrics,
            },
            "actions": [_action_payload(action) for action in actions],
        }
        data = llm_client.request_json(prompt, payload)
        if isinstance(data, dict):
            try:
                output = RankerOutput(**data)
            except ValidationError:
                output = None
            if output and output.status == "OK" and output.ranked_action_ids:
                rejected = [Rejection(action_id=item.action_id, reason=item.reason) for item in output.rejected]
                rejected_ids = {rej.action_id for rej in rejected}
                order = {str(action_id): idx for idx, action_id in enumerate(output.ranked_action_ids)}
                ordered = sorted(
                    actions,
                    key=lambda a: (order.get(a.action_id, len(order)), a.action_id),
                )
                ranked: List[RankedAction] = []
                for action in ordered:
                    if action.action_id in rejected_ids:
                        continue
                    score = float(len(order) - order.get(action.action_id, 0))
                    breakdown = output.score_breakdown.get(action.action_id, {"llm_rank": score})
                    ranked.append(
                        RankedAction(
                            action=action,
                            score=score,
                            score_breakdown=breakdown,
                        )
                    )
                notes = output.scoring_notes or "llm ranking"
                return ranked, notes, rejected
    scored = [(_score_action(action, profile), action) for action in actions]
    scored.sort(key=lambda item: (-item[0][0], item[1].action_id))
    ranked = [
        RankedAction(action=action, score=score, score_breakdown=breakdown)
        for (score, breakdown), action in scored
    ]
    return ranked, "heuristic ranking", []


def _action_payload(action: ActionIR) -> Dict[str, object]:
    return {
        "action_id": action.action_id,
        "family": action.family,
        "description": action.description,
        "expected_effect": action.expected_effect,
        "risk_level": action.risk_level,
        "preconditions": action.preconditions,
        "constraints": action.constraints,
    }


def _parse_rejections(raw: object) -> List[Rejection]:
    if not isinstance(raw, list):
        return []
    rejections: List[Rejection] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        action_id = str(item.get("action_id", "")).strip()
        if not action_id:
            continue
        reason = str(item.get("reason", "llm_reject")).strip() or "llm_reject"
        rejections.append(Rejection(action_id=action_id, reason=reason))
    return rejections


def _load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "agents" / f"{name}.md"
    return path.read_text(encoding="utf-8")


def _score_action(action: ActionIR, profile: ProfileReport) -> Tuple[float, Dict[str, float]]:
    timing = profile.timing_breakdown
    total = timing.get("total", 0.0) or 0.0
    comm_ratio = (timing.get("comm", 0.0) / total) if total else 0.0
    output_ratio = (timing.get("output", 0.0) / total) if total else 0.0
    cpu = profile.system_metrics.get("cpu_percent_avg", 100.0)
    score = 0.0
    breakdown: Dict[str, float] = {}
    if comm_ratio > 0.2 and "comm_reduce" in action.expected_effect:
        score += 2.0
        breakdown["comm_match"] = 2.0
    if output_ratio > 0.2 and "io_reduce" in action.expected_effect:
        score += 2.0
        breakdown["io_match"] = 2.0
    if cpu < 70.0 and (
        "compute_opt" in action.expected_effect or "mem_locality" in action.expected_effect
    ):
        score += 1.0
        breakdown["cpu_gap"] = 1.0
    if action.risk_level == "low":
        score += 0.5
        breakdown["risk_bonus"] = 0.5
    elif action.risk_level == "high":
        score -= 0.5
        breakdown["risk_penalty"] = -0.5
    return score, breakdown
