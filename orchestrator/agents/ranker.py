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
        profile_features: Optional[Dict[str, object]] = None,
        hotspot_map: Optional[Dict[str, object]] = None,
        rank_cfg: Optional[Dict[str, object]] = None,
        tested_actions: Optional[List[str]] = None,
        memory_scores: Optional[Dict[str, float]] = None,
        patch_stats: Optional[Dict[str, Dict[str, int]]] = None,
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
        ranked_actions, notes, llm_rejections = _rank_actions(
            filtered,
            profile,
            profile_features or {},
            hotspot_map or {},
            self.llm_client,
            rank_cfg or {},
            tested_actions or [],
            memory_scores or {},
            patch_stats or {},
        )
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
    profile_features: Dict[str, object],
    hotspot_map: Dict[str, object],
    llm_client: Optional[LLMClient],
    rank_cfg: Dict[str, object],
    tested_actions: List[str],
    memory_scores: Dict[str, float],
    patch_stats: Dict[str, Dict[str, int]],
) -> Tuple[List[RankedAction], str, List[Rejection]]:
    weights = (rank_cfg.get("weights") if isinstance(rank_cfg.get("weights"), dict) else {}) or {}
    w_memory = float(weights.get("memory", 1.0))
    w_llm = float(weights.get("llm", 1.0))
    w_heuristic = float(weights.get("heuristic", 1.0))
    w_evidence = float(weights.get("evidence", 1.0))
    novelty_bonus = float(weights.get("novelty_bonus", 0.2))
    risk_penalty_weight = float(weights.get("risk_penalty", 0.3))
    epsilon_explore = float(rank_cfg.get("epsilon_explore", 0.0))
    evidence_thresholds = rank_cfg.get("evidence_thresholds", {}) if isinstance(rank_cfg, dict) else {}

    evidence_index = _build_evidence_index(profile, profile_features, hotspot_map)
    llm_scores: Dict[str, float] = {}
    llm_evidence: Dict[str, List[str]] = {}
    llm_notes: str = ""
    if llm_client and llm_client.config.enabled:
        prompt = _load_prompt("ranker")
        payload = {
            "profile": {
                "timing_breakdown": profile.timing_breakdown,
                "system_metrics": profile.system_metrics,
            },
            "actions": [_action_payload(action) for action in actions],
            "evidence_index": evidence_index,
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
                llm_notes = output.scoring_notes or "llm ranking"
                llm_scores = dict(output.llm_scores or {})
                llm_evidence = dict(output.evidence_refs or {})
                if not llm_scores:
                    for action in actions:
                        llm_scores[action.action_id] = float(len(order) - order.get(action.action_id, 0))
                ranked = _final_rank(
                    actions=actions,
                    profile=profile,
                    profile_features=profile_features,
                    hotspot_map=hotspot_map,
                    tested_actions=tested_actions,
                    memory_scores=memory_scores,
                    w_memory=w_memory,
                    w_llm=w_llm,
                    w_heuristic=w_heuristic,
                    w_evidence=w_evidence,
                    novelty_bonus=novelty_bonus,
                    risk_penalty_weight=risk_penalty_weight,
                    llm_scores=llm_scores,
                    llm_evidence=llm_evidence,
                    evidence_index=evidence_index,
                    evidence_thresholds=evidence_thresholds if isinstance(evidence_thresholds, dict) else {},
                    patch_stats=patch_stats,
                )
                ranked = _apply_exploration(ranked, tested_actions, epsilon_explore)
                return ranked, llm_notes, rejected
    ranked = _final_rank(
        actions=actions,
        profile=profile,
        profile_features=profile_features,
        hotspot_map=hotspot_map,
        tested_actions=tested_actions,
        memory_scores=memory_scores,
        w_memory=w_memory,
        w_llm=w_llm,
        w_heuristic=w_heuristic,
        w_evidence=w_evidence,
        novelty_bonus=novelty_bonus,
        risk_penalty_weight=risk_penalty_weight,
        llm_scores=llm_scores,
        llm_evidence=llm_evidence,
        evidence_index=evidence_index,
        evidence_thresholds=evidence_thresholds if isinstance(evidence_thresholds, dict) else {},
        patch_stats=patch_stats,
    )
    ranked = _apply_exploration(ranked, tested_actions, epsilon_explore)
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


def _build_evidence_index(
    profile: ProfileReport,
    profile_features: Dict[str, object],
    hotspot_map: Dict[str, object],
) -> Dict[str, object]:
    timing = profile.timing_breakdown
    total = timing.get("total", 0.0) or 0.0
    pair_ratio = (timing.get("pair", 0.0) / total) if total else 0.0
    neigh_ratio = (timing.get("neigh", 0.0) / total) if total else 0.0
    comm_ratio = (timing.get("comm", 0.0) / total) if total else 0.0
    output_ratio = (timing.get("output", 0.0) / total) if total else 0.0
    compute_ratio = (timing.get("pair", 0.0) / total) if total else 0.0
    cpu = profile.system_metrics.get("cpu_percent_avg", 0.0)
    features_metrics = profile_features.get("metrics", {}) if isinstance(profile_features, dict) else {}
    hotspot_files = hotspot_map.get("hotspot_files", []) if isinstance(hotspot_map, dict) else []
    return {
        "pair_ratio": pair_ratio,
        "neigh_ratio": neigh_ratio,
        "comm_ratio": comm_ratio,
        "output_ratio": output_ratio,
        "compute_ratio": compute_ratio or float(features_metrics.get("compute_ratio", 0.0) or 0.0),
        "cpu_percent_avg": float(cpu) if cpu is not None else 0.0,
        "hotspot_files": hotspot_files,
    }


def _final_rank(
    actions: List[ActionIR],
    profile: ProfileReport,
    profile_features: Dict[str, object],
    hotspot_map: Dict[str, object],
    tested_actions: List[str],
    memory_scores: Dict[str, float],
    w_memory: float,
    w_llm: float,
    w_heuristic: float,
    w_evidence: float,
    novelty_bonus: float,
    risk_penalty_weight: float,
    llm_scores: Dict[str, float],
    llm_evidence: Dict[str, List[str]],
    evidence_index: Dict[str, object],
    evidence_thresholds: Dict[str, float],
    patch_stats: Dict[str, Dict[str, int]],
) -> List[RankedAction]:
    scored: List[Tuple[float, RankedAction]] = []
    for action in actions:
        heur_score, breakdown = _score_action(action, profile)
        mem_score = float(memory_scores.get(action.action_id, 0.0))
        llm_score = float(llm_scores.get(action.action_id, 0.0))
        evidence_refs = llm_evidence.get(action.action_id, [])
        evidence_ok = 1.0
        if evidence_refs:
            for ref in evidence_refs:
                if ref not in evidence_index:
                    evidence_ok = 0.0
                    llm_score = 0.0
                    break
        evidence_score = _evidence_score(
            action,
            evidence_index,
            evidence_thresholds,
        )
        novelty = novelty_bonus if action.action_id not in tested_actions else 0.0
        risk_penalty = _risk_penalty(action.risk_level, risk_penalty_weight)
        patch_penalty = _patch_penalty(action.action_id, patch_stats)
        final_score = (
            w_memory * mem_score
            + w_llm * llm_score
            + w_heuristic * heur_score
            + w_evidence * evidence_score
            + novelty
            - risk_penalty
            - patch_penalty
        )
        breakdown.update(
            {
                "memory_score": mem_score,
                "llm_score": llm_score,
                "heuristic_score": heur_score,
                "novelty_bonus": novelty,
                "risk_penalty": -risk_penalty,
                "evidence_ok": evidence_ok,
                "evidence_score": evidence_score,
                "patch_penalty": -patch_penalty,
            }
        )
        ranked = RankedAction(action=action, score=final_score, score_breakdown=breakdown)
        scored.append((final_score, ranked))
    scored.sort(key=lambda item: (-item[0], item[1].action.action_id))
    return [item[1] for item in scored]


def _apply_exploration(
    ranked: List[RankedAction],
    tested_actions: List[str],
    epsilon_explore: float,
) -> List[RankedAction]:
    if not ranked or epsilon_explore <= 0:
        return ranked
    explore_slots = max(1, int(round(len(ranked) * epsilon_explore)))
    tested = set(tested_actions)
    novel = [item for item in ranked if item.action.action_id not in tested]
    if not novel:
        return ranked
    novel = novel[:explore_slots]
    remaining = [item for item in ranked if item not in novel]
    return novel + remaining


def _risk_penalty(risk_level: str, weight: float) -> float:
    if risk_level == "low":
        return 0.0
    if risk_level == "high":
        return weight * 2.0
    return weight


def _patch_penalty(action_id: str, patch_stats: Dict[str, Dict[str, int]]) -> float:
    stats = patch_stats.get(action_id, {})
    context_misses = int(stats.get("context_miss", 0) or 0)
    preflight_fails = int(stats.get("preflight_fail", 0) or 0)
    build_fails = int(stats.get("build_fail", 0) or 0)
    penalty = 0.0
    penalty += 0.2 * context_misses
    penalty += 0.6 * preflight_fails
    penalty += 0.8 * build_fails
    return min(penalty, 3.0)


def _evidence_score(
    action: ActionIR,
    evidence_index: Dict[str, object],
    thresholds: Dict[str, float],
) -> float:
    score = 0.5
    action_id = action.action_id.lower()
    params = action.parameters or {}
    patch_family = str(params.get("patch_family", "")).lower()
    target_file = str(params.get("target_file", ""))
    evidence = params.get("evidence", [])

    neigh_ratio = float(evidence_index.get("neigh_ratio", 0.0) or 0.0)
    compute_ratio = float(evidence_index.get("compute_ratio", 0.0) or 0.0)
    comm_ratio = float(evidence_index.get("comm_ratio", 0.0) or 0.0)
    output_ratio = float(evidence_index.get("output_ratio", 0.0) or 0.0)
    hotspot_files = set(evidence_index.get("hotspot_files", []) or [])

    neigh_threshold = float(thresholds.get("neigh_ratio", 0.08) or 0.08)
    comm_threshold = float(thresholds.get("comm_ratio", 0.08) or 0.08)
    output_threshold = float(thresholds.get("output_ratio", 0.08) or 0.08)

    if action.family == "neighbor_tune" or "neighbor_" in action_id:
        if neigh_ratio >= neigh_threshold:
            score += 0.6
        else:
            score -= 0.3

    if action.family == "comm_tune" or "comm_" in action_id:
        if comm_ratio >= comm_threshold:
            score += 0.5
        else:
            score -= 0.2

    if action.family in {"io_tune", "output_tune"} or "output_" in action_id:
        if output_ratio >= output_threshold:
            score += 0.5
        else:
            score -= 0.2

    if action.family == "source_patch":
        target_files_list = params.get("target_files") or []
        origin = str(params.get("origin", ""))

        # Collect all candidate target paths
        candidates = []
        if target_file:
            candidates.append(target_file)
        if isinstance(target_files_list, list):
            candidates.extend(str(tf) for tf in target_files_list if tf)

        # Use suffix matching to handle path prefix mismatches
        # (e.g. "src/OPENMP/foo.cpp" vs "third_party/lammps/src/OPENMP/foo.cpp")
        matched = False
        for cand in candidates:
            for hf in hotspot_files:
                if hf.endswith(cand) or cand.endswith(hf) or hf == cand:
                    matched = True
                    break
            if matched:
                break

        if matched:
            score += 0.6
        elif candidates:
            score -= 0.2
        else:
            score -= 0.4

        # Deep analysis bonus: agent spent many turns producing this evidence
        if origin == "deep_code_analysis":
            score += 0.3

    if "vectorization" in action_id or patch_family.startswith("vectorization"):
        if compute_ratio >= 0.6:
            score += 0.3
        else:
            score -= 0.2

    if isinstance(evidence, list) and evidence:
        score += 0.1

    return max(0.0, min(score, 1.5))

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
