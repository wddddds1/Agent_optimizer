from __future__ import annotations

from math import sqrt, tanh
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import ValidationError

from schemas.action_ir import ActionIR
from schemas.candidate_ir import CandidateList
from schemas.ranking_ir import RankedAction, RankedActions, Rejection
from schemas.profile_report import ProfileReport
from schemas.ranker_output_ir import RankerOutput
from orchestrator.errors import LLMUnavailableError
from orchestrator.llm_client import LLMClient
from orchestrator.router import RuleContext, filter_actions


class RouterRankerAgent:
    def __init__(self, llm_client: Optional[LLMClient]) -> None:
        self.llm_client = llm_client
        self.last_llm_trace: Optional[Dict[str, object]] = None

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
        memory_posteriors: Optional[Dict[str, Dict[str, float]]] = None,
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
        trace_out: Dict[str, object] = {}
        ranked_actions, notes, llm_rejections = _rank_actions(
            filtered,
            profile,
            profile_features or {},
            hotspot_map or {},
            self.llm_client,
            rank_cfg or {},
            tested_actions or [],
            memory_scores or {},
            memory_posteriors or {},
            patch_stats or {},
            trace_out=trace_out,
        )
        self.last_llm_trace = trace_out or None
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
    memory_posteriors: Dict[str, Dict[str, float]],
    patch_stats: Dict[str, Dict[str, int]],
    trace_out: Optional[Dict[str, object]] = None,
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
    bayesian_cfg = rank_cfg.get("bayesian", {}) if isinstance(rank_cfg, dict) else {}

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
        if isinstance(trace_out, dict):
            trace_out["payload"] = payload
            trace_out["response"] = data
        if isinstance(data, dict):
            try:
                output = RankerOutput(**data)
            except ValidationError:
                if llm_client and llm_client.config.strict_availability:
                    raise LLMUnavailableError("RouterRankerAgent returned invalid RankerOutput JSON")
                output = None
            if output and output.status == "OK" and output.ranked_action_ids:
                rejected = [Rejection(action_id=item.action_id, reason=item.reason) for item in output.rejected]
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
                    memory_posteriors=memory_posteriors,
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
                    bayesian_cfg=bayesian_cfg if isinstance(bayesian_cfg, dict) else {},
                    patch_stats=patch_stats,
                )
                ranked = _apply_exploration(ranked, tested_actions, epsilon_explore)
                ranked = _apply_value_density_filter(ranked, rank_cfg)
                ranked = _apply_macro_first_policy(ranked, tested_actions, rank_cfg)
                return ranked, llm_notes, rejected
            if output and output.status != "OK" and llm_client and llm_client.config.strict_availability:
                raise LLMUnavailableError(
                    f"RouterRankerAgent returned non-OK status: {output.status}"
                )
    ranked = _final_rank(
        actions=actions,
        profile=profile,
        profile_features=profile_features,
        hotspot_map=hotspot_map,
        tested_actions=tested_actions,
        memory_scores=memory_scores,
        memory_posteriors=memory_posteriors,
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
        bayesian_cfg=bayesian_cfg if isinstance(bayesian_cfg, dict) else {},
        patch_stats=patch_stats,
    )
    ranked = _apply_exploration(ranked, tested_actions, epsilon_explore)
    ranked = _apply_value_density_filter(ranked, rank_cfg)
    ranked = _apply_macro_first_policy(ranked, tested_actions, rank_cfg)
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
    tau_hotspots = profile.tau_hotspots or []
    compute_ratio = (timing.get("pair", 0.0) / total) if total else 0.0
    if compute_ratio <= 0.0 and tau_hotspots:
        compute_ratio = 0.65
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
        "has_function_hotspots": bool(tau_hotspots),
    }


def _final_rank(
    actions: List[ActionIR],
    profile: ProfileReport,
    profile_features: Dict[str, object],
    hotspot_map: Dict[str, object],
    tested_actions: List[str],
    memory_scores: Dict[str, float],
    memory_posteriors: Dict[str, Dict[str, float]],
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
    bayesian_cfg: Dict[str, object],
    patch_stats: Dict[str, Dict[str, int]],
) -> List[RankedAction]:
    pseudo_heur = max(float(bayesian_cfg.get("heuristic_pseudo", 1.0) or 1.0), 0.0)
    pseudo_evidence = max(float(bayesian_cfg.get("evidence_pseudo", 1.2) or 1.2), 0.0)
    pseudo_llm = max(float(bayesian_cfg.get("llm_pseudo", 0.6) or 0.6), 0.0)
    heuristic_scale = max(float(bayesian_cfg.get("heuristic_scale", 2.0) or 2.0), 1.0e-6)
    llm_scale = max(float(bayesian_cfg.get("llm_scale", 6.0) or 6.0), 1.0e-6)
    uncertainty_weight = max(float(bayesian_cfg.get("uncertainty_weight", 0.15) or 0.15), 0.0)
    gain_floor = max(float(bayesian_cfg.get("gain_floor", 0.05) or 0.05), 0.0)
    gain_boost_heur = max(float(bayesian_cfg.get("heuristic_gain_boost", 0.15) or 0.15), 0.0)
    gain_boost_evidence = max(float(bayesian_cfg.get("evidence_gain_boost", 0.2) or 0.2), 0.0)

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

        heur_norm = tanh(heur_score / heuristic_scale)
        llm_norm = tanh(llm_score / llm_scale)
        evidence_norm = _normalize_evidence_score(evidence_score)
        mem_posterior = memory_posteriors.get(action.action_id, {})
        alpha = max(float(mem_posterior.get("alpha", 1.0) or 1.0), 1.0e-6)
        beta = max(float(mem_posterior.get("beta", 1.0) or 1.0), 1.0e-6)
        mem_gain_norm = max(0.0, min(float(mem_posterior.get("gain_norm", 0.0) or 0.0), 1.0))
        if not mem_posterior:
            # Legacy fallback when only scalar memory score is available.
            alpha = 1.0 + max(mem_score, 0.0)
            beta = 1.0 + max(-mem_score, 0.0)
            mem_gain_norm = max(0.0, min(abs(mem_score), 1.0))

        alpha_post = alpha
        beta_post = beta
        alpha_post += pseudo_heur * max(heur_norm, 0.0)
        beta_post += pseudo_heur * max(-heur_norm, 0.0)
        alpha_post += pseudo_evidence * max(evidence_norm, 0.0)
        beta_post += pseudo_evidence * max(-evidence_norm, 0.0)
        alpha_post += pseudo_llm * max(llm_norm, 0.0)
        beta_post += pseudo_llm * max(-llm_norm, 0.0)

        denom = alpha_post + beta_post
        p_success_post = alpha_post / denom if denom > 0.0 else 0.5
        uncertainty_post = sqrt(max(p_success_post * (1.0 - p_success_post), 0.0) / (denom + 1.0))
        gain_term = max(
            gain_floor,
            min(
                1.0,
                mem_gain_norm
                + gain_boost_heur * max(heur_norm, 0.0)
                + gain_boost_evidence * max(evidence_norm, 0.0),
            ),
        )
        bayes_utility = p_success_post * gain_term - uncertainty_weight * uncertainty_post
        bayes_utility = max(-1.0, min(1.0, bayes_utility))

        final_score = (
            w_memory * bayes_utility
            + 0.25 * w_llm * llm_norm
            + 0.25 * w_heuristic * heur_norm
            + 0.25 * w_evidence * evidence_norm
            + novelty
            - risk_penalty
            - patch_penalty
        )
        breakdown.update(
            {
                "memory_score": mem_score,
                "llm_score": llm_score,
                "heuristic_score": heur_score,
                "llm_norm": llm_norm,
                "heuristic_norm": heur_norm,
                "novelty_bonus": novelty,
                "risk_penalty": -risk_penalty,
                "evidence_ok": evidence_ok,
                "evidence_score": evidence_score,
                "evidence_norm": evidence_norm,
                "memory_alpha": alpha,
                "memory_beta": beta,
                "memory_gain_norm": mem_gain_norm,
                "bayes_p_success": p_success_post,
                "bayes_uncertainty": uncertainty_post,
                "bayes_utility": bayes_utility,
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


def _apply_value_density_filter(
    ranked: List[RankedAction],
    rank_cfg: Dict[str, object],
) -> List[RankedAction]:
    if not ranked:
        return ranked
    min_density = float(rank_cfg.get("value_density_min", 0.0) or 0.0)
    if min_density <= 0.0:
        return ranked
    keep: List[RankedAction] = []
    demote: List[RankedAction] = []
    for item in ranked:
        action = item.action
        if action.family != "source_patch":
            keep.append(item)
            continue
        params = action.parameters or {}
        p50 = float(params.get("expected_gain_p50", 0.0) or 0.0)
        cost = float(params.get("implementation_cost", 0.0) or 0.0)
        density = p50 / max(cost, 1.0e-6) if cost > 0 else 0.0
        if density < min_density:
            item.score_breakdown["value_density"] = density
            item.score_breakdown["value_density_filtered"] = -1.0
            demote.append(item)
        else:
            item.score_breakdown["value_density"] = density
            keep.append(item)
    return keep + demote


def _apply_macro_first_policy(
    ranked: List[RankedAction],
    tested_actions: List[str],
    rank_cfg: Dict[str, object],
) -> List[RankedAction]:
    if not ranked:
        return ranked
    macro_cfg = rank_cfg.get("macro_first", {}) if isinstance(rank_cfg, dict) else {}
    enabled = True if not isinstance(macro_cfg, dict) else bool(macro_cfg.get("enabled", True))
    if not enabled:
        return ranked
    top_n = int(macro_cfg.get("protect_top_n", 2) or 2) if isinstance(macro_cfg, dict) else 2
    macro_mechanisms = {
        "data_layout",
        "memory_path",
        "vectorization",
        "algorithmic",
    }
    tested = set(tested_actions or [])

    def _mechanism(action: ActionIR) -> str:
        params = action.parameters or {}
        direct = str(params.get("graph_mechanism", "") or "").strip().lower()
        if direct:
            return direct
        patch_family = str(params.get("patch_family", "") or "").strip().lower()
        if patch_family.startswith("source_patch:"):
            return patch_family.split(":", 1)[1]
        return ""

    macro_candidates = [
        item
        for item in ranked
        if item.action.family == "source_patch"
        and item.action.action_id not in tested
        and _mechanism(item.action) in macro_mechanisms
    ]
    if not macro_candidates:
        return ranked

    reordered: List[RankedAction] = []
    used_ids: set[str] = set()
    needed = max(0, min(top_n, len(macro_candidates)))
    for item in macro_candidates[:needed]:
        reordered.append(item)
        used_ids.add(item.action.action_id)
        item.score_breakdown["macro_first_boost"] = 1.0
    for item in ranked:
        if item.action.action_id in used_ids:
            continue
        mech = _mechanism(item.action)
        if len(reordered) < top_n and mech == "micro_opt":
            continue
        reordered.append(item)
    final_ids = {it.action.action_id for it in reordered}
    for item in ranked:
        if item.action.action_id in final_ids:
            continue
        reordered.append(item)
        final_ids.add(item.action.action_id)
    return reordered


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
    # Keep patch exploration alive: penalize repeated failures gently, do not
    # let early misses suppress promising source-level actions.
    penalty += 0.05 * context_misses
    penalty += 0.15 * preflight_fails
    penalty += 0.20 * build_fails
    return min(penalty, 0.9)


def _normalize_evidence_score(evidence_score: float) -> float:
    # Evidence score typically lies in [0, 1.5]. Re-center to roughly [-0.5, 1.0].
    centered = evidence_score - 0.5
    return max(-1.0, min(centered, 1.0))


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
