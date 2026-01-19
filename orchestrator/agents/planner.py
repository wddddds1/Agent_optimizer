from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import ValidationError

from schemas.plan_ir import EvaluationPlan, FusePlan, PlanIR, StopPlan
from schemas.analysis_ir import AnalysisResult
from schemas.job_ir import Budgets
from orchestrator.llm_client import LLMClient


class PlannerAgent:
    def __init__(self, defaults: Dict[str, object], llm_client: Optional[LLMClient]) -> None:
        self.defaults = defaults or {}
        self.llm_client = llm_client

    def plan(
        self,
        iteration_id: int,
        analysis: AnalysisResult,
        budgets: Budgets,
        history: Dict[str, object],
        availability: Optional[Dict[str, int]] = None,
    ) -> PlanIR:
        llm_plan = self._try_llm(iteration_id, analysis, budgets, history, availability)
        if llm_plan is not None:
            return _normalize_plan(llm_plan, analysis, self.defaults, budgets, availability)
        chosen = _select_families(
            analysis.allowed_families, history, analysis.bottleneck, availability
        )
        evaluation = _build_evaluation(self.defaults.get("evaluation", {}))
        fuse_rules = _build_fuse(self.defaults.get("fuse_rules", {}))
        stop_plan = _build_stop(self.defaults.get("stop_plan", {}), budgets)
        max_candidates = int(self.defaults.get("max_candidates", 3))
        reason = analysis.rationale or "planner default"
        return PlanIR(
            iteration_id=iteration_id,
            chosen_families=chosen,
            max_candidates=max_candidates,
            evaluation=evaluation,
            enable_debug_mode=False,
            fuse_rules=fuse_rules,
            stop_condition=stop_plan,
            reason=reason,
        )

    def _try_llm(
        self,
        iteration_id: int,
        analysis: AnalysisResult,
        budgets: Budgets,
        history: Dict[str, object],
        availability: Optional[Dict[str, int]],
    ) -> Optional[PlanIR]:
        if not self.llm_client or not self.llm_client.config.enabled:
            return None
        prompt = _load_prompt("planner")
        payload = {
            "iteration_id": iteration_id,
            "analysis": analysis.model_dump(),
            "budgets": budgets.model_dump(),
            "history": history,
            "defaults": self.defaults,
            "availability": availability or {},
        }
        data = self.llm_client.request_json(prompt, payload)
        if not data:
            return None
        try:
            return PlanIR(**data)
        except ValidationError:
            return None


def _select_families(
    allowed: List[str],
    history: Dict[str, object],
    bottleneck: str,
    availability: Optional[Dict[str, int]],
) -> List[str]:
    if not allowed:
        return []
    allowed = _filter_available(allowed, availability)
    priority = _priority_families(bottleneck)
    ordered: List[str] = []
    for fam in priority:
        if fam in allowed and fam not in ordered:
            ordered.append(fam)
    for fam in sorted(allowed):
        if fam not in ordered:
            ordered.append(fam)
    gains = history.get("family_best_gain", {})
    if isinstance(gains, dict) and gains:
        base_rank = {fam: idx for idx, fam in enumerate(ordered)}
        ordered.sort(key=lambda fam: (-gains.get(fam, 0.0), base_rank.get(fam, 999)))
    return ordered[:2]


def _priority_families(bottleneck: str) -> List[str]:
    if bottleneck == "comm":
        return ["comm_tune", "parallel_omp", "omp_pkg", "affinity_tune", "wait_policy"]
    if bottleneck == "io":
        return ["io_tune", "parallel_omp", "affinity_tune", "wait_policy"]
    return ["parallel_omp", "omp_pkg", "affinity_tune", "wait_policy", "sched_granularity"]


def _normalize_plan(
    plan: PlanIR,
    analysis: AnalysisResult,
    defaults: Dict[str, object],
    budgets: Budgets,
    availability: Optional[Dict[str, int]],
) -> PlanIR:
    allowed = set(analysis.allowed_families)
    chosen = [fam for fam in plan.chosen_families if fam in allowed]
    if availability:
        chosen = [fam for fam in chosen if availability.get(fam, 0) > 0]
    if not chosen:
        chosen = _select_families(
            analysis.allowed_families, {}, analysis.bottleneck, availability
        )
    evaluation = plan.evaluation or _build_evaluation(defaults.get("evaluation", {}))
    fuse_rules = plan.fuse_rules or _build_fuse(defaults.get("fuse_rules", {}))
    stop_plan = plan.stop_condition or _build_stop(defaults.get("stop_plan", {}), budgets)
    max_candidates = plan.max_candidates or int(defaults.get("max_candidates", 3))
    return PlanIR(
        iteration_id=plan.iteration_id or 0,
        chosen_families=chosen,
        max_candidates=max_candidates,
        evaluation=evaluation,
        enable_debug_mode=plan.enable_debug_mode,
        fuse_rules=fuse_rules,
        stop_condition=stop_plan,
        reason=plan.reason or analysis.rationale,
    )


def _load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "agents" / f"{name}.md"
    return path.read_text(encoding="utf-8")


def _build_evaluation(raw: Dict[str, object]) -> EvaluationPlan:
    return EvaluationPlan(
        baseline_repeats=int(raw.get("baseline_repeats", 1)),
        candidate_repeats_stage0=int(raw.get("candidate_repeats_stage0", 1)),
        candidate_repeats_stage1=int(raw.get("candidate_repeats_stage1", 1)),
        top1_validation_repeats=int(raw.get("top1_validation_repeats", 0)),
        use_successive_halving=bool(raw.get("use_successive_halving", False)),
    )


def _build_fuse(raw: Dict[str, object]) -> FusePlan:
    return FusePlan(
        max_compile_fails=int(raw.get("max_compile_fails", 2)),
        max_runtime_fails=int(raw.get("max_runtime_fails", 3)),
        cooldown_rounds=int(raw.get("cooldown_rounds", 1)),
        fallback_family=raw.get("fallback_family"),
    )


def _build_stop(raw: Dict[str, object], budgets: Budgets) -> StopPlan:
    max_iters = raw.get("max_iterations")
    if max_iters is None:
        max_iters = budgets.max_iters
    return StopPlan(
        max_iterations=max_iters,
        min_relative_gain=float(raw.get("min_relative_gain", 0.0)),
        patience_rounds=int(raw.get("patience_rounds", 2)),
    )


def _filter_available(
    allowed: List[str],
    availability: Optional[Dict[str, int]],
) -> List[str]:
    if not availability:
        return list(allowed)
    filtered = [fam for fam in allowed if availability.get(fam, 0) > 0]
    return filtered or list(allowed)
