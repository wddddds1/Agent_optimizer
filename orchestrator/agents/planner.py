from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import ValidationError

from schemas.plan_ir import EvaluationPlan, FusePlan, PlanIR, StopPlan
from schemas.analysis_ir import AnalysisResult
from schemas.job_ir import Budgets
from schemas.profile_report import ProfileReport
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
        cost_model: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, object]] = None,
    ) -> PlanIR:
        if (not analysis.allowed_families) and availability:
            analysis.allowed_families = [
                fam for fam, count in availability.items() if count > 0
            ]
        llm_plan = self._try_llm(
            iteration_id, analysis, budgets, history, availability, cost_model, context
        )
        if llm_plan is not None:
            normalized = _normalize_plan(
                llm_plan, analysis, self.defaults, budgets, availability, cost_model
            )
            return _apply_cost_guard(normalized, cost_model, self.defaults)
        chosen = _select_families(
            analysis.allowed_families, history, analysis.bottleneck, availability
        )
        if analysis.confidence < 0.5:
            chosen = [fam for fam in chosen if fam not in {"build_config", "source_patch"}]
        evaluation = _build_evaluation(self.defaults.get("evaluation", {}))
        fuse_rules = _build_fuse(self.defaults.get("fuse_rules", {}))
        stop_plan = _build_stop(self.defaults.get("stop_plan", {}), budgets)
        max_candidates = int(self.defaults.get("max_candidates", 3))
        reason = analysis.rationale or "planner default"
        plan = PlanIR(
            iteration_id=iteration_id,
            chosen_families=chosen,
            max_candidates=max_candidates,
            evaluation=evaluation,
            enable_debug_mode=False,
            fuse_rules=fuse_rules,
            stop_condition=stop_plan,
            reason=reason,
        )
        return _apply_cost_guard(plan, cost_model, self.defaults)

    def analyze(
        self,
        profile: ProfileReport,
        history: Dict[str, object],
        policy: Dict[str, object],
        case_tags: List[str],
        profile_features: Optional[Dict[str, object]] = None,
    ) -> AnalysisResult:
        del history, policy, case_tags, profile_features
        return _heuristic_analysis(profile)

    def _try_llm(
        self,
        iteration_id: int,
        analysis: AnalysisResult,
        budgets: Budgets,
        history: Dict[str, object],
        availability: Optional[Dict[str, int]],
        cost_model: Optional[Dict[str, float]],
        context: Optional[Dict[str, object]],
    ) -> Optional[PlanIR]:
        if not self.llm_client or not self.llm_client.config.enabled:
            return None
        prompt = _load_prompt("planner")
        payload = {
            "iteration_id": iteration_id,
            "analysis": analysis.model_dump(),
            "context": context or {},
            "budgets": budgets.model_dump(),
            "history": history,
            "defaults": self.defaults,
            "availability": availability or {},
            "cost_model": cost_model or {},
        }
        data = self.llm_client.request_json(prompt, payload)
        if not data:
            return None
        try:
            plan = PlanIR(**data)
        except ValidationError:
            return None
        if plan.status != "OK":
            return None
        return plan


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
        return ["comm_tune", "parallel_omp", "parallel_pthread", "omp_pkg", "affinity_tune", "wait_policy"]
    if bottleneck == "io":
        return ["io_tune", "parallel_omp", "parallel_pthread", "affinity_tune", "wait_policy"]
    return ["parallel_omp", "parallel_pthread", "omp_pkg", "affinity_tune", "wait_policy", "sched_granularity"]


def _normalize_plan(
    plan: PlanIR,
    analysis: AnalysisResult,
    defaults: Dict[str, object],
    budgets: Budgets,
    availability: Optional[Dict[str, int]],
    cost_model: Optional[Dict[str, float]],
) -> PlanIR:
    allowed = set(analysis.allowed_families)
    chosen = [fam for fam in plan.chosen_families if fam in allowed]
    if availability:
        chosen = [fam for fam in chosen if availability.get(fam, 0) > 0]
    if analysis.confidence < 0.5:
        chosen = [fam for fam in chosen if fam not in {"build_config", "source_patch"}]
    if not chosen:
        chosen = _select_families(
            analysis.allowed_families, {}, analysis.bottleneck, availability
        )
    evaluation = plan.evaluation or _build_evaluation(defaults.get("evaluation", {}))
    fuse_rules = plan.fuse_rules or _build_fuse(defaults.get("fuse_rules", {}))
    stop_plan = plan.stop_condition or _build_stop(defaults.get("stop_plan", {}), budgets)
    max_candidates = plan.max_candidates or int(defaults.get("max_candidates", 3))
    if fuse_rules.fallback_family is None and "run_config" in analysis.allowed_families:
        fuse_rules.fallback_family = "run_config"
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


def _apply_cost_guard(
    plan: PlanIR,
    cost_model: Optional[Dict[str, float]],
    defaults: Dict[str, object],
) -> PlanIR:
    if not cost_model:
        return plan
    guard = defaults.get("cost_guard", {})
    try:
        build_high = float(guard.get("build_seconds_high", 120.0))
        run_high = float(guard.get("run_seconds_high", 600.0))
        max_candidates = int(guard.get("max_candidates_if_high_cost", plan.max_candidates))
        force_halving = bool(guard.get("force_halving", False))
    except (TypeError, ValueError):
        return plan
    avg_build = float(cost_model.get("avg_build_seconds", 0.0) or 0.0)
    avg_run = float(cost_model.get("avg_run_seconds", 0.0) or 0.0)
    if avg_build >= build_high or avg_run >= run_high:
        plan.max_candidates = min(plan.max_candidates, max_candidates)
        if force_halving:
            plan.evaluation.use_successive_halving = True
    return plan


def _load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "agents" / f"{name}.md"
    return path.read_text(encoding="utf-8")


def _analysis_confidence(profile: ProfileReport) -> float:
    timing = profile.timing_breakdown or {}
    total = timing.get("total", 0.0) or 0.0
    if total <= 0.0:
        return 0.2
    keys = ["pair", "kspace", "neigh", "comm", "modify", "output"]
    present = [key for key in keys if timing.get(key) is not None]
    positive = [key for key in keys if (timing.get(key) or 0.0) > 0.0]
    if len(positive) >= 2:
        return 0.9
    if len(present) >= 2:
        return 0.6
    return 0.3


def _build_rationale(
    bottleneck: str,
    comm_ratio: float,
    output_ratio: float,
    confidence: float,
) -> str:
    if confidence < 0.5:
        return "profiling signal weak; allow low-risk run_config only"
    if bottleneck == "comm":
        return f"comm ratio {comm_ratio:.2f} suggests communication tuning"
    if bottleneck == "io":
        return f"output ratio {output_ratio:.2f} suggests IO tuning"
    return "compute-dominant; prioritize parallel strategy and affinity"


def _heuristic_analysis(profile: ProfileReport) -> AnalysisResult:
    timing = profile.timing_breakdown or {}
    total = timing.get("total", 0.0) or 0.0
    comm_ratio = (timing.get("comm", 0.0) / total) if total else 0.0
    output_ratio = (timing.get("output", 0.0) / total) if total else 0.0
    bottleneck = "compute"
    if comm_ratio > 0.2:
        bottleneck = "comm"
    if output_ratio > 0.2:
        bottleneck = "io"
    confidence = _analysis_confidence(profile)
    # Generic base set: parallelism families for all threading models,
    # plus safe env-only tuning families.  The downstream availability
    # filter removes families that have no actions in the action_space.
    allowed = {
        "parallel_omp",
        "parallel_pthread",
        "affinity_tune",
        "runtime_backend_select",
        "runtime_lib",
        "sched_granularity",
        "wait_policy",
        "neighbor_tune",
        "output_tune",
    }
    if bottleneck == "comm":
        allowed.update({"comm_tune", "load_balance"})
    if bottleneck == "io":
        allowed.update({"io_tune", "output_tune"})
    if confidence >= 0.7:
        allowed.update({"build_config", "source_patch", "lib_threading", "omp_pkg", "accuracy_tune"})
    rationale = _build_rationale(bottleneck, comm_ratio, output_ratio, confidence)
    return AnalysisResult(
        bottleneck=bottleneck,
        allowed_families=sorted(allowed),
        allowed_transforms=[],
        forbidden_transforms=[],
        risk_overrides={},
        confidence=confidence,
        rationale=rationale,
    )


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
