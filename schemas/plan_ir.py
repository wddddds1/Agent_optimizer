from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from schemas.strict_base import LLMStatus, StrictBaseModel


class EvaluationPlan(StrictBaseModel):
    baseline_repeats: int = 1
    candidate_repeats_stage0: int = 1
    candidate_repeats_stage1: int = 1
    top1_validation_repeats: int = 0
    use_successive_halving: bool = False


class FusePlan(StrictBaseModel):
    max_compile_fails: int = 2
    max_runtime_fails: int = 3
    cooldown_rounds: int = 1
    fallback_family: Optional[str] = None


class StopPlan(StrictBaseModel):
    max_iterations: Optional[int] = None
    min_relative_gain: float = 0.0
    patience_rounds: int = 2


class PlanIR(StrictBaseModel):
    iteration_id: int
    chosen_families: List[str] = Field(default_factory=list)
    max_candidates: int = 3
    evaluation: EvaluationPlan = Field(default_factory=EvaluationPlan)
    enable_debug_mode: bool = False
    fuse_rules: FusePlan = Field(default_factory=FusePlan)
    stop_condition: StopPlan = Field(default_factory=StopPlan)
    reason: str = ""
    status: LLMStatus = "OK"
    missing_fields: List[str] = Field(default_factory=list)
