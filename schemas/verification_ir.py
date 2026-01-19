from __future__ import annotations

from typing import Dict, Optional

from pydantic import Field

from schemas.strict_base import LLMStatus, StrictBaseModel

class GateOutcome(StrictBaseModel):
    passed: bool
    details: Dict[str, object] = Field(default_factory=dict)


class VerifiedMetrics(StrictBaseModel):
    baseline_median_s: Optional[float] = None
    candidate_median_s: Optional[float] = None
    relative_improvement: Optional[float] = None
    variance_cv: Optional[float] = None


class VerificationResult(StrictBaseModel):
    run_id: str
    action_id: str
    verdict: str
    runtime_gate: GateOutcome
    performance_gate: GateOutcome
    correctness_gate: GateOutcome
    metrics: VerifiedMetrics = Field(default_factory=VerifiedMetrics)
    reasons: list[str] = Field(default_factory=list)
    status: LLMStatus = "OK"
    missing_fields: list[str] = Field(default_factory=list)
