from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class GateOutcome(BaseModel):
    passed: bool
    details: Dict[str, object] = Field(default_factory=dict)


class VerifiedMetrics(BaseModel):
    baseline_median_s: Optional[float] = None
    candidate_median_s: Optional[float] = None
    relative_improvement: Optional[float] = None
    variance_cv: Optional[float] = None


class VerificationResult(BaseModel):
    run_id: str
    action_id: str
    verdict: str
    runtime_gate: GateOutcome
    performance_gate: GateOutcome
    correctness_gate: GateOutcome
    metrics: VerifiedMetrics = Field(default_factory=VerifiedMetrics)
    reasons: list[str] = Field(default_factory=list)
