from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


AppliesTo = Literal["run_config", "input_script", "build_config"]
ExpectedEffect = Literal[
    "comm_reduce",
    "mem_locality",
    "compute_opt",
    "imbalance_reduce",
    "io_reduce",
]
RiskLevel = Literal["low", "medium", "high"]


class VerificationPlan(BaseModel):
    gates: List[Literal["runtime", "correctness", "variance"]] = Field(default_factory=list)
    thresholds: Dict[str, float] = Field(default_factory=dict)


class ActionIR(BaseModel):
    action_id: str
    family: str
    description: str
    applies_to: List[AppliesTo]
    parameters: Dict[str, object] = Field(default_factory=dict)
    preconditions: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    expected_effect: List[ExpectedEffect] = Field(default_factory=list)
    risk_level: RiskLevel = "low"
    verification_plan: VerificationPlan = Field(default_factory=VerificationPlan)
    notes: Optional[str] = None
