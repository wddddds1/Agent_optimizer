from __future__ import annotations

from typing import List

from pydantic import Field

from schemas.action_ir import AppliesTo, ExpectedEffect, RiskLevel
from schemas.strict_base import LLMStatus, StrictBaseModel


class OptimizationIdea(StrictBaseModel):
    idea_id: str
    family_hint: str
    applies_to: List[AppliesTo] = Field(default_factory=list)
    mechanism: str = ""
    expected_effect: List[ExpectedEffect] = Field(default_factory=list)
    risk_level: RiskLevel = "low"
    rationale: str = ""
    evidence: List[str] = Field(default_factory=list)


class IdeaList(StrictBaseModel):
    ideas: List[OptimizationIdea] = Field(default_factory=list)
    status: LLMStatus = "OK"
    missing_fields: List[str] = Field(default_factory=list)
