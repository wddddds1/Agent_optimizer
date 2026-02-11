from __future__ import annotations

from typing import Dict, List

from pydantic import Field

from schemas.strict_base import LLMStatus, StrictBaseModel


class CandidateGroup(StrictBaseModel):
    family: str
    action_ids: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    family_rationale: str = ""
    action_rationales: Dict[str, str] = Field(default_factory=dict)


class OptimizerOutput(StrictBaseModel):
    candidates: List[CandidateGroup] = Field(default_factory=list)
    overall_rationale: str = ""
    status: LLMStatus = "OK"
    missing_fields: List[str] = Field(default_factory=list)
