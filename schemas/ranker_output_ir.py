from __future__ import annotations

from typing import Dict, List

from pydantic import Field

from schemas.strict_base import LLMStatus, StrictBaseModel


class RankerRejection(StrictBaseModel):
    action_id: str
    reason: str


class RankerOutput(StrictBaseModel):
    ranked_action_ids: List[str] = Field(default_factory=list)
    rejected: List[RankerRejection] = Field(default_factory=list)
    scoring_notes: str = ""
    confidence: float = 0.0
    score_breakdown: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    status: LLMStatus = "OK"
    missing_fields: List[str] = Field(default_factory=list)
