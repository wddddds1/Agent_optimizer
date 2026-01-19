from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import Field

from schemas.strict_base import LLMStatus, StrictBaseModel


class ReviewDecision(StrictBaseModel):
    should_stop: bool = False
    confidence: float = 0.0
    reason: str = ""
    evidence: Dict[str, object] = Field(default_factory=dict)
    suggested_next_step: Literal["continue", "stop", "switch_family", "tighten_gates"] = "continue"
    status: LLMStatus = "OK"
    missing_fields: List[str] = Field(default_factory=list)
