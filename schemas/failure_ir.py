from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from schemas.strict_base import LLMStatus, StrictBaseModel


class NextStep(StrictBaseModel):
    type: str
    detail: str


class FailureSummary(StrictBaseModel):
    run_id: str
    action_id: str
    category: str
    signature: str
    top_causes: List[str] = Field(default_factory=list)
    next_steps: List[NextStep] = Field(default_factory=list)
    suggest_debug_mode: bool = False
    suggest_disable_family: Optional[str] = None
    cooldown_rounds: int = 0
    repro_hint: Optional[str] = None
    confidence: float = 0.0
    status: LLMStatus = "OK"
    missing_fields: List[str] = Field(default_factory=list)
