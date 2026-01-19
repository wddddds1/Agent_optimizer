from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class NextStep(BaseModel):
    type: str
    detail: str


class FailureSummary(BaseModel):
    run_id: str
    action_id: str
    category: str
    signature: str
    top_causes: List[str] = Field(default_factory=list)
    next_steps: List[NextStep] = Field(default_factory=list)
    suggest_debug_mode: bool = False
    suggest_disable_family: Optional[str] = None
    confidence: float = 0.0
