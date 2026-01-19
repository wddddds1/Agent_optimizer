from __future__ import annotations

from typing import Dict, List

from pydantic import Field

from schemas.action_ir import ActionIR
from schemas.strict_base import StrictBaseModel


class RankedAction(StrictBaseModel):
    action: ActionIR
    score: float = 0.0
    score_breakdown: Dict[str, float] = Field(default_factory=dict)


class Rejection(StrictBaseModel):
    action_id: str
    reason: str


class RankedActions(StrictBaseModel):
    ranked: List[RankedAction] = Field(default_factory=list)
    rejected: List[Rejection] = Field(default_factory=list)
    scoring_notes: str = ""
