from __future__ import annotations

from typing import List

from pydantic import Field

from schemas.action_ir import ActionIR
from schemas.strict_base import StrictBaseModel


class CandidateList(StrictBaseModel):
    family: str
    candidates: List[ActionIR] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    confidence: float = 0.0
