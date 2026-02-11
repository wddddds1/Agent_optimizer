from __future__ import annotations

from typing import List, Literal

from pydantic import Field

from schemas.strict_base import LLMStatus, StrictBaseModel


class PatchReview(StrictBaseModel):
    verdict: Literal["PASS", "FAIL"] = "FAIL"
    reasons: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    status: LLMStatus = "OK"
    missing_fields: List[str] = Field(default_factory=list)
