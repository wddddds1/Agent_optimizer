from __future__ import annotations

from typing import Dict, List

from pydantic import Field

from schemas.strict_base import LLMStatus, StrictBaseModel


class AnalysisResult(StrictBaseModel):
    bottleneck: str
    allowed_families: List[str] = Field(default_factory=list)
    allowed_transforms: List[str] = Field(default_factory=list)
    forbidden_transforms: List[str] = Field(default_factory=list)
    risk_overrides: Dict[str, object] = Field(default_factory=dict)
    confidence: float = 0.0
    rationale: str = ""
    status: LLMStatus = "OK"
    missing_fields: List[str] = Field(default_factory=list)
