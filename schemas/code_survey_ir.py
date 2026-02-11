from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from schemas.strict_base import LLMStatus, StrictBaseModel


class CodeOpportunity(StrictBaseModel):
    opportunity_id: str
    family_hint: str
    patch_family: Optional[str] = None
    file_path: str
    snippet_tag: Optional[str] = None
    anchor_hint: Optional[str] = None
    rationale: str = ""
    evidence: List[str] = Field(default_factory=list)
    confidence: float = 0.5


class CodeSurveyResult(StrictBaseModel):
    opportunities: List[CodeOpportunity] = Field(default_factory=list)
    status: LLMStatus = "OK"
    missing_fields: List[str] = Field(default_factory=list)
