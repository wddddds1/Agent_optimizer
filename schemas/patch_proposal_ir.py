from __future__ import annotations

from typing import List

from pydantic import Field

from schemas.strict_base import LLMStatus, StrictBaseModel


class PatchProposal(StrictBaseModel):
    status: LLMStatus = "OK"
    patch_diff: str = ""
    touched_files: List[str] = Field(default_factory=list)
    rationale: str = ""
    assumptions: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    missing_fields: List[str] = Field(default_factory=list)
