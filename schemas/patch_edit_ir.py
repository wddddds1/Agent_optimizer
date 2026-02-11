from __future__ import annotations

from typing import List, Literal

from pydantic import Field

from schemas.strict_base import LLMStatus, StrictBaseModel


class PatchEdit(StrictBaseModel):
    file: str
    op: Literal["replace", "delete", "insert_before", "insert_after"]
    anchor: str
    old_text: str = ""
    new_text: str = ""


class PatchEditProposal(StrictBaseModel):
    status: LLMStatus = "OK"
    edits: List[PatchEdit] = Field(default_factory=list)
    touched_files: List[str] = Field(default_factory=list)
    rationale: str = ""
    assumptions: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    missing_fields: List[str] = Field(default_factory=list)
