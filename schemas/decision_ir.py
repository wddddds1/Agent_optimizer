from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field

from schemas.strict_base import StrictBaseModel


class ArgRule(StrictBaseModel):
    flag: str
    position: Literal["prepend", "after_subcommand", "append"] = "append"
    replace_if_exists: bool = True
    reason: Optional[str] = None


class DecisionIR(StrictBaseModel):
    status: Literal["OK", "PARTIAL", "ERROR"] = "OK"
    allowed_families: List[str] = Field(default_factory=list)
    blocked_families: List[str] = Field(default_factory=list)
    arg_rules: List[ArgRule] = Field(default_factory=list)
    candidate_cids: List[int] = Field(default_factory=list)
    ranking_cids: List[int] = Field(default_factory=list)
    # Backward-compatible legacy fields. New protocol should use *_cids.
    candidates: List[str] = Field(default_factory=list)
    ranking: List[str] = Field(default_factory=list)
    max_candidates: Optional[int] = None
    stop: bool = False
    reason: str = ""
    notes: Optional[str] = None
