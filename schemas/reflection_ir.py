from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from schemas.strict_base import LLMStatus, StrictBaseModel


class OpportunityReflection(StrictBaseModel):
    """Per-opportunity reflection after a batch iteration."""

    opportunity_id: str
    status: str = ""  # "succeeded" | "failed" | "untried" | "deprioritized"
    gain_pct: Optional[float] = None
    lesson: str = ""


class ReflectionResult(StrictBaseModel):
    """Output of the ReflectionAgent: reprioritised opportunities + strategy."""

    status: LLMStatus = "OK"
    missing_fields: List[str] = Field(default_factory=list)
    reprioritized_ids: List[str] = Field(default_factory=list)
    reflections: List[OpportunityReflection] = Field(default_factory=list)
    strategy_note: str = ""
    direction_hint: str = ""
    skip_ids: List[str] = Field(default_factory=list)
