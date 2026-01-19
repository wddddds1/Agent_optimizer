from __future__ import annotations

from typing import Dict, List

from pydantic import Field

from schemas.strict_base import LLMStatus, StrictBaseModel


class FigureSpec(StrictBaseModel):
    title: str
    path: str
    caption: str = ""


class ReportBundle(StrictBaseModel):
    report_md: str
    report_json: Dict[str, object]
    tables: Dict[str, List[Dict[str, object]]] = Field(default_factory=dict)
    figures: List[FigureSpec] = Field(default_factory=list)
    key_takeaways: List[str] = Field(default_factory=list)
    status: LLMStatus = "OK"
    missing_fields: List[str] = Field(default_factory=list)
