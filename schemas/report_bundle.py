from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class FigureSpec(BaseModel):
    title: str
    path: str
    caption: str = ""


class ReportBundle(BaseModel):
    report_md: str
    report_json: Dict[str, object]
    tables: Dict[str, List[Dict[str, object]]] = Field(default_factory=dict)
    figures: List[FigureSpec] = Field(default_factory=list)
    key_takeaways: List[str] = Field(default_factory=list)
