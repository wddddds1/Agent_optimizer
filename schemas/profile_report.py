from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ProfileReport(BaseModel):
    timing_breakdown: Dict[str, float] = Field(default_factory=dict)
    system_metrics: Dict[str, float] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)
    log_path: Optional[str] = None
    tau_hotspots: List[Dict[str, object]] = Field(default_factory=list)
