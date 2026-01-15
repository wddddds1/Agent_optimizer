from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ResultIR(BaseModel):
    runtime_seconds: float
    derived_metrics: Dict[str, float] = Field(default_factory=dict)
    correctness_metrics: Dict[str, object] = Field(default_factory=dict)
    logs: List[str] = Field(default_factory=list)
    exit_code: int
    samples: Optional[List[float]] = None
