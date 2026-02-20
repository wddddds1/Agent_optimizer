from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ProfileReport(BaseModel):
    timing_breakdown: Dict[str, float] = Field(default_factory=dict)
    system_metrics: Dict[str, float] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)
    log_path: Optional[str] = None
    tau_hotspots: List[Dict[str, object]] = Field(default_factory=list)
    bottleneck_portrait: Dict[str, object] = Field(default_factory=dict)


class BottleneckClassification(BaseModel):
    """Classification of an application's performance bottleneck type."""
    bottleneck_type: str = "mixed"   # compute, memory, branch, mixed
    arithmetic_intensity_estimate: str = ""
    memory_bandwidth_utilization: str = ""
    ipc: float = 0.0
    branch_miss_rate: float = 0.0
    effective_directions: List[str] = Field(default_factory=list)
    ineffective_directions: List[str] = Field(default_factory=list)
    rationale: str = ""
