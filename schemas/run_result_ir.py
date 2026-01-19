from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class RunArtifactsIndex(BaseModel):
    run_dir: str
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    log_path: Optional[str] = None
    time_path: Optional[str] = None
    build_log: Optional[str] = None


class RunProvenance(BaseModel):
    binary_path: Optional[str] = None
    binary_sha256: Optional[str] = None
    build_dir: Optional[str] = None
    git_commit: Optional[str] = None


class TimingSummary(BaseModel):
    build_seconds: Optional[float] = None
    run_seconds: Optional[float] = None


class RunResultIR(BaseModel):
    run_id: str
    action_id: str
    status: str
    phase: str
    artifacts: RunArtifactsIndex
    provenance: RunProvenance = Field(default_factory=RunProvenance)
    timings: TimingSummary = Field(default_factory=TimingSummary)
    error_message: Optional[str] = None
    metrics: Dict[str, object] = Field(default_factory=dict)
