from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from schemas.action_ir import ActionIR
from schemas.job_ir import JobIR
from schemas.profile_report import ProfileReport
from schemas.result_ir import ResultIR


class ExperimentIR(BaseModel):
    exp_id: str
    parent_exp_id: Optional[str] = None
    job: JobIR
    action: Optional[ActionIR] = None
    git_commit_before: Optional[str] = None
    git_commit_after: Optional[str] = None
    patch_path: Optional[str] = None
    run_id: str
    timestamps: List[str] = Field(default_factory=list)
    profile_report: ProfileReport
    results: ResultIR
    verdict: Literal["PASS", "FAIL"]
    reasons: List[str] = Field(default_factory=list)
