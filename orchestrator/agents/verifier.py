from __future__ import annotations

from typing import Dict, Optional

from schemas.action_ir import ActionIR
from schemas.experiment_ir import ExperimentIR
from schemas.job_ir import JobIR
from schemas.profile_report import ProfileReport
from schemas.result_ir import ResultIR
from skills.verify import verify_run


class VerifierAgent:
    def verify(
        self,
        job: JobIR,
        action: Optional[ActionIR],
        result: ResultIR,
        profile: ProfileReport,
        gates: Dict[str, object],
        baseline_exp: Optional[ExperimentIR],
    ):
        return verify_run(job, action, result, profile, gates, baseline_exp)
