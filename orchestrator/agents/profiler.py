from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

from schemas.job_ir import JobIR
from schemas.profile_report import ProfileReport
from skills.profiling_local import profile_job


class ProfilerAgent:
    def run(
        self,
        job: JobIR,
        run_args: list[str],
        env_overrides: Dict[str, str],
        workdir: Path,
        artifacts_dir: Path,
        time_command: Optional[str],
        repeats: int,
    ) -> Tuple[object, ProfileReport]:
        return profile_job(
            job=job,
            run_args=run_args,
            env_overrides=env_overrides,
            workdir=workdir,
            artifacts_dir=artifacts_dir,
            time_command=time_command,
            repeats=repeats,
        )
