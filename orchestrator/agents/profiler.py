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
        wrapper_command: Optional[list[str]],
        repeats: int,
        launcher_cfg: Optional[Dict[str, object]] = None,
        profiling_cfg: Optional[Dict[str, object]] = None,
        is_baseline: bool = False,
    ) -> Tuple[object, ProfileReport]:
        return profile_job(
            job=job,
            run_args=run_args,
            env_overrides=env_overrides,
            workdir=workdir,
            artifacts_dir=artifacts_dir,
            time_command=time_command,
            wrapper_command=wrapper_command,
            repeats=repeats,
            launcher_cfg=launcher_cfg,
            profiling_cfg=profiling_cfg,
            is_baseline=is_baseline,
        )
