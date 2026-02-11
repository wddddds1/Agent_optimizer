from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from schemas.profile_report import ProfileReport
from skills.metrics_parse import parse_lammps_timing, parse_time_output
from skills.run_local import RunOutput, run_job
from schemas.job_ir import JobIR


def profile_job(
    job: JobIR,
    run_args: list[str],
    env_overrides: Dict[str, str],
    workdir: Path,
    artifacts_dir: Path,
    time_command: Optional[str],
    wrapper_command: Optional[list[str]] = None,
    repeats: int = 1,
    launcher_cfg: Optional[Dict[str, object]] = None,
) -> tuple[RunOutput, ProfileReport]:
    run_output = run_job(
        job=job,
        run_args=run_args,
        env_overrides=env_overrides,
        workdir=workdir,
        artifacts_dir=artifacts_dir,
        time_command=time_command,
        wrapper_command=wrapper_command,
        repeats=repeats,
        launcher_cfg=launcher_cfg,
    )

    log_text = ""
    try:
        log_text = Path(run_output.log_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        pass

    time_text = ""
    if run_output.time_output_path:
        try:
            time_text = Path(run_output.time_output_path).read_text(encoding="utf-8")
        except FileNotFoundError:
            pass

    timing_breakdown = parse_lammps_timing(log_text) if job.app == "lammps" else {}
    system_metrics = parse_time_output(time_text)
    system_metrics.update(run_output.system_metrics)

    report = ProfileReport(
        timing_breakdown=timing_breakdown,
        system_metrics=system_metrics,
        notes=[],
        log_path=run_output.log_path,
    )
    return run_output, report
