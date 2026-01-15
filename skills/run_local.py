from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import psutil

from schemas.job_ir import JobIR


@dataclass
class RunOutput:
    runtime_seconds: float
    exit_code: int
    stdout_path: str
    stderr_path: str
    log_path: str
    time_output_path: Optional[str]
    system_metrics: Dict[str, float]
    samples: List[float]


def _extract_log_path(run_args: List[str], workdir: Path) -> Path:
    if "-log" in run_args:
        idx = run_args.index("-log")
        if idx + 1 < len(run_args):
            return (workdir / run_args[idx + 1]).resolve()
    return (workdir / "log.lammps").resolve()


def _monitor_process(proc: psutil.Process, interval: float, metrics: Dict[str, float]) -> None:
    cpu_samples: List[float] = []
    rss_samples: List[float] = []
    while True:
        if not proc.is_running():
            break
        try:
            cpu_samples.append(proc.cpu_percent(interval=interval))
            rss_samples.append(proc.memory_info().rss)
        except psutil.Error:
            break
    if cpu_samples:
        metrics["cpu_percent_avg"] = sum(cpu_samples) / len(cpu_samples)
    if rss_samples:
        metrics["rss_mb_avg"] = sum(rss_samples) / len(rss_samples) / (1024.0 * 1024.0)


def run_job(
    job: JobIR,
    run_args: List[str],
    env_overrides: Dict[str, str],
    workdir: Path,
    artifacts_dir: Path,
    time_command: Optional[str],
    repeats: int = 1,
    monitor_interval: float = 0.5,
) -> RunOutput:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = artifacts_dir / "stdout.log"
    stderr_path = artifacts_dir / "stderr.log"
    time_output_path = artifacts_dir / "time.log"
    if repeats > 1:
        stdout_path = artifacts_dir / "stdout_0.log"
        stderr_path = artifacts_dir / "stderr_0.log"
        time_output_path = artifacts_dir / "time_0.log"

    env = os.environ.copy()
    env.update(job.env)
    env.update(env_overrides)

    runtime_samples: List[float] = []
    exit_code = 0
    system_metrics: Dict[str, float] = {}

    for idx in range(repeats):
        stdout_file = stdout_path if repeats == 1 else artifacts_dir / f"stdout_{idx}.log"
        stderr_file = stderr_path if repeats == 1 else artifacts_dir / f"stderr_{idx}.log"
        time_file = time_output_path if repeats == 1 else artifacts_dir / f"time_{idx}.log"

        cmd = [job.lammps_bin] + run_args
        if time_command:
            cmd = time_command.split() + cmd

        start = time.monotonic()
        with stdout_file.open("w", encoding="utf-8") as out, stderr_file.open(
            "w", encoding="utf-8"
        ) as err:
            proc = psutil.Popen(cmd, cwd=str(workdir), env=env, stdout=out, stderr=err)
            ps_proc = psutil.Process(proc.pid)
            _monitor_process(ps_proc, monitor_interval, system_metrics)
            exit_code = proc.wait()
        end = time.monotonic()
        runtime_samples.append(end - start)

        if time_command:
            if time_file != stderr_file:
                try:
                    time_file.write_text(stderr_file.read_text(encoding="utf-8"), encoding="utf-8")
                except FileNotFoundError:
                    pass

    log_path = _extract_log_path(run_args, workdir)

    mean_runtime = sum(runtime_samples) / len(runtime_samples) if runtime_samples else 0.0
    return RunOutput(
        runtime_seconds=mean_runtime,
        exit_code=exit_code,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        log_path=str(log_path),
        time_output_path=str(time_output_path) if time_command else None,
        system_metrics=system_metrics,
        samples=runtime_samples,
    )
