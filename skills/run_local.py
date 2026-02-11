from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import psutil

from schemas.job_ir import JobIR


# Launcher type → rank flag mapping
_LAUNCHER_RANK_FLAG = {
    "mpirun": "-np",
    "mpiexec": "-np",
    "srun": "-n",
}


def build_launch_cmd(
    binary: str,
    run_args: List[str],
    launcher_cfg: Optional[Dict[str, object]] = None,
    time_command: Optional[str] = None,
    wrapper_command: Optional[List[str]] = None,
) -> List[str]:
    """Build the full command list with optional MPI launcher prefix.

    launcher_cfg keys:
        type: "direct" | "mpirun" | "mpiexec" | "srun"
        np: int (rank count)
        extra_args: List[str] (additional launcher flags)
    """
    app_cmd = [binary] + run_args
    if not launcher_cfg or launcher_cfg.get("type", "direct") == "direct":
        prefix: List[str] = []
    else:
        launcher_type = str(launcher_cfg["type"])
        np = int(launcher_cfg.get("np", 1))
        extra = list(launcher_cfg.get("extra_args", []))
        rank_flag = _LAUNCHER_RANK_FLAG.get(launcher_type, "-np")
        prefix = [launcher_type, rank_flag, str(np)] + extra
    wrapper = wrapper_command or []
    cmd = prefix + wrapper + app_cmd
    if time_command:
        cmd = time_command.split() + cmd
    return cmd


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


def _monitor_process(
    proc: psutil.Process,
    interval: float,
    metrics: Dict[str, float],
    stop_event: threading.Event,
) -> None:
    cpu_samples: List[float] = []
    rss_samples: List[float] = []
    while not stop_event.is_set():
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


def _select_monitor_process(
    parent: psutil.Process,
    prefer_child: bool,
    target_bin: str,
    timeout_s: float = 2.0,
    poll_s: float = 0.05,
) -> psutil.Process:
    if not prefer_child:
        return parent
    target_name = Path(target_bin).name
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            children = parent.children(recursive=False)
        except psutil.Error:
            break
        if children:
            for child in children:
                try:
                    exe = child.exe()
                except psutil.Error:
                    exe = ""
                name = ""
                try:
                    name = child.name()
                except psutil.Error:
                    name = ""
                if target_name and (target_name in exe or target_name in name):
                    return child
            return children[0]
        time.sleep(poll_s)
    return parent


def run_job(
    job: JobIR,
    run_args: List[str],
    env_overrides: Dict[str, str],
    workdir: Path,
    artifacts_dir: Path,
    time_command: Optional[str],
    wrapper_command: Optional[List[str]] = None,
    repeats: int = 1,
    monitor_interval: float = 0.5,
    launcher_cfg: Optional[Dict[str, object]] = None,
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

        cmd = build_launch_cmd(
            job.lammps_bin,
            run_args,
            launcher_cfg,
            time_command,
            wrapper_command,
        )
        _has_launcher = bool(
            launcher_cfg and launcher_cfg.get("type", "direct") != "direct"
        )

        start = time.monotonic()
        with stdout_file.open("w", encoding="utf-8") as out, stderr_file.open(
            "w", encoding="utf-8"
        ) as err:
            proc = psutil.Popen(cmd, cwd=str(workdir), env=env, stdout=out, stderr=err)
            ps_proc = psutil.Process(proc.pid)
            monitor_proc = _select_monitor_process(
                ps_proc,
                prefer_child=bool(time_command) or _has_launcher,
                target_bin=job.lammps_bin,
            )
            stop_event = threading.Event()
            monitor_thread = threading.Thread(
                target=_monitor_process,
                args=(monitor_proc, monitor_interval, system_metrics, stop_event),
                daemon=True,
            )
            monitor_thread.start()
            exit_code = proc.wait()
            stop_event.set()
            monitor_thread.join(timeout=monitor_interval * 2 if monitor_interval else 0.1)
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
