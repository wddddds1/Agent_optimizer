from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from schemas.job_ir import JobIR
from schemas.profile_report import ProfileReport
from skills.applications import parse_timing_breakdown as app_parse_timing_breakdown
from skills.metrics_parse import (
    parse_tau_profile,
    parse_time_output,
    parse_xctrace_report,
    parse_xctrace_time_profile_xml,
)
from skills.run_local import RunOutput, build_launch_cmd, run_job


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
    profiling_cfg: Optional[Dict[str, object]] = None,
    is_baseline: bool = False,
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

    timing_breakdown = app_parse_timing_breakdown(job.app, log_text)
    system_metrics = parse_time_output(time_text)
    system_metrics.update(run_output.system_metrics)

    notes: List[str] = []
    tau_hotspots: List[Dict[str, object]] = []

    tau_dir = artifacts_dir / "tau"
    tau_empty_note = (
        "tau hotspot profile missing/empty (check TAU runtime config or instrumentation)"
    )
    if tau_dir.is_dir():
        tau_hotspots = parse_tau_profile(str(tau_dir))
        if not tau_hotspots:
            notes.append(tau_empty_note)

    xctrace_hotspots: List[Dict[str, object]] = []
    xctrace_note = _maybe_run_xctrace(
        job=job,
        run_args=run_args,
        env_overrides=env_overrides,
        workdir=workdir,
        artifacts_dir=artifacts_dir,
        launcher_cfg=launcher_cfg,
        profiling_cfg=profiling_cfg,
        is_baseline=is_baseline,
    )
    if xctrace_note:
        notes.append(xctrace_note)

    xctrace_cfg = profiling_cfg.get("xctrace", {}) if isinstance(profiling_cfg, dict) else {}
    xctrace_subdir = str(xctrace_cfg.get("profile_subdir", "xctrace"))
    xctrace_report = artifacts_dir / xctrace_subdir / "report.xml"
    if xctrace_report.is_file():
        try:
            report_text = xctrace_report.read_text(encoding="utf-8", errors="replace")
            top_n = xctrace_cfg.get("top_n", 30)
            try:
                top_n = int(top_n)
            except (TypeError, ValueError):
                top_n = 30
            xctrace_hotspots = parse_xctrace_report(report_text, top_n=top_n)
            if not xctrace_hotspots:
                xctrace_hotspots = parse_xctrace_time_profile_xml(report_text, top_n=top_n)
        except Exception:
            xctrace_hotspots = []
        if xctrace_hotspots:
            tau_hotspots = xctrace_hotspots
            if tau_empty_note in notes:
                notes = [n for n in notes if n != tau_empty_note]
            notes.append(f"xctrace_report={xctrace_report}")

    if not tau_hotspots:
        notes.append(
            "no function hotspots available; decisions rely on wall-clock/system metrics"
        )

    report = ProfileReport(
        timing_breakdown=timing_breakdown,
        system_metrics=system_metrics,
        notes=notes,
        log_path=run_output.log_path,
        tau_hotspots=tau_hotspots,
    )
    return run_output, report


def _maybe_run_xctrace(
    job: JobIR,
    run_args: list[str],
    env_overrides: Dict[str, str],
    workdir: Path,
    artifacts_dir: Path,
    launcher_cfg: Optional[Dict[str, object]],
    profiling_cfg: Optional[Dict[str, object]],
    is_baseline: bool,
) -> Optional[str]:
    cfg = profiling_cfg.get("xctrace", {}) if isinstance(profiling_cfg, dict) else {}
    if not isinstance(cfg, dict):
        return None
    if not cfg.get("enabled", False):
        return None
    # Default to baseline-only profiling to avoid exploding phase-1 cost.
    # Users can explicitly set baseline_only: false when they want full tracing.
    if cfg.get("baseline_only", True) and not is_baseline:
        return None
    if platform.system().lower() != "darwin":
        return "xctrace skipped: non-macos"
    if not shutil.which("xcrun"):
        return "xctrace skipped: xcrun not found (install Xcode Command Line Tools)"
    if not _resolve_xctrace_path():
        return (
            "xctrace skipped: utility unavailable from active developer dir "
            "(run: sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer)"
        )

    template = str(cfg.get("template", "Time Profiler"))
    timeout_seconds = cfg.get("timeout_seconds", 1800)
    try:
        timeout_seconds = max(30, int(timeout_seconds))
    except (TypeError, ValueError):
        timeout_seconds = 1800

    record_args = cfg.get("record_args", [])
    export_args = cfg.get("export_args", [])
    if not isinstance(record_args, list):
        record_args = []
    if not isinstance(export_args, list):
        export_args = []

    trace_subdir = str(cfg.get("profile_subdir", "xctrace"))
    trace_dir = artifacts_dir / trace_subdir
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "trace.trace"
    report_path = trace_dir / "report.xml"
    stdout_path = trace_dir / "xctrace_stdout.log"
    stderr_path = trace_dir / "xctrace_stderr.log"

    cmd = build_launch_cmd(
        job.app_bin,
        run_args,
        launcher_cfg=launcher_cfg,
        time_command=None,
        wrapper_command=None,
    )
    record_cmd = [
        "xcrun",
        "xctrace",
        "record",
        *record_args,
        "--template",
        template,
        "--output",
        str(trace_path),
        "--launch",
        "--",
        *cmd,
    ]

    env = os.environ.copy()
    env.update(job.env)
    for key, value in env_overrides.items():
        if not key.startswith("TAU_"):
            env[key] = value

    try:
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
            "w", encoding="utf-8"
        ) as err:
            result = subprocess.run(
                record_cmd,
                cwd=str(workdir),
                env=env,
                stdout=out,
                stderr=err,
                check=False,
                timeout=timeout_seconds,
            )
    except Exception as exc:
        return f"xctrace record failed: {exc}"

    if result.returncode != 0:
        detail = _tail_file(stderr_path, max_lines=6)
        suffix = f", detail={detail}" if detail else ""
        return (
            f"xctrace record failed (code={result.returncode}, trace={trace_path}{suffix})"
        )

    export_cmd = [
        "xcrun",
        "xctrace",
        "export",
        "--input",
        str(trace_path),
        "--xpath",
        '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]',
        *export_args,
    ]
    try:
        result = subprocess.run(
            export_cmd,
            cwd=str(workdir),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except Exception as exc:
        return f"xctrace export failed: {exc}"
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()
        detail = tail[-1].strip() if tail else ""
        suffix = f", detail={detail}" if detail else ""
        return f"xctrace export failed (code={result.returncode}, report={report_path}{suffix})"
    try:
        report_path.write_text(result.stdout or "", encoding="utf-8")
    except Exception as exc:
        return f"xctrace export write failed: {exc}"
    if not report_path.exists() or report_path.stat().st_size == 0:
        return f"xctrace export failed: empty report ({report_path})"
    return None


def _resolve_xctrace_path() -> Optional[str]:
    try:
        result = subprocess.run(
            ["xcrun", "--find", "xctrace"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    path = (result.stdout or "").strip()
    return path if path else None


def _tail_file(path: Path, max_lines: int = 8) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return " | ".join(lines[-max_lines:])
