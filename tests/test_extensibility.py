from pathlib import Path

from orchestrator.graph import _ensure_log_path, _resolve_app_repo_root
from schemas.job_ir import Budgets, JobIR
from skills.applications import (
    ensure_log_path as app_ensure_log_path,
    input_edit_allowlist,
    parse_timing_breakdown,
    requires_structured_correctness,
    supports_agentic_correctness,
)
from skills.run_local import _extract_log_path


def _job(app: str) -> JobIR:
    return JobIR(
        app=app,
        case_id="case",
        workdir="/tmp",
        app_bin="/usr/bin/true",
        input_script="",
        budgets=Budgets(max_iters=1, max_runs=1, max_wall_seconds=1),
    )


def test_job_ir_accepts_custom_app_name() -> None:
    job = _job("my_hpc_app")
    assert job.app == "my_hpc_app"


def test_resolve_app_repo_root_prefers_adapter_patch_root(tmp_path: Path) -> None:
    repo_root = tmp_path
    app_root = repo_root / "apps" / "my_hpc_app"
    app_root.mkdir(parents=True, exist_ok=True)
    job = _job("my_hpc_app")
    resolved = _resolve_app_repo_root(
        repo_root,
        job,
        {"patch_root": "apps/my_hpc_app"},
    )
    assert resolved == app_root.resolve()


def test_resolve_app_repo_root_uses_legacy_third_party_layout(tmp_path: Path) -> None:
    repo_root = tmp_path
    legacy = repo_root / "third_party" / "my_hpc_app"
    legacy.mkdir(parents=True, exist_ok=True)
    job = _job("my_hpc_app")
    resolved = _resolve_app_repo_root(repo_root, job, None)
    assert resolved == legacy.resolve()


def test_resolve_app_repo_root_falls_back_to_repo_root(tmp_path: Path) -> None:
    repo_root = tmp_path
    job = _job("unknown_app")
    resolved = _resolve_app_repo_root(repo_root, job, None)
    assert resolved == repo_root


def test_app_hooks_default_for_unknown_app(tmp_path: Path) -> None:
    args = ["--foo", "bar"]
    patched = app_ensure_log_path("my_hpc_app", args, tmp_path)
    assert patched == args
    assert patched is not args
    assert input_edit_allowlist("my_hpc_app") == []
    assert parse_timing_breakdown("my_hpc_app", "log text") == {}
    assert requires_structured_correctness("my_hpc_app") is False
    assert supports_agentic_correctness("my_hpc_app") is True


def test_lammps_log_path_hook_keeps_special_behavior(tmp_path: Path) -> None:
    args = ["-in", "in.melt"]
    patched = app_ensure_log_path("lammps", args, tmp_path)
    assert patched[:2] == ["-in", "in.melt"]
    assert patched[-2:] == ["-log", str(tmp_path / "log.lammps")]
    graph_patched = _ensure_log_path(args, tmp_path, app="lammps")
    assert graph_patched == patched


def test_extract_log_path_without_log_flag_uses_stdout_capture(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    resolved = _extract_log_path(["-in", "in.any"], tmp_path, artifacts_dir=artifacts_dir)
    assert resolved == (artifacts_dir / "stdout.log").resolve()


def test_extract_log_path_prefers_stdout_fallback_when_provided(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    stdout_capture = artifacts_dir / "stdout_0.log"
    resolved = _extract_log_path(
        ["-in", "in.any"],
        tmp_path,
        artifacts_dir=artifacts_dir,
        stdout_fallback=stdout_capture,
    )
    assert resolved == stdout_capture.resolve()
