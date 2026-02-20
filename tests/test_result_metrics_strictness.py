from types import SimpleNamespace

from orchestrator.graph import _build_result_ir
from schemas.profile_report import ProfileReport


def _run_output(runtime: float, exit_code: int) -> SimpleNamespace:
    return SimpleNamespace(
        runtime_seconds=runtime,
        exit_code=exit_code,
        stdout_path="stdout.log",
        stderr_path="stderr.log",
        log_path="run.log",
        samples=[runtime],
    )


def test_build_result_ir_omits_speedup_for_failed_run() -> None:
    profile = ProfileReport(timing_breakdown={}, system_metrics={}, notes=[])
    result = _build_result_ir(
        _run_output(runtime=0.5, exit_code=-11),
        profile,
        baseline_runtime=10.0,
        runtime_agg="mean",
        prior_samples=None,
    )
    assert "speedup_vs_baseline" not in result.derived_metrics


def test_build_result_ir_keeps_speedup_for_pass_run() -> None:
    profile = ProfileReport(timing_breakdown={}, system_metrics={}, notes=[])
    result = _build_result_ir(
        _run_output(runtime=5.0, exit_code=0),
        profile,
        baseline_runtime=10.0,
        runtime_agg="mean",
        prior_samples=None,
    )
    assert result.derived_metrics.get("speedup_vs_baseline") == 2.0
