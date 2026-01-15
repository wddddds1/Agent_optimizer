from pathlib import Path

from schemas.action_ir import ActionIR, VerificationPlan
from schemas.profile_report import ProfileReport
from schemas.result_ir import ResultIR
from skills.verify import verify_run


def _gates():
    return {
        "runtime": {"require_exit_code_zero": True, "error_regex": "ERROR"},
        "correctness": {"allow_skip_for_low_risk_run_config": True, "scalar_thresholds": {"abs": 1e-6, "rel": 1e-5}},
        "variance": {"cv_max": 0.03},
    }


def test_runtime_gate_pass():
    profile = ProfileReport(
        timing_breakdown={},
        system_metrics={},
        notes=[],
        log_path="examples/sample_lammps_log/log.lammps",
    )
    result = ResultIR(
        runtime_seconds=1.0,
        derived_metrics={},
        correctness_metrics={},
        logs=[],
        exit_code=0,
        samples=[1.0, 1.0],
    )
    action = ActionIR(
        action_id="omp_threads_2",
        family="omp_threads",
        description="",
        applies_to=["run_config"],
        parameters={},
        preconditions=[],
        constraints=[],
        expected_effect=["compute_opt"],
        risk_level="low",
        verification_plan=VerificationPlan(gates=["runtime"]),
    )
    verdict = verify_run(action, result, profile, _gates(), None)
    assert verdict.verdict == "PASS"
    assert "correctness_skipped_reason" in verdict.correctness_metrics


def test_correctness_requires_baseline_for_input_edit():
    profile = ProfileReport(
        timing_breakdown={},
        system_metrics={},
        notes=[],
        log_path="examples/sample_lammps_log/log.lammps",
    )
    result = ResultIR(
        runtime_seconds=1.0,
        derived_metrics={},
        correctness_metrics={},
        logs=[],
        exit_code=0,
        samples=[1.0, 1.0],
    )
    action = ActionIR(
        action_id="neigh_every_2",
        family="neigh_modify",
        description="",
        applies_to=["input_script"],
        parameters={},
        preconditions=[],
        constraints=[],
        expected_effect=["mem_locality"],
        risk_level="medium",
        verification_plan=VerificationPlan(gates=["runtime", "correctness"]),
    )
    verdict = verify_run(action, result, profile, _gates(), None)
    assert verdict.verdict == "FAIL"
