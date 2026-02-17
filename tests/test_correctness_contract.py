from __future__ import annotations

from pathlib import Path

from schemas.action_ir import ActionIR, VerificationPlan
from schemas.experiment_ir import ExperimentIR
from schemas.job_ir import Budgets, JobIR
from schemas.profile_report import ProfileReport
from schemas.result_ir import ResultIR
from skills.verify import verify_run


def _gates() -> dict:
    return {
        "runtime": {"require_exit_code_zero": True, "error_regex": "ERROR"},
        "correctness": {
            "baseline_require_thermo": True,
            "agentic": {"enabled": True, "mode": "agent_only"},
            "scalar_thresholds": {"abs": 1e-6, "rel": 1e-5},
            "series_compare": {"window": 10, "abs_max": 1e-5, "rel_max": 1e-4},
        },
        "variance": {"cv_max": 0.03},
    }


def _job(tmp_path: Path) -> JobIR:
    return JobIR(
        app="bwa",
        case_id="bwa_case",
        workdir=str(tmp_path),
        app_bin="/bin/true",
        input_script="",
        env={},
        run_args=["mem", "-t", "1", "chr1.fa", "reads.fq"],
        budgets=Budgets(max_iters=1, max_runs=1, max_wall_seconds=1),
        tags=[],
    )


def _action() -> ActionIR:
    return ActionIR(
        action_id="parallel_pthread.t4_template",
        family="parallel_pthread",
        description="",
        applies_to=["run_config"],
        parameters={},
        preconditions=[],
        constraints=[],
        expected_effect=["compute_opt"],
        risk_level="low",
        verification_plan=VerificationPlan(gates=["runtime", "correctness"]),
    )


def _result(runtime: float = 1.0) -> ResultIR:
    return ResultIR(
        runtime_seconds=runtime,
        derived_metrics={},
        correctness_metrics={},
        logs=[],
        exit_code=0,
        samples=[runtime, runtime],
    )


def _baseline_exp(job: JobIR, baseline_log: Path) -> ExperimentIR:
    profile = ProfileReport(timing_breakdown={}, system_metrics={}, notes=[], log_path=str(baseline_log))
    return ExperimentIR(
        exp_id="baseline",
        job=job,
        action=None,
        git_commit_before=None,
        git_commit_after=None,
        patch_path=None,
        run_id="baseline",
        timestamps=[],
        profile_report=profile,
        results=_result(),
        verdict="PASS",
        reasons=[],
    )


def _sam_text(total: int = 12, unmapped_every: int = 0) -> str:
    rows = []
    seq = "A" * 50
    qual = "I" * 50
    for i in range(total):
        flag = 4 if unmapped_every and i % unmapped_every == 0 else 0
        rname = "*" if flag & 4 else "chr1"
        pos = 0 if flag & 4 else 100 + i
        mapq = 0 if flag & 4 else 60
        cigar = "*" if flag & 4 else "50M"
        nm = 0 if flag & 4 else 1
        rows.append(
            f"read{i}\t{flag}\t{rname}\t{pos}\t{mapq}\t{cigar}\t*\t0\t0\t{seq}\t{qual}\tNM:i:{nm}"
        )
    return "\n".join(rows) + "\n"


def test_baseline_records_generic_contract(tmp_path: Path) -> None:
    job = _job(tmp_path)
    gates = _gates()
    baseline_log = tmp_path / "baseline.sam"
    baseline_log.write_text(_sam_text(), encoding="utf-8")
    profile = ProfileReport(timing_breakdown={}, system_metrics={}, notes=[], log_path=str(baseline_log))
    cache: dict = {}

    def putter(j: JobIR, contract: dict) -> None:
        cache["contract"] = contract

    verdict = verify_run(
        job=job,
        action=None,
        result=_result(),
        profile=profile,
        gates=gates,
        baseline_exp=None,
        contract_putter=putter,
    )
    assert verdict.verdict == "PASS"
    assert "contract" in cache
    assert cache["contract"]["kind"] == "generic_signature_v1"


def test_candidate_uses_cached_contract_for_deterministic_check(tmp_path: Path) -> None:
    job = _job(tmp_path)
    gates = _gates()
    baseline_log = tmp_path / "baseline.sam"
    run_log = tmp_path / "run.sam"
    baseline_log.write_text(_sam_text(), encoding="utf-8")
    run_log.write_text(_sam_text(), encoding="utf-8")
    cache: dict = {}

    # First, baseline stores contract.
    verify_run(
        job=job,
        action=None,
        result=_result(),
        profile=ProfileReport(timing_breakdown={}, system_metrics={}, notes=[], log_path=str(baseline_log)),
        gates=gates,
        baseline_exp=None,
        contract_putter=lambda j, contract: cache.update({"contract": contract}),
    )
    base_exp = _baseline_exp(job, baseline_log=tmp_path / "missing_baseline.sam")
    verdict = verify_run(
        job=job,
        action=_action(),
        result=_result(),
        profile=ProfileReport(timing_breakdown={}, system_metrics={}, notes=[], log_path=str(run_log)),
        gates=gates,
        baseline_exp=base_exp,
        contract_getter=lambda j: cache.get("contract"),
    )
    assert verdict.verdict == "PASS"
    contract_metrics = verdict.correctness_metrics.get("generic_contract", {})
    assert contract_metrics.get("status") == "PASS"


def test_candidate_mismatch_fails_by_generic_contract(tmp_path: Path) -> None:
    job = _job(tmp_path)
    gates = _gates()
    baseline_log = tmp_path / "baseline.sam"
    run_log = tmp_path / "run_bad.sam"
    baseline_log.write_text(_sam_text(), encoding="utf-8")
    # Force mismatch by introducing frequent unmapped reads.
    run_log.write_text(_sam_text(unmapped_every=2), encoding="utf-8")
    base_exp = _baseline_exp(job, baseline_log=baseline_log)

    verdict = verify_run(
        job=job,
        action=_action(),
        result=_result(),
        profile=ProfileReport(timing_breakdown={}, system_metrics={}, notes=[], log_path=str(run_log)),
        gates=gates,
        baseline_exp=base_exp,
    )
    assert verdict.verdict == "FAIL"
    assert any("generic correctness mismatch" in reason for reason in verdict.reasons)


def test_unsure_can_be_adjudicated_by_agent(tmp_path: Path) -> None:
    job = _job(tmp_path)
    gates = _gates()
    run_log = tmp_path / "run.sam"
    run_log.write_text(_sam_text(), encoding="utf-8")

    def decider(payload: dict) -> dict:
        return {
            "verdict": "PASS",
            "rationale": "fallback agent decision",
            "confidence": 0.6,
            "allowed_drift": {"policy": "agent_fallback", "notes": ""},
        }

    verdict = verify_run(
        job=job,
        action=_action(),
        result=_result(),
        profile=ProfileReport(timing_breakdown={}, system_metrics={}, notes=[], log_path=str(run_log)),
        gates=gates,
        baseline_exp=None,
        agentic_decider=decider,
        agentic_cfg={"mode": "agent_only", "enabled": True},
    )
    assert verdict.verdict == "PASS"
    agent_decision = verdict.correctness_metrics.get("agent_decision", {})
    assert agent_decision.get("verdict") == "PASS"


def test_bwa_devnull_output_treated_as_intentional_signature_suppression(tmp_path: Path) -> None:
    job = _job(tmp_path).model_copy(
        update={"run_args": ["mem", "-t", "1", "-o", "/dev/null", "chr1.fa", "reads.fq"]}
    )
    gates = _gates()
    empty_log = tmp_path / "empty.log"
    empty_log.write_text("", encoding="utf-8")
    profile = ProfileReport(timing_breakdown={}, system_metrics={}, notes=[], log_path=str(empty_log))

    baseline_verdict = verify_run(
        job=job,
        action=None,
        result=_result(),
        profile=profile,
        gates=gates,
        baseline_exp=None,
    )
    assert baseline_verdict.verdict == "PASS"
    baseline_contract = baseline_verdict.correctness_metrics.get("generic_contract", {})
    assert baseline_contract.get("status") == "PASS"
    assert baseline_contract.get("suppressed_signature") is True

    base_exp = _baseline_exp(job, baseline_log=empty_log)
    candidate_verdict = verify_run(
        job=job,
        action=_action(),
        result=_result(),
        profile=profile,
        gates=gates,
        baseline_exp=base_exp,
    )
    assert candidate_verdict.verdict == "PASS"
