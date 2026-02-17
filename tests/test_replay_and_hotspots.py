from pathlib import Path

from orchestrator.agents.planner import _analysis_confidence
from orchestrator.graph import _load_patch_replay_actions, _resolve_patch_family_hint
from schemas.action_ir import ActionIR, VerificationPlan
from schemas.experiment_ir import ExperimentIR
from schemas.job_ir import Budgets, JobIR
from schemas.profile_report import ProfileReport
from schemas.result_ir import ResultIR


def _job(case_id: str = "case") -> JobIR:
    return JobIR(
        app="bwa",
        case_id=case_id,
        workdir="/tmp",
        app_bin="/usr/bin/true",
        input_script="",
        budgets=Budgets(max_iters=1, max_runs=1, max_wall_seconds=1),
    )


def _exp(run_id: str, action: ActionIR, patch_path: str, case_id: str = "case") -> ExperimentIR:
    return ExperimentIR(
        exp_id=run_id,
        run_id=run_id,
        job=_job(case_id),
        action=action,
        patch_path=patch_path,
        profile_report=ProfileReport(timing_breakdown={}, system_metrics={}, notes=[]),
        results=ResultIR(
            runtime_seconds=1.0,
            exit_code=0,
            derived_metrics={"speedup_vs_baseline": 1.2},
            correctness_metrics={},
            logs=[],
        ),
        verdict="PASS",
        reasons=[],
    )


def test_analysis_confidence_uses_tau_hotspots_when_timing_missing() -> None:
    profile = ProfileReport(
        timing_breakdown={},
        system_metrics={},
        notes=[],
        tau_hotspots=[{"function": "foo", "exclusive_us": 10.0}],
    )
    assert _analysis_confidence(profile) >= 0.7


def test_resolve_patch_family_hint_maps_common_aliases() -> None:
    cfg = {
        "families": [
            {"id": "vectorization_hints"},
            {"id": "array_packing"},
            {"id": "loop_unroll"},
        ]
    }
    assert _resolve_patch_family_hint(cfg, "SIMD") == "vectorization_hints"
    assert _resolve_patch_family_hint(cfg, "Memory Layout") == "array_packing"
    assert _resolve_patch_family_hint(cfg, "unroll loops") == "loop_unroll"


def test_load_patch_replay_actions_skips_replay_of_replay_and_compacts_id(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions" / "s1" / "runs"
    sessions.mkdir(parents=True, exist_ok=True)
    patch_file = tmp_path / "p.diff"
    patch_file.write_text("diff --git a/a.c b/a.c\n", encoding="utf-8")

    base_action = ActionIR(
        action_id="generated.source_patch.loop_unroll.bwa_c_1",
        family="source_patch",
        description="x",
        applies_to=["source_patch"],
        parameters={"patch_family": "loop_unroll"},
        expected_effect=["compute_opt"],
        risk_level="medium",
        verification_plan=VerificationPlan(gates=["runtime"]),
    )
    replay_action = ActionIR(
        action_id="replay.generated.source_patch.loop_unroll.bwa_c_1",
        family="source_patch",
        description="x",
        applies_to=["source_patch"],
        parameters={"patch_family": "loop_unroll", "origin": "memory_replay"},
        expected_effect=["compute_opt"],
        risk_level="medium",
        verification_plan=VerificationPlan(gates=["runtime"]),
    )

    exp_a = _exp("iter1-a", base_action, str(patch_file))
    exp_b = _exp("iter1-b", replay_action, str(patch_file))
    run_a = sessions / "iter1-a"
    run_b = sessions / "iter1-b"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)
    (run_a / "experiment.json").write_text(exp_a.model_dump_json(indent=2), encoding="utf-8")
    (run_b / "experiment.json").write_text(exp_b.model_dump_json(indent=2), encoding="utf-8")

    out = _load_patch_replay_actions(
        artifacts_root=tmp_path,
        case_id="case",
        app_name="bwa",
        backend=None,
        min_gain_pct=1.0,
        max_actions=5,
        existing_action_ids=set(),
        patch_families_cfg={"families": [{"id": "loop_unroll", "patch_tags": ["compute_opt"]}]},
    )
    assert len(out) == 1
    assert out[0].action_id.startswith("replay.")
    assert "replay.replay" not in out[0].action_id
    assert len(out[0].action_id) < 128


def test_load_patch_replay_actions_falls_back_to_same_app_when_case_missing(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions" / "s1" / "runs"
    sessions.mkdir(parents=True, exist_ok=True)
    patch_file = tmp_path / "p2.diff"
    patch_file.write_text("diff --git a/a.c b/a.c\n", encoding="utf-8")
    action = ActionIR(
        action_id="generated.source_patch.loop_unroll.bwa_c_1",
        family="source_patch",
        description="x",
        applies_to=["source_patch"],
        parameters={"patch_family": "loop_unroll"},
        expected_effect=["compute_opt"],
        risk_level="medium",
        verification_plan=VerificationPlan(gates=["runtime"]),
    )
    exp = _exp("iter1-a", action, str(patch_file), case_id="bwa_chr1")
    run = sessions / "iter1-a"
    run.mkdir(parents=True, exist_ok=True)
    (run / "experiment.json").write_text(exp.model_dump_json(indent=2), encoding="utf-8")

    out = _load_patch_replay_actions(
        artifacts_root=tmp_path,
        case_id="bwa_ecoli",
        app_name="bwa",
        backend=None,
        min_gain_pct=1.0,
        max_actions=3,
        existing_action_ids=set(),
        patch_families_cfg={"families": [{"id": "loop_unroll", "patch_tags": ["compute_opt"]}]},
    )
    assert len(out) == 1
