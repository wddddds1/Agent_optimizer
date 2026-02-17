import json
from pathlib import Path
from types import SimpleNamespace

from orchestrator.graph import (
    _bootstrap_phase1_cache_from_history,
    _load_phase1_cache,
    _lookup_phase1_cached_action,
    _phase1_cache_key,
    _record_phase1_cache_entry,
)
from schemas.action_ir import ActionIR
from schemas.job_ir import Budgets, JobIR


def _job(case_id: str = "bwa_chr1") -> JobIR:
    return JobIR(
        app="bwa",
        case_id=case_id,
        workdir="/tmp",
        app_bin="/usr/bin/true",
        input_script="",
        env={},
        run_args=[],
        budgets=Budgets(max_iters=1, max_runs=1, max_wall_seconds=1),
    )


def _run_action(action_id: str = "runtime_lib.disable_nano") -> ActionIR:
    return ActionIR(
        action_id=action_id,
        family="runtime_lib",
        description="cached phase1 action",
        applies_to=["run_config"],
        parameters={"env": {"MALLOC_ARENA_MAX": "1"}},
        expected_effect=["compute_opt"],
    )


def test_phase1_cache_key_is_app_case() -> None:
    assert _phase1_cache_key(_job("foo")) == "bwa:foo"


def test_phase1_cache_record_and_lookup(tmp_path: Path) -> None:
    cache_path = tmp_path / "phase1_cache.json"
    payload = _load_phase1_cache(cache_path)
    action = _run_action("runtime_lib.disable_nano")

    baseline_exp = SimpleNamespace(results=SimpleNamespace(runtime_seconds=10.0))
    tuned_exp = SimpleNamespace(
        action=action,
        run_id="phase1-runtime_lib.disable_nano",
        results=SimpleNamespace(runtime_seconds=5.0),
    )
    job = _job("bwa_chr1")

    _record_phase1_cache_entry(cache_path, payload, job, baseline_exp, tuned_exp)
    assert cache_path.exists()

    loaded = _load_phase1_cache(cache_path)
    loaded_action, entry = _lookup_phase1_cached_action([], job, loaded)
    assert loaded_action is not None
    assert loaded_action.action_id == action.action_id
    assert isinstance(entry, dict)
    assert entry.get("best_run_id") == "phase1-runtime_lib.disable_nano"


def test_phase1_cache_prefers_current_action_definition(tmp_path: Path) -> None:
    cache_path = tmp_path / "phase1_cache.json"
    payload = _load_phase1_cache(cache_path)
    cached_action = _run_action("runtime_lib.disable_nano")
    baseline_exp = SimpleNamespace(results=SimpleNamespace(runtime_seconds=10.0))
    tuned_exp = SimpleNamespace(
        action=cached_action,
        run_id="phase1-runtime_lib.disable_nano",
        results=SimpleNamespace(runtime_seconds=9.0),
    )
    job = _job("bwa_chr1")
    _record_phase1_cache_entry(cache_path, payload, job, baseline_exp, tuned_exp)

    current_action = _run_action("runtime_lib.disable_nano")
    current_action.parameters["env"] = {"MALLOC_ARENA_MAX": "2"}
    loaded = _load_phase1_cache(cache_path)
    picked, _ = _lookup_phase1_cached_action([current_action], job, loaded)
    assert picked is not None
    assert picked.parameters.get("env") == {"MALLOC_ARENA_MAX": "2"}


def test_phase1_cache_bootstrap_from_best_state(tmp_path: Path) -> None:
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir(parents=True, exist_ok=True)
    history_session = sessions_root / "20260213-010101"
    history_session.mkdir(parents=True, exist_ok=True)
    action = _run_action("runtime_lib.disable_nano")
    (history_session / "best_state.json").write_text(
        json.dumps(
            {
                "case_id": "bwa_chr1",
                "best_run_id": "phase1-runtime_lib.disable_nano",
                "best_run_action": action.model_dump(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    current_session = sessions_root / "20260213-020202"
    current_session.mkdir(parents=True, exist_ok=True)
    cache_path = sessions_root / "knowledge" / "phase1_cache.json"
    payload = _load_phase1_cache(cache_path)
    job = _job("bwa_chr1")

    picked, entry = _bootstrap_phase1_cache_from_history(
        artifacts_dir=current_session,
        actions=[],
        job=job,
        cache_path=cache_path,
        cache_payload=payload,
    )
    assert picked is not None
    assert picked.action_id == "runtime_lib.disable_nano"
    assert isinstance(entry, dict)
    assert entry.get("source") == "best_state_bootstrap"
