from __future__ import annotations

from datetime import datetime, timezone

from orchestrator.agents.optimizer import _ensure_memory_keep
from schemas.action_ir import ActionIR
from schemas.candidate_ir import CandidateList
from schemas.experience_ir import ExperienceRecord
from skills.experience_memory import ExperienceConfig, ExperienceMemory


def _action(action_id: str, threads: int) -> ActionIR:
    return ActionIR(
        action_id=action_id,
        family="parallel_pthread",
        description="",
        applies_to=["run_config"],
        parameters={
            "run_args": {
                "set_flags": [
                    {
                        "flag": "-t",
                        "values": [str(threads)],
                    }
                ]
            }
        },
        preconditions=[],
        constraints=[],
        expected_effect=["compute_opt"],
        risk_level="low",
    )


def test_bayesian_memory_scores_are_bounded_and_prefer_strong_history() -> None:
    now = datetime.now(timezone.utc).isoformat()
    records = [
        ExperienceRecord(
            action_id="parallel_pthread.t16_template",
            family="parallel_pthread",
            outcome="PASS",
            improvement_pct=90.0,
            speedup_vs_baseline=10.0,
            case_id="bwa_chr1",
            app="bwa",
            backend=None,
            timestamp=now,
            strength="strong",
            weight=1.0,
        ),
        ExperienceRecord(
            action_id="parallel_pthread.t1_template",
            family="parallel_pthread",
            outcome="FAIL",
            improvement_pct=0.0,
            speedup_vs_baseline=1.0,
            case_id="bwa_chr1",
            app="bwa",
            backend=None,
            timestamp=now,
            strength="negative",
            weight=-0.5,
        ),
    ]
    memory = ExperienceMemory(config=ExperienceConfig(enabled=True, path=None), records=records)
    actions = [_action("parallel_pthread.t16_template", 16), _action("parallel_pthread.t1_template", 1)]

    scored = memory.score_actions(actions, {"case_id": "bwa_chr1", "app": "bwa", "backend": ""})
    assert -1.0 <= scored["parallel_pthread.t16_template"] <= 1.0
    assert -1.0 <= scored["parallel_pthread.t1_template"] <= 1.0
    assert scored["parallel_pthread.t16_template"] > scored["parallel_pthread.t1_template"]


def test_memory_keep_for_parallel_threads_does_not_drop_high_value_thread_tier() -> None:
    family = "parallel_pthread"
    base_candidates = [
        _action("parallel_pthread.t1_template", 1),
        _action("parallel_pthread.t2_template", 2),
        _action("parallel_pthread.t4_template", 4),
        _action("parallel_pthread.t8_template", 8),
        _action("parallel_pthread.t12_template", 12),
    ]
    family_actions = {
        family: [_action("parallel_pthread.t16_template", 16)] + base_candidates,
    }
    candidate_lists = [
        CandidateList(
            family=family,
            candidates=base_candidates,
            assumptions=[],
            confidence=0.6,
        )
    ]

    updated = _ensure_memory_keep(
        candidate_lists=candidate_lists,
        family_actions=family_actions,
        keep_set={"parallel_pthread.t16_template"},
        max_candidates=5,
    )
    action_ids = [a.action_id for a in updated[0].candidates]

    assert "parallel_pthread.t16_template" in action_ids
    assert "parallel_pthread.t12_template" in action_ids
    assert "parallel_pthread.t1_template" not in action_ids
