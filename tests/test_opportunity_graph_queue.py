from orchestrator.graph import (
    _maybe_stop_on_opportunity_graph_exhausted,
    _seed_graph_actions_for_iteration,
)
from schemas.action_ir import ActionIR, VerificationPlan


def _graph_action(action_id: str, deep_id: str) -> ActionIR:
    return ActionIR(
        action_id=action_id,
        family="source_patch",
        description=action_id,
        applies_to=["source_patch"],
        parameters={"deep_analysis_id": deep_id},
        expected_effect=["compute_opt"],
        risk_level="medium",
        verification_plan=VerificationPlan(gates=["runtime"]),
    )


def test_seed_graph_actions_skips_failed_and_blocked() -> None:
    actions = [
        _graph_action("a1", "d1"),
        _graph_action("a2", "d2"),
        _graph_action("a3", "d3"),
    ]
    trace_events = []
    generated = _seed_graph_actions_for_iteration(
        deep_analysis_opportunities=actions,
        tested_actions=[],
        use_batch_selection=False,
        batch_min=1,
        batch_max=3,
        succeeded_ids=set(),
        failed_ids={"d1"},
        blocked_action_ids={"a2"},
        iteration=1,
        trace_events=trace_events,
    )
    assert [item.action_id for item in generated] == ["a3"]


def test_opportunity_graph_exhausted_when_only_failed_or_blocked_remaining() -> None:
    actions = [
        _graph_action("a1", "d1"),
        _graph_action("a2", "d2"),
    ]
    trace_events = []
    should_stop = _maybe_stop_on_opportunity_graph_exhausted(
        phase="PATCH",
        opportunity_graph_mode=True,
        generated_actions=[],
        deep_analysis_opportunities=actions,
        tested_actions=[],
        failed_ids={"d1"},
        blocked_action_ids={"a2"},
        iteration=2,
        trace_events=trace_events,
        reporter=None,
    )
    assert should_stop is True
    assert trace_events[-1]["event"] == "opportunity_graph_exhausted"

