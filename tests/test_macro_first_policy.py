from orchestrator.agents.ranker import _apply_macro_first_policy
from schemas.action_ir import ActionIR, VerificationPlan
from schemas.ranking_ir import RankedAction


def _patch_action(action_id: str, mechanism: str) -> RankedAction:
    action = ActionIR(
        action_id=action_id,
        family="source_patch",
        description=action_id,
        applies_to=["source_patch"],
        parameters={
            "graph_mechanism": mechanism,
            "patch_family": f"source_patch:{mechanism}",
            "expected_gain_p50": 0.08,
            "implementation_cost": 3.0,
        },
        expected_effect=["compute_opt"],
        risk_level="medium",
        verification_plan=VerificationPlan(gates=["runtime"]),
    )
    return RankedAction(action=action, score=1.0, score_breakdown={})


def test_macro_first_pushes_micro_out_of_top2_when_macro_exists() -> None:
    ranked = [
        _patch_action("micro_1", "micro_opt"),
        _patch_action("macro_vec", "vectorization"),
        _patch_action("macro_alg", "algorithmic"),
        _patch_action("micro_2", "micro_opt"),
    ]

    out = _apply_macro_first_policy(ranked, tested_actions=[], rank_cfg={})
    top2 = [item.action.parameters.get("graph_mechanism") for item in out[:2]]

    assert top2[0] in {"vectorization", "algorithmic", "data_layout", "memory_path"}
    assert top2[1] in {"vectorization", "algorithmic", "data_layout", "memory_path"}


def test_macro_first_does_not_reorder_when_no_macro_candidates() -> None:
    ranked = [
        _patch_action("micro_1", "micro_opt"),
        _patch_action("micro_2", "micro_opt"),
    ]

    out = _apply_macro_first_policy(ranked, tested_actions=[], rank_cfg={})

    assert [item.action.action_id for item in out] == ["micro_1", "micro_2"]
