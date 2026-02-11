from schemas.action_ir import ActionIR, VerificationPlan
from skills.patch_triage import validate_patch_action


def test_math_equivalence_requires_strict_correctness():
    action = ActionIR(
        action_id="patch_math",
        family="math_equivalence",
        description="",
        applies_to=["source_patch"],
        parameters={"patch_family": "math_equivalence"},
        expected_effect=["compute_opt"],
        risk_level="high",
        verification_plan=VerificationPlan(gates=["runtime", "correctness", "variance"], thresholds={}),
    )
    patch_families = {"families": [{"id": "math_equivalence", "mandatory_gates": ["runtime", "correctness", "variance"]}]}
    gates = {"correctness": {"default_mode": "relaxed"}}
    ok, reason = validate_patch_action(action, patch_families, gates, strict_required=False)
    assert ok is False
    assert "strict" in (reason or "")
