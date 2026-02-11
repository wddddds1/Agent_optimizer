from __future__ import annotations

from typing import Dict, Tuple

from schemas.action_ir import ActionIR


def validate_patch_action(
    action: ActionIR,
    patch_families: Dict[str, object],
    gates: Dict[str, object],
    strict_required: bool,
) -> Tuple[bool, str | None]:
    if "source_patch" not in (action.applies_to or []):
        return True, None

    family_cfg = _find_patch_family(action, patch_families)
    if not family_cfg:
        return True, None

    mandatory = set(family_cfg.get("mandatory_gates", []) or [])
    provided = set(action.verification_plan.gates or [])
    if mandatory and not mandatory.issubset(provided):
        missing = ", ".join(sorted(mandatory - provided))
        return False, f"missing mandatory gates: {missing}"

    if family_cfg.get("requires_strict_correctness") or _is_math_equivalence(action):
        default_mode = str(gates.get("correctness", {}).get("default_mode", "strict"))
        if default_mode != "strict":
            return False, "strict correctness gate required for this patch family"

    return True, None


def _find_patch_family(action: ActionIR, patch_families: Dict[str, object]) -> Dict[str, object] | None:
    family_id = action.parameters.get("patch_family") if action.parameters else None
    if not family_id:
        family_id = action.family
    families = patch_families.get("families", []) if isinstance(patch_families, dict) else []
    for entry in families:
        if entry.get("id") == family_id:
            return entry
    return None


def _is_math_equivalence(action: ActionIR) -> bool:
    if action.family == "math_equivalence":
        return True
    if action.parameters and action.parameters.get("patch_family") == "math_equivalence":
        return True
    return False
