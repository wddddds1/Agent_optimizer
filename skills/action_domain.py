from __future__ import annotations

from typing import Dict, List, Tuple

from schemas.action_ir import ActionIR


def select_actions_for_direction(
    actions: List[ActionIR],
    direction_cfg: Dict[str, object],
    prefer_effects: List[str] | None = None,
) -> List[ActionIR]:
    expected_effects = set(direction_cfg.get("expected_effect", []) or [])
    if prefer_effects:
        expected_effects = expected_effects & set(prefer_effects)
    if not expected_effects:
        return []
    selected = [a for a in actions if expected_effects & set(a.expected_effect or [])]
    return selected


def enforce_candidate_policy(
    actions: List[ActionIR],
    policy: Dict[str, object],
    features: Dict[str, object],
) -> Tuple[List[ActionIR], List[Dict[str, str]]]:
    rejections: List[Dict[str, str]] = []
    adjusted = list(actions)

    adjustments = policy.get("domain_adjustments", {}) if isinstance(policy, dict) else {}
    tags = set(features.get("bottleneck_tags") or [])
    best_threads = _safe_int(features.get("best_threads"))
    if "mem_bound" in tags:
        cap = _safe_int(adjustments.get("mem_bound", {}).get("max_threads_cap"))
        if cap:
            adjusted, rejected = _cap_threads(adjusted, cap)
            rejections.extend(rejected)
    floor_cfg = adjustments.get("thread_floor", {}) if isinstance(adjustments, dict) else {}
    if best_threads:
        min_when_best = _safe_int(floor_cfg.get("min_when_best"))
        min_fraction = _safe_float(floor_cfg.get("min_fraction_of_best"))
        floor = None
        if min_fraction is not None:
            floor = int(best_threads * min_fraction)
        if min_when_best:
            floor = max(floor or 0, min_when_best)
        if floor and floor > 1:
            adjusted, rejected = _min_threads(adjusted, floor)
            rejections.extend(rejected)
    adjusted, rejected = _filter_low_ratio_effects(adjusted, adjustments, features)
    rejections.extend(rejected)

    return adjusted, rejections


def _filter_low_ratio_effects(
    actions: List[ActionIR],
    adjustments: Dict[str, object],
    features: Dict[str, object],
) -> Tuple[List[ActionIR], List[Dict[str, str]]]:
    rejected: List[Dict[str, str]] = []
    comm_ratio = float(features.get("comm_ratio") or 0.0)
    io_ratio = float(features.get("io_ratio") or 0.0)
    comm_min = _safe_float(adjustments.get("comm_bound", {}).get("min_ratio"))
    io_min = _safe_float(adjustments.get("io_bound", {}).get("min_ratio"))
    kept: List[ActionIR] = []
    for action in actions:
        effects = set(action.expected_effect or [])
        if comm_min is not None and comm_ratio < comm_min and "comm_reduce" in effects:
            rejected.append(
                {
                    "action_id": action.action_id,
                    "reason": f"comm_ratio<{comm_min}",
                }
            )
            continue
        if io_min is not None and io_ratio < io_min and "io_reduce" in effects:
            rejected.append(
                {
                    "action_id": action.action_id,
                    "reason": f"io_ratio<{io_min}",
                }
            )
            continue
        kept.append(action)
    return kept, rejected


def _cap_threads(actions: List[ActionIR], cap: int) -> Tuple[List[ActionIR], List[Dict[str, str]]]:
    kept: List[ActionIR] = []
    rejected: List[Dict[str, str]] = []
    for action in actions:
        threads = _extract_threads(action)
        if threads is not None and threads > cap:
            rejected.append({"action_id": action.action_id, "reason": f"threads>{cap} cap"})
            continue
        kept.append(action)
    return kept, rejected


def _min_threads(actions: List[ActionIR], floor: int) -> Tuple[List[ActionIR], List[Dict[str, str]]]:
    kept: List[ActionIR] = []
    rejected: List[Dict[str, str]] = []
    for action in actions:
        threads = _extract_threads(action)
        if threads is not None and threads < floor:
            rejected.append({"action_id": action.action_id, "reason": f"threads<{floor} floor"})
            continue
        kept.append(action)
    return kept, rejected


def _extract_threads(action: ActionIR) -> int | None:
    env = action.parameters.get("env", {}) if action.parameters else {}
    if not isinstance(env, dict):
        return None
    raw = env.get("OMP_NUM_THREADS")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
