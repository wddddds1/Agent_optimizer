from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from schemas.job_ir import Budgets


@dataclass
class StopState:
    no_improve_iters: int = 0
    run_count: int = 0
    fail_count: int = 0
    last_family: str | None = None
    blocked_families: set[str] = field(default_factory=set)
    blocked_actions: set[str] = field(default_factory=set)
    neighbor_tune_done: bool = False
    patch_action_context_misses: Dict[str, int] = field(default_factory=dict)
    patch_action_preflight_fails: Dict[str, int] = field(default_factory=dict)
    patch_family_preflight_fails: Dict[str, int] = field(default_factory=dict)
    patch_family_blocked_until: Dict[str, int] = field(default_factory=dict)
    patch_scope_level: int = 0
    patch_scope_no_gain_iters: int = 0
    patch_scope_no_candidates: int = 0
    patch_scope_failures: int = 0


def should_stop(
    budgets: Budgets,
    iteration: int,
    state: StopState,
    elapsed_seconds: float,
    min_delta_seconds: float,
) -> bool:
    if iteration >= budgets.max_iters:
        return True
    if state.run_count >= budgets.max_runs:
        return True
    if elapsed_seconds >= budgets.max_wall_seconds:
        return True
    if min_delta_seconds > 0 and state.no_improve_iters >= 2:
        return True
    if min_delta_seconds <= 0:
        return False
    return False
