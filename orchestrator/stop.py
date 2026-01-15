from __future__ import annotations

from dataclasses import dataclass

from schemas.job_ir import Budgets


@dataclass
class StopState:
    no_improve_iters: int = 0
    run_count: int = 0
    fail_count: int = 0


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
    if state.no_improve_iters >= 2:
        return True
    if min_delta_seconds <= 0:
        return False
    return False
