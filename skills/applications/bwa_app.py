from __future__ import annotations

from typing import Dict, Optional

from schemas.action_ir import ActionIR
from schemas.job_ir import JobIR


def apply_adapter(
    action: ActionIR,
    job: JobIR,
    adapter_cfg: Optional[Dict[str, object]] = None,
) -> ActionIR:
    """BWA adapter — minimal.

    BWA thread count is handled via set_flags (-t N) in action_space.yaml,
    so no special injection is needed here (unlike LAMMPS's -sf omp).
    """
    return action
