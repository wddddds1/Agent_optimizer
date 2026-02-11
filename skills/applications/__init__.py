from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from schemas.action_ir import ActionIR
    from schemas.job_ir import JobIR


def apply_adapter(
    action: "ActionIR",
    job: "JobIR",
    adapter_cfg: Optional[Dict[str, object]] = None,
) -> "ActionIR":
    if job.app == "lammps":
        from skills.applications.lammps_app import apply_adapter as _lammps

        return _lammps(action, job, adapter_cfg)
    if job.app == "bwa":
        from skills.applications.bwa_app import apply_adapter as _bwa

        return _bwa(action, job, adapter_cfg)
    return action


def input_edit_allowlist(app: str = "lammps") -> List[str]:
    if app == "lammps":
        from skills.applications.lammps_app import input_edit_allowlist as _lammps

        return _lammps()
    return []


__all__ = ["apply_adapter", "input_edit_allowlist"]
