from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Objective(BaseModel):
    metric: Literal["time_to_solution"] = "time_to_solution"
    goal: Literal["minimize"] = "minimize"


class Budgets(BaseModel):
    max_iters: int = Field(..., ge=1)
    max_runs: int = Field(..., ge=1)
    max_wall_seconds: int = Field(..., ge=1)


class JobIR(BaseModel):
    app: Literal["lammps", "bwa"] = "lammps"
    case_id: str
    workdir: str
    app_bin: str = ""
    lammps_bin: str = ""  # backward compat alias for app_bin
    input_script: str = ""
    env: Dict[str, str] = Field(default_factory=dict)
    run_args: List[str] = Field(default_factory=list)
    objective: Objective = Field(default_factory=Objective)
    budgets: Budgets
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

    @model_validator(mode="after")
    def _validate_paths(self) -> "JobIR":
        # Sync app_bin ↔ lammps_bin for backward compatibility
        if self.app_bin and not self.lammps_bin:
            self.lammps_bin = self.app_bin
        elif self.lammps_bin and not self.app_bin:
            self.app_bin = self.lammps_bin
        if not self.app_bin:
            raise ValueError("app_bin (or lammps_bin) must be set")
        if not self.workdir:
            raise ValueError("workdir must be set")
        # input_script only required for LAMMPS
        if self.app == "lammps" and not self.input_script:
            raise ValueError("input_script must be set for LAMMPS")
        return self
