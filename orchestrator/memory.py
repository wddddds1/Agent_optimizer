from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from schemas.experiment_ir import ExperimentIR


@dataclass
class OptimizationMemory:
    experiments: List[ExperimentIR] = field(default_factory=list)
    baseline: Optional[ExperimentIR] = None
    best: Optional[ExperimentIR] = None

    def record(self, exp: ExperimentIR) -> None:
        self.experiments.append(exp)
        if exp.action is None:
            self.baseline = exp
            if self.best is None:
                self.best = exp
            return
        if exp.verdict == "PASS":
            if self.best is None or exp.results.runtime_seconds < self.best.results.runtime_seconds:
                self.best = exp
