from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from schemas.experiment_ir import ExperimentIR


@dataclass
class OptimizationMemory:
    experiments: List[ExperimentIR] = field(default_factory=list)
    baseline: Optional[ExperimentIR] = None
    best: Optional[ExperimentIR] = None
    min_best_improvement_pct: float = 0.001

    def record(self, exp: ExperimentIR) -> None:
        self.experiments.append(exp)
        if exp.action is None:
            self.baseline = exp
            if self.best is None:
                self.best = exp
            return
        if exp.verdict == "PASS":
            if self.best is None:
                self.best = exp
                return
            best_rt = self.best.results.runtime_seconds
            exp_rt = exp.results.runtime_seconds
            if best_rt > 0.0 and exp_rt > 0.0:
                improvement = (best_rt - exp_rt) / best_rt
                if improvement >= self.min_best_improvement_pct:
                    self.best = exp
                return
            if exp_rt > 0.0 and exp_rt < best_rt:
                self.best = exp
