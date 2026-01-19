from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional

from schemas.failure_ir import FailureSummary, NextStep
from schemas.experiment_ir import ExperimentIR


class TriageAgent:
    def classify(self, exp: ExperimentIR, run_dir: Path) -> FailureSummary:
        category, causes = _classify_failure(exp, run_dir)
        signature = _signature(exp.run_id, causes)
        next_steps = _next_steps(category)
        return FailureSummary(
            run_id=exp.run_id,
            action_id=exp.action.action_id if exp.action else "baseline",
            category=category,
            signature=signature,
            top_causes=causes,
            next_steps=next_steps,
            suggest_debug_mode=category in {"BUILD", "RUNTIME"},
            suggest_disable_family=None,
            confidence=0.6,
        )


def _classify_failure(exp: ExperimentIR, run_dir: Path) -> tuple[str, List[str]]:
    notes = exp.profile_report.notes or []
    reasons = exp.reasons or []
    stderr_path = run_dir / "stderr.log"
    stderr_text = ""
    if stderr_path.exists():
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    blob = "\n".join(notes + reasons + [stderr_text])
    if "CMake Error" in blob or "ninja:" in blob or "make:" in blob:
        return "BUILD", ["build failure detected"]
    if "ERROR" in blob or "nonzero exit code" in blob:
        return "RUNTIME", ["runtime error detected"]
    if "correctness" in blob or "drift" in blob:
        return "CORRECTNESS", ["correctness gate failed"]
    if "variance" in blob:
        return "PERF_NOISE", ["variance gate failed"]
    return "UNKNOWN", ["unclassified failure"]


def _signature(run_id: str, causes: List[str]) -> str:
    blob = run_id + "|" + "|".join(causes)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _next_steps(category: str) -> List[NextStep]:
    if category == "BUILD":
        return [NextStep(type="fix_toolchain", detail="inspect build.log and CMakeCache.txt")]
    if category == "RUNTIME":
        return [NextStep(type="collect_more_logs", detail="inspect stderr.log and log.lammps")]
    if category == "CORRECTNESS":
        return [NextStep(type="adjust_case", detail="run longer correctness check or compare series")]
    if category == "PERF_NOISE":
        return [NextStep(type="retry", detail="increase repeats or tighten variance")]
    return [NextStep(type="collect_more_logs", detail="inspect run artifacts and trace")]
