from __future__ import annotations

from typing import Dict, List, Optional

from schemas.experiment_ir import ExperimentIR
from skills.report import write_report


class ReporterAgent:
    def write(
        self,
        experiments: List[ExperimentIR],
        baseline: ExperimentIR,
        best: Optional[ExperimentIR],
        report_dir,
        success_info: Optional[Dict[str, object]],
        agent_trace_path: Optional[str],
        llm_summary_zh: Optional[Dict[str, object]],
        candidate_policy: Optional[Dict[str, object]],
        review_decision: Optional[Dict[str, object]],
    ) -> Dict[str, object]:
        return write_report(
            experiments,
            baseline,
            best,
            report_dir,
            success_info,
            agent_trace_path,
            llm_summary_zh,
            candidate_policy,
            review_decision,
        )
