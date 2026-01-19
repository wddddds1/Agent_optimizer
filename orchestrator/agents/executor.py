from __future__ import annotations

from typing import Callable, Dict, Optional

from schemas.action_ir import ActionIR
from schemas.experiment_ir import ExperimentIR
from schemas.job_ir import JobIR


class ExecutorAgent:
    def __init__(self, runner: Callable[..., ExperimentIR]) -> None:
        self.runner = runner

    def execute(
        self,
        exp_id: str,
        job: JobIR,
        base_job: Optional[JobIR],
        base_run_id: Optional[str],
        base_action_id: Optional[str],
        action: Optional[ActionIR],
        actions_root,
        policy: Dict[str, object],
        gates: Dict[str, object],
        profiler,
        verifier,
        artifacts_dir,
        time_command: Optional[str],
        build_cfg: Dict[str, object],
        repeats: int,
        runtime_agg: str,
        baseline_exp,
        baseline_runtime: Optional[float],
        prior_samples,
        trace_events,
        parent_run_id: Optional[str],
        iteration: Optional[int],
        llm_trace: Optional[Dict[str, object]],
        reporter,
    ) -> ExperimentIR:
        return self.runner(
            exp_id=exp_id,
            job=job,
            base_job=base_job,
            base_run_id=base_run_id,
            base_action_id=base_action_id,
            action=action,
            actions_root=actions_root,
            policy=policy,
            gates=gates,
            profiler=profiler,
            verifier=verifier,
            artifacts_dir=artifacts_dir,
            time_command=time_command,
            build_cfg=build_cfg,
            repeats=repeats,
            runtime_agg=runtime_agg,
            baseline_exp=baseline_exp,
            baseline_runtime=baseline_runtime,
            prior_samples=prior_samples,
            trace_events=trace_events,
            parent_run_id=parent_run_id,
            iteration=iteration,
            llm_trace=llm_trace,
            reporter=reporter,
        )
