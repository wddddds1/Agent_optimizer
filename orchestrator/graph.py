from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from schemas.action_ir import ActionIR
from schemas.experiment_ir import ExperimentIR
from schemas.job_ir import JobIR
from schemas.profile_report import ProfileReport
from schemas.result_ir import ResultIR
from skills.patch_git import GitPatchContext, get_git_head
from skills.profiling_local import profile_job
from skills.report import write_report
from skills.verify import verify_run
from orchestrator.memory import OptimizationMemory
from orchestrator.llm_client import LLMClient
from orchestrator.router import RuleContext, filter_actions, rank_actions, rank_actions_with_llm
from orchestrator.stop import StopState, should_stop


@dataclass
class AnalysisResult:
    bottlenecks: List[str]
    allowed_families: List[str]


class ProfilerAgent:
    def run(self, job: JobIR, run_args: List[str], env_overrides: Dict[str, str], workdir: Path,
            artifacts_dir: Path, time_command: Optional[str], repeats: int) -> Tuple:
        return profile_job(
            job=job,
            run_args=run_args,
            env_overrides=env_overrides,
            workdir=workdir,
            artifacts_dir=artifacts_dir,
            time_command=time_command,
            repeats=repeats,
        )


class AnalystAgent:
    def analyze(self, profile: ProfileReport) -> AnalysisResult:
        timing = profile.timing_breakdown
        total = timing.get("total", 0.0) or 0.0
        comm_ratio = (timing.get("comm", 0.0) / total) if total else 0.0
        output_ratio = (timing.get("output", 0.0) / total) if total else 0.0
        bottlenecks = []
        allowed = []
        if comm_ratio > 0.2:
            bottlenecks.append("communication")
            allowed.extend(["neigh_modify"])
        if output_ratio > 0.2:
            bottlenecks.append("io")
            allowed.extend(["thermo_io"])
        bottlenecks.append("compute")
        allowed.extend(["omp_threads", "omp_binding", "omp_places", "omp_dynamic", "omp_wait_policy", "lammps_flags"])
        return AnalysisResult(bottlenecks=bottlenecks, allowed_families=sorted(set(allowed)))


class OptimizerAgent:
    def __init__(self, llm_client: Optional[LLMClient]) -> None:
        self.llm_client = llm_client

    def propose(
        self,
        actions: List[ActionIR],
        ctx: RuleContext,
        allowed_families: List[str],
        top_k: int,
        profile: ProfileReport,
        policy: Dict[str, object],
        exclude_action_ids: List[str],
    ) -> List[ActionIR]:
        filtered = filter_actions(actions, ctx, allowed_families, policy)
        filtered = [action for action in filtered if action.action_id not in exclude_action_ids]
        if self.llm_client and self.llm_client.config.enabled:
            context = {
                "case_id": ctx.job.case_id,
                "allowed_families": allowed_families,
                "run_args": ctx.run_args,
                "env": ctx.env,
            }
            ranked = rank_actions_with_llm(filtered, profile, context, self.llm_client)
        else:
            ranked = rank_actions(filtered, profile)
        return ranked[:top_k]


class VerifierAgent:
    def verify(
        self,
        action: Optional[ActionIR],
        result: ResultIR,
        profile: ProfileReport,
        gates: Dict[str, object],
        baseline_profile: Optional[ProfileReport],
    ):
        return verify_run(action, result, profile, gates, baseline_profile)


def run_optimization(
    job: JobIR,
    actions: List[ActionIR],
    policy: Dict[str, object],
    gates: Dict[str, object],
    artifacts_dir: Path,
    time_command: Optional[str],
    min_delta_seconds: float,
    top_k: int,
    llm_client: Optional[LLMClient],
) -> Dict[str, object]:
    profiler = ProfilerAgent()
    analyst = AnalystAgent()
    optimizer = OptimizerAgent(llm_client)
    verifier = VerifierAgent()
    memory = OptimizationMemory()
    state = StopState()

    start_time = time.monotonic()
    repo_root = Path(__file__).resolve().parents[1]
    baseline_exp = _run_experiment(
        exp_id="baseline",
        job=job,
        action=None,
        actions_root=repo_root,
        policy=policy,
        gates=gates,
        profiler=profiler,
        verifier=verifier,
        artifacts_dir=artifacts_dir,
        time_command=time_command,
        repeats=1,
        baseline_profile=None,
        baseline_runtime=None,
    )
    memory.record(baseline_exp)
    state.run_count += 1

    prev_best_time = baseline_exp.results.runtime_seconds
    iteration = 0
    while True:
        elapsed = time.monotonic() - start_time
        if should_stop(job.budgets, iteration, state, elapsed, min_delta_seconds):
            break
        iteration += 1

        input_text = _safe_read(Path(job.input_script))
        ctx = RuleContext(job=job, input_text=input_text, run_args=job.run_args, env=job.env)
        analysis = analyst.analyze(memory.best.profile_report if memory.best else baseline_exp.profile_report)
        tested_actions = [exp.action.action_id for exp in memory.experiments if exp.action]
        candidates = optimizer.propose(
            actions=actions,
            ctx=ctx,
            allowed_families=analysis.allowed_families,
            top_k=top_k,
            profile=memory.best.profile_report if memory.best else baseline_exp.profile_report,
            policy=policy,
            exclude_action_ids=tested_actions,
        )
        if not candidates:
            break

        for action in candidates:
            exp_id = f"iter{iteration}-{action.action_id}"
            exp = _run_experiment(
                exp_id=exp_id,
                job=job,
                action=action,
                actions_root=repo_root,
                policy=policy,
                gates=gates,
                profiler=profiler,
                verifier=verifier,
                artifacts_dir=artifacts_dir,
                time_command=time_command,
                repeats=gates.get("variance", {}).get("repeats", 1),
                baseline_profile=baseline_exp.profile_report,
                baseline_runtime=baseline_exp.results.runtime_seconds,
            )
            memory.record(exp)
            state.run_count += 1
            if exp.verdict == "FAIL":
                state.fail_count += 1

        if memory.best and memory.best.results.runtime_seconds + min_delta_seconds < prev_best_time:
            state.no_improve_iters = 0
            prev_best_time = memory.best.results.runtime_seconds
        else:
            state.no_improve_iters += 1

    report_info = write_report(memory.experiments, baseline_exp, memory.best, artifacts_dir)
    return report_info


def _run_experiment(
    exp_id: str,
    job: JobIR,
    action: Optional[ActionIR],
    actions_root: Path,
    policy: Dict[str, object],
    gates: Dict[str, object],
    profiler: ProfilerAgent,
    verifier: VerifierAgent,
    artifacts_dir: Path,
    time_command: Optional[str],
    repeats: int,
    baseline_profile: Optional[ProfileReport],
    baseline_runtime: Optional[float],
) -> ExperimentIR:
    run_id = exp_id
    run_dir = artifacts_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env_overrides, run_args = _apply_run_config_action(job, action)
    run_args = _ensure_log_path(run_args, run_dir)

    patch_path = None
    git_before = None
    git_after = None
    workdir = Path(job.workdir)
    input_script = Path(job.input_script)

    input_edit = None
    allowlist = policy.get("input_edit_allowlist", [])
    if action and "input_script" in action.applies_to:
        input_edit = action.parameters.get("input_edit")
        if not _is_under_repo(input_script, actions_root):
            result = ResultIR(
                runtime_seconds=0.0,
                derived_metrics={},
                correctness_metrics={},
                logs=[],
                exit_code=1,
                samples=[],
            )
            profile = ProfileReport(timing_breakdown={}, system_metrics={}, notes=["input script outside repo"])
            verdict = "FAIL"
            reasons = ["input script outside repo; cannot apply git patch"]
            exp = _build_experiment_ir(
                exp_id=exp_id,
                job=job,
                action=action,
                patch_path=None,
                git_before=None,
                git_after=None,
                run_id=run_id,
                profile=profile,
                result=result,
                verdict=verdict,
                reasons=reasons,
            )
            _write_experiment(run_dir, exp)
            return exp
        try:
            with GitPatchContext(
                repo_root=actions_root,
                exp_id=exp_id,
                artifacts_dir=run_dir,
                input_script=input_script,
                input_edit=input_edit,
                allowlist=allowlist,
            ) as ctx:
                patch_path = str(ctx.patch_path)
                git_before = ctx.git_commit_before
                git_after = ctx.git_commit_after
                workdir = ctx.worktree_dir
                input_script = ctx.map_to_worktree(input_script)
                job_snapshot = job.model_copy(deep=True)
                job_snapshot.workdir = str(workdir)
                job_snapshot.input_script = str(input_script)
                run_output, profile = profiler.run(
                    job_snapshot, run_args, env_overrides, workdir, run_dir, time_command, repeats
                )
                result = _build_result_ir(run_output, profile, baseline_runtime)
        except Exception as exc:
            result = ResultIR(
                runtime_seconds=0.0,
                derived_metrics={},
                correctness_metrics={},
                logs=[],
                exit_code=1,
                samples=[],
            )
            profile = ProfileReport(timing_breakdown={}, system_metrics={}, notes=[str(exc)])
            verdict = "FAIL"
            reasons = [str(exc)]
            exp = _build_experiment_ir(
                exp_id=exp_id,
                job=job,
                action=action,
                patch_path=None,
                git_before=None,
                git_after=None,
                run_id=run_id,
                profile=profile,
                result=result,
                verdict=verdict,
                reasons=reasons,
            )
            _write_experiment(run_dir, exp)
            return exp
    else:
        job_snapshot = job.model_copy(deep=True)
        job_snapshot.env.update(env_overrides)
        job_snapshot.run_args = run_args
        run_output, profile = profiler.run(
            job_snapshot, run_args, env_overrides, workdir, run_dir, time_command, repeats
        )
        result = _build_result_ir(run_output, profile, baseline_runtime)
        patch_path = _write_run_config_diff(run_dir, job, run_args, env_overrides)
        git_before = get_git_head(actions_root)
        git_after = git_before

    verify = verifier.verify(action, result, profile, gates, baseline_profile)
    result.correctness_metrics.update(verify.correctness_metrics)
    exp = _build_experiment_ir(
        exp_id=exp_id,
        job=job_snapshot,
        action=action,
        patch_path=patch_path,
        git_before=git_before,
        git_after=git_after,
        run_id=run_id,
        profile=profile,
        result=result,
        verdict=verify.verdict,
        reasons=verify.reasons,
    )
    _write_experiment(run_dir, exp)
    return exp


def _apply_run_config_action(job: JobIR, action: Optional[ActionIR]) -> Tuple[Dict[str, str], List[str]]:
    env_overrides: Dict[str, str] = {}
    run_args = list(job.run_args)
    if not action:
        return env_overrides, run_args

    env_overrides.update(action.parameters.get("env", {}))
    run_args_cfg = action.parameters.get("run_args", {})
    for entry in run_args_cfg.get("set_flags", []):
        flag = entry.get("flag")
        values = entry.get("values", [])
        arg_count = entry.get("arg_count", len(values))
        run_args = _remove_flag(run_args, flag, arg_count)
        run_args.extend([flag] + list(values))
    return env_overrides, run_args


def _remove_flag(run_args: List[str], flag: str, arg_count: int) -> List[str]:
    cleaned: List[str] = []
    i = 0
    while i < len(run_args):
        if run_args[i] == flag:
            i += 1 + arg_count
            continue
        cleaned.append(run_args[i])
        i += 1
    return cleaned


def _ensure_log_path(run_args: List[str], run_dir: Path) -> List[str]:
    args = list(run_args)
    log_path = run_dir / "log.lammps"
    if "-log" in args:
        idx = args.index("-log")
        if idx + 1 < len(args):
            args[idx + 1] = str(log_path)
        else:
            args.append(str(log_path))
        return args
    return args + ["-log", str(log_path)]


def _build_result_ir(run_output, profile: ProfileReport, baseline_runtime: Optional[float]) -> ResultIR:
    timing = profile.timing_breakdown
    total = timing.get("total", run_output.runtime_seconds) or run_output.runtime_seconds
    comm_ratio = (timing.get("comm", 0.0) / total) if total else 0.0
    derived = {"comm_ratio": comm_ratio}
    if baseline_runtime and run_output.runtime_seconds:
        derived["speedup_vs_baseline"] = baseline_runtime / run_output.runtime_seconds
    samples = run_output.samples or []
    if len(samples) >= 2:
        mean = sum(samples) / len(samples)
        var = sum((x - mean) ** 2 for x in samples) / len(samples)
        derived["variance"] = var
        derived["variance_cv"] = (var ** 0.5) / mean if mean else 0.0
    return ResultIR(
        runtime_seconds=run_output.runtime_seconds,
        derived_metrics=derived,
        correctness_metrics={},
        logs=[run_output.stdout_path, run_output.stderr_path, run_output.log_path],
        exit_code=run_output.exit_code,
        samples=run_output.samples,
    )


def _build_experiment_ir(
    exp_id: str,
    job: JobIR,
    action: Optional[ActionIR],
    patch_path: Optional[str],
    git_before: Optional[str],
    git_after: Optional[str],
    run_id: str,
    profile: ProfileReport,
    result: ResultIR,
    verdict: str,
    reasons: List[str],
) -> ExperimentIR:
    timestamps = [time.strftime("%Y-%m-%dT%H:%M:%S")]
    return ExperimentIR(
        exp_id=exp_id,
        parent_exp_id=None,
        job=job,
        action=action,
        git_commit_before=git_before,
        git_commit_after=git_after,
        patch_path=patch_path,
        run_id=run_id,
        timestamps=timestamps,
        profile_report=profile,
        results=result,
        verdict=verdict,
        reasons=reasons,
    )


def _write_run_config_diff(
    run_dir: Path, job: JobIR, run_args: List[str], env_overrides: Dict[str, str]
) -> str:
    diff = {
        "env_overrides": env_overrides,
        "run_args_before": job.run_args,
        "run_args_after": run_args,
    }
    diff_path = run_dir / "run_config.diff.json"
    diff_path.write_text(json.dumps(diff, indent=2), encoding="utf-8")
    return str(diff_path)


def _write_experiment(run_dir: Path, exp: ExperimentIR) -> None:
    path = run_dir / "experiment.json"
    path.write_text(json.dumps(exp.model_dump(), indent=2), encoding="utf-8")


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _is_under_repo(path: Path, repo_root: Path) -> bool:
    try:
        path.resolve().relative_to(repo_root.resolve())
        return True
    except ValueError:
        return False
