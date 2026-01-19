from __future__ import annotations

import hashlib
import json
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from schemas.action_ir import ActionIR
from schemas.analysis_ir import AnalysisResult
from schemas.experiment_ir import ExperimentIR
from schemas.job_ir import JobIR, Budgets
from schemas.profile_report import ProfileReport
from schemas.result_ir import ResultIR
from skills.build_local import BuildOutput, build_job, collect_binary_provenance
from skills.patch_git import GitPatchContext, get_git_head, get_git_status
from orchestrator.console import ConsoleUI
from orchestrator.agents import (
    AnalystAgent,
    ExecutorAgent,
    PlannerAgent,
    ProfilerAgent,
    OptimizerAgent,
    ReporterAgent,
    RouterRankerAgent,
    TriageAgent,
    VerifierAgent,
)
from orchestrator.memory import OptimizationMemory
from orchestrator.llm_client import LLMClient
from orchestrator.router import RuleContext, filter_actions
from orchestrator.stop import StopState, should_stop


def _best_pass_exp(experiments: List[ExperimentIR]) -> Optional[ExperimentIR]:
    best = None
    for exp in experiments:
        if exp.verdict != "PASS":
            continue
        if best is None or exp.results.runtime_seconds < best.results.runtime_seconds:
            best = exp
    return best


def _stop_reason(
    budgets: Budgets,
    iteration: int,
    state: StopState,
    elapsed_seconds: float,
    min_delta_seconds: float,
) -> str:
    if iteration >= budgets.max_iters:
        return f"budget max_iters reached ({budgets.max_iters})"
    if state.run_count >= budgets.max_runs:
        return f"budget max_runs reached ({budgets.max_runs})"
    if elapsed_seconds >= budgets.max_wall_seconds:
        return f"budget max_wall_seconds reached ({budgets.max_wall_seconds}s)"
    if state.no_improve_iters >= 2:
        return "no improvement for 2 consecutive iterations"
    if min_delta_seconds <= 0:
        return "stop condition triggered"
    return "stop condition triggered"


def _build_history_summary(experiments: List[ExperimentIR]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "family_best_gain": {},
        "family_failures": {},
        "family_attempts": {},
    }
    for exp in experiments:
        if exp.action is None:
            continue
        family = exp.action.family
        summary["family_attempts"][family] = summary["family_attempts"].get(family, 0) + 1
        if exp.verdict == "FAIL":
            summary["family_failures"][family] = summary["family_failures"].get(family, 0) + 1
            continue
        speedup = exp.results.derived_metrics.get("speedup_vs_baseline")
        if speedup is None:
            continue
        gain = speedup - 1.0
        best = summary["family_best_gain"].get(family)
        if best is None or gain > best:
            summary["family_best_gain"][family] = gain
    return summary


def _apply_confidence_policy(
    actions: List[ActionIR],
    analysis: AnalysisResult,
    policy: Dict[str, object],
) -> List[ActionIR]:
    if analysis.confidence >= 0.8:
        return actions
    rules = policy.get("profiling_rules", {})
    rule_key = "low_confidence_allow" if analysis.confidence >= 0.5 else "unknown_confidence_allow"
    rule = rules.get(rule_key) or rules.get("low_confidence_allow") or {}
    if not rule:
        return []
    allowed_applies_to = set(rule.get("applies_to", []))
    allowed_risk = set(rule.get("risk_levels", []))
    allowed_families = set(rule.get("families", []))
    filtered: List[ActionIR] = []
    for action in actions:
        if allowed_applies_to and not any(target in allowed_applies_to for target in action.applies_to):
            continue
        if allowed_risk and action.risk_level not in allowed_risk:
            continue
        if allowed_families and action.family not in allowed_families:
            continue
        filtered.append(action)
    return filtered


def run_optimization(
    job: JobIR,
    actions: List[ActionIR],
    policy: Dict[str, object],
    gates: Dict[str, object],
    artifacts_dir: Path,
    time_command: Optional[str],
    min_delta_seconds: float,
    top_k: int,
    selection_mode: str,
    direction_top_k: int,
    llm_client: Optional[LLMClient],
    planner_cfg: Optional[Dict[str, object]] = None,
    reporter: Optional[ConsoleUI] = None,
    build_cfg: Optional[Dict[str, object]] = None,
    baseline_repeats: int = 1,
    baseline_stat: str = "mean",
    validate_top1_repeats: int = 0,
    min_improvement_pct: float = 0.0,
) -> Dict[str, object]:
    profiler = ProfilerAgent()
    analyst = AnalystAgent(llm_client)
    planner = PlannerAgent(planner_cfg or {}, llm_client)
    optimizer = OptimizerAgent(llm_client)
    ranker = RouterRankerAgent(llm_client)
    verifier = VerifierAgent()
    executor = ExecutorAgent(_run_experiment)
    reporter_agent = ReporterAgent()
    triage = TriageAgent()
    memory = OptimizationMemory()
    state = StopState()
    trace_events: List[Dict[str, object]] = []

    start_time = time.monotonic()
    repo_root = Path(__file__).resolve().parents[1]
    if reporter:
        reporter.header(
            job=job,
            selection_mode=selection_mode,
            top_k=top_k,
            direction_top_k=direction_top_k,
            llm_enabled=bool(llm_client and llm_client.config.enabled),
            llm_model=llm_client.config.model if llm_client else "n/a",
            artifacts_dir=str(artifacts_dir),
            baseline_repeats=baseline_repeats,
            baseline_stat=baseline_stat,
            validate_top1_repeats=validate_top1_repeats,
            min_improvement_pct=min_improvement_pct,
        )
    baseline_exp = executor.execute(
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
        build_cfg=build_cfg or {},
        repeats=baseline_repeats,
        runtime_agg=baseline_stat,
        baseline_exp=None,
        baseline_runtime=None,
        prior_samples=None,
        trace_events=trace_events,
        parent_run_id=None,
        iteration=None,
        llm_trace=None,
        reporter=reporter,
    )
    memory.record(baseline_exp)
    state.run_count += 1
    if reporter:
        reporter.update_baseline(baseline_exp)
    _append_run_index(artifacts_dir, baseline_exp, parent_run_id=None, iteration=None)

    if baseline_exp.verdict == "FAIL":
        failure_info = {
            "success": False,
            "reason": "baseline failed gates",
            "target_improvement_pct": min_improvement_pct,
            "achieved_improvement_pct": 0.0,
            "candidate_run_id": None,
        }
        trace_events.append(
            {
                "event": "baseline_failure",
                "agent": "VerifierAgent",
                "success": failure_info,
            }
        )
        if reporter:
            reporter.stop("baseline failed gates")
        agent_trace_path = _write_agent_trace(artifacts_dir, trace_events)
        report_info = reporter_agent.write(
            memory.experiments,
            baseline_exp,
            baseline_exp,
            artifacts_dir,
            failure_info,
            agent_trace_path,
            llm_summary_zh=None,
        )
        report_info["agent_trace"] = agent_trace_path
        return report_info

    prev_best_time = baseline_exp.results.runtime_seconds
    iteration = 0
    while True:
        elapsed = time.monotonic() - start_time
        if should_stop(job.budgets, iteration, state, elapsed, min_delta_seconds):
            if reporter:
                reporter.stop(
                    _stop_reason(job.budgets, iteration, state, elapsed, min_delta_seconds)
                )
            break
        iteration += 1

        input_text = _safe_read(Path(job.input_script))
        ctx = RuleContext(job=job, input_text=input_text, run_args=job.run_args, env=job.env)
        profile_ref = memory.best.profile_report if memory.best else baseline_exp.profile_report
        history_summary = _build_history_summary(memory.experiments)
        tested_actions = [exp.action.action_id for exp in memory.experiments if exp.action]
        analysis = analyst.analyze(profile_ref, history_summary, policy, job.tags)
        if selection_mode == "direction":
            analysis.allowed_families = _select_direction_families(actions, profile_ref)
        if "allow_build" in job.tags:
            analysis.allowed_families.append("build_config")
        if "allow_source_patch" in job.tags:
            analysis.allowed_families.append("source_patch")
        analysis.allowed_families = sorted(set(analysis.allowed_families))
        if selection_mode != "direction":
            allowed_targets = set(analysis.allowed_families)
            actions_by_target = [
                action for action in actions if any(target in allowed_targets for target in action.applies_to)
            ]
            analysis.allowed_families = sorted({action.family for action in actions_by_target})
            available_actions = _apply_confidence_policy(actions_by_target, analysis, policy)
        else:
            available_actions = _apply_confidence_policy(actions, analysis, policy)
        eligible_actions = filter_actions(available_actions, ctx, analysis.allowed_families, policy)
        analysis.allowed_families = sorted({action.family for action in eligible_actions})
        allowed_after_confidence = sorted({action.family for action in available_actions})
        if analysis.allowed_families:
            analysis.allowed_families = sorted(
                set(analysis.allowed_families) & set(allowed_after_confidence)
            )
        else:
            analysis.allowed_families = allowed_after_confidence
        trace_events.append(
            {
                "event": "analysis",
                "agent": "AnalystAgent",
                "iteration": iteration,
                "bottleneck": analysis.bottleneck,
                "allowed_families": analysis.allowed_families,
                "confidence": analysis.confidence,
                "rationale": analysis.rationale,
            }
        )
        if reporter:
            reporter.analysis(
                bottlenecks=[analysis.bottleneck],
                allowed_families=analysis.allowed_families,
                confidence=f"{analysis.confidence:.2f}",
                rationale=analysis.rationale,
            )
        availability: Dict[str, int] = {}
        for action in eligible_actions:
            if action.action_id in tested_actions:
                continue
            availability[action.family] = availability.get(action.family, 0) + 1
        plan = planner.plan(iteration, analysis, job.budgets, history_summary, availability)
        trace_events.append(
            {
                "event": "plan",
                "agent": "PlannerAgent",
                "iteration": iteration,
                "plan": plan.model_dump(),
            }
        )
        if reporter:
            reporter.plan_summary(plan)

        candidate_lists = optimizer.propose(
            actions=available_actions,
            ctx=ctx,
            plan=plan,
            policy=policy,
            profile=memory.best.profile_report if memory.best else baseline_exp.profile_report,
            exclude_action_ids=tested_actions,
        )
        candidate_ids: List[str] = []
        for candidate_list in candidate_lists:
            candidate_ids.extend([action.action_id for action in candidate_list.candidates])
        trace_events.append(
            {
                "event": "candidate_proposal",
                "agent": "OptimizerAgent",
                "iteration": iteration,
                "selection_mode": selection_mode,
                "allowed_families": analysis.allowed_families,
                "analysis_confidence": analysis.confidence,
                "candidate_action_ids": candidate_ids,
            }
        )
        ranked = ranker.rank(candidate_lists, ctx, policy, profile_ref)
        ranked_actions = [item.action for item in ranked.ranked]
        rank_limit = min(plan.max_candidates, top_k, max(len(ranked_actions), 1))
        ranked_actions = ranked_actions[:rank_limit]
        trace_events.append(
            {
                "event": "ranked_actions",
                "agent": "RouterRankerAgent",
                "iteration": iteration,
                "ranked_action_ids": [action.action_id for action in ranked_actions],
                "rejected_action_ids": [item.action_id for item in ranked.rejected],
                "scoring_notes": ranked.scoring_notes,
            }
        )
        if reporter:
            reporter.candidates(
                actions=ranked_actions,
                ranking_mode=ranked.scoring_notes or "heuristic",
                selection_mode=selection_mode,
                llm_explanation=None,
            )
            reporter.rank_summary(ranked)
        if not ranked_actions:
            if reporter:
                reporter.stop("无可执行候选")
            break

        iteration_experiments: List[ExperimentIR] = []
        candidate_repeats = max(1, plan.evaluation.candidate_repeats_stage1)
        for action in ranked_actions:
            if state.run_count >= job.budgets.max_runs:
                break
            exp_id = f"iter{iteration}-{action.action_id}"
            exp = executor.execute(
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
                build_cfg=build_cfg or {},
                repeats=candidate_repeats,
                runtime_agg="mean",
                baseline_exp=baseline_exp,
                baseline_runtime=baseline_exp.results.runtime_seconds,
                prior_samples=None,
                trace_events=trace_events,
                parent_run_id=baseline_exp.run_id,
                iteration=iteration,
                llm_trace=None,
                reporter=reporter,
            )
            memory.record(exp)
            iteration_experiments.append(exp)
            state.run_count += 1
            if exp.verdict == "FAIL":
                state.fail_count += 1
                failure = triage.classify(exp, artifacts_dir / "runs" / exp.run_id)
                trace_events.append(
                    {
                        "event": "triage",
                        "agent": "TriageAgent",
                        "run_id": exp.run_id,
                        "summary": failure.model_dump(),
                    }
                )
            _append_run_index(artifacts_dir, exp, parent_run_id=baseline_exp.run_id, iteration=iteration)

        llm_iteration_summary = None
        if iteration_experiments and llm_client and llm_client.config.enabled:
            llm_iteration_summary = _build_llm_iteration_summary_zh(
                iteration=iteration,
                analysis=analysis,
                candidates=ranked_actions,
                experiments=iteration_experiments,
                baseline=baseline_exp,
                llm_client=llm_client,
            )
            if llm_iteration_summary:
                trace_events.append(
                    {
                        "event": "llm_iteration_summary",
                        "agent": "OptimizerAgent",
                        "iteration": iteration,
                        "summary": llm_iteration_summary,
                    }
                )
        if iteration_experiments:
            _write_iteration_summary(
                artifacts_dir=artifacts_dir,
                iteration=iteration,
                analysis=analysis,
                candidates=ranked_actions,
                experiments=iteration_experiments,
                baseline=baseline_exp,
                llm_enabled=bool(llm_client and llm_client.config.enabled),
                llm_trace=None,
                llm_summary_zh=llm_iteration_summary,
            )
            if reporter:
                reporter.iteration_summary(
                    iteration,
                    _best_pass_exp(iteration_experiments),
                    memory.best,
                )

        if memory.best and memory.best.results.runtime_seconds + min_delta_seconds < prev_best_time:
            state.no_improve_iters = 0
            prev_best_time = memory.best.results.runtime_seconds
        else:
            state.no_improve_iters += 1

    validation_exp = None
    if (
        memory.best
        and memory.best.action
        and validate_top1_repeats > 0
        and state.run_count < job.budgets.max_runs
    ):
        repeats = validate_top1_repeats
        exp_id = f"{memory.best.exp_id}-validate"
        validation_exp = executor.execute(
            exp_id=exp_id,
            job=job,
            action=memory.best.action,
            actions_root=repo_root,
            policy=policy,
            gates=gates,
            profiler=profiler,
            verifier=verifier,
            artifacts_dir=artifacts_dir,
            time_command=time_command,
            build_cfg=build_cfg or {},
            repeats=repeats,
            runtime_agg=baseline_stat,
            baseline_exp=baseline_exp,
            baseline_runtime=baseline_exp.results.runtime_seconds,
            prior_samples=memory.best.results.samples,
            trace_events=trace_events,
            parent_run_id=memory.best.run_id,
            iteration=iteration,
            llm_trace=None,
            reporter=reporter,
        )
        memory.record(validation_exp)
        state.run_count += 1
        if validation_exp.verdict == "FAIL":
            state.fail_count += 1
        _append_run_index(
            artifacts_dir, validation_exp, parent_run_id=memory.best.run_id, iteration=iteration
        )

    success_info = _evaluate_success(
        baseline_exp=baseline_exp,
        best_exp=memory.best,
        validated_exp=validation_exp,
        min_improvement_pct=min_improvement_pct,
        validation_expected=validate_top1_repeats > 0,
    )
    trace_events.append(
        {
            "event": "success_evaluation",
            "agent": "VerifierAgent",
            "success": success_info,
        }
    )
    llm_summary_zh = _build_llm_summary_zh(memory.experiments, baseline_exp, memory.best, success_info, llm_client)
    if llm_summary_zh:
        trace_events.append(
            {
                "event": "llm_report_summary",
                "agent": "OptimizerAgent",
                "summary": llm_summary_zh,
            }
        )
    agent_trace_path = _write_agent_trace(artifacts_dir, trace_events)
    report_info = reporter_agent.write(
        memory.experiments,
        baseline_exp,
        memory.best,
        artifacts_dir,
        success_info,
        agent_trace_path,
        llm_summary_zh,
    )
    report_info["agent_trace"] = agent_trace_path
    if reporter:
        reporter.final(
            best=memory.best,
            report_md=report_info.get("report_md", ""),
            report_zh=report_info.get("report_zh"),
        )
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
    build_cfg: Dict[str, object],
    repeats: int,
    runtime_agg: str,
    baseline_exp: Optional[ExperimentIR],
    baseline_runtime: Optional[float],
    prior_samples: Optional[List[float]],
    trace_events: Optional[List[Dict[str, object]]],
    parent_run_id: Optional[str],
    iteration: Optional[int],
    llm_trace: Optional[Dict[str, object]],
    reporter: Optional[ConsoleUI],
) -> ExperimentIR:
    run_id = exp_id
    run_dir = artifacts_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env_overrides, run_args = _apply_run_config_action(job, action)
    run_args = _ensure_log_path(run_args, run_dir)
    run_trace: List[Dict[str, object]] = []
    if reporter:
        reporter.run_start(exp_id=exp_id, action=action, env_overrides=env_overrides, run_args=run_args)
    _append_trace(
        trace_events,
        run_trace,
        {
            "event": "experiment_start",
            "agent": "OptimizerAgent",
            "run_id": run_id,
            "action_id": action.action_id if action else "baseline",
            "env_overrides": env_overrides,
            "run_args": run_args,
        },
    )
    if llm_trace:
        llm_event = dict(llm_trace)
        llm_event["run_id"] = run_id
        _append_trace(None, run_trace, llm_event)

    patch_path = None
    git_before = None
    git_after = None
    workdir = Path(job.workdir)
    input_script = Path(job.input_script)
    source_root = actions_root

    build_output: Optional[BuildOutput] = None
    build_config_diff_path: Optional[str] = None
    binary_provenance: Optional[Dict[str, object]] = None
    run_output = None

    job_snapshot = job.model_copy(deep=True)
    job_snapshot.env.update(env_overrides)
    job_snapshot.run_args = run_args

    input_edit = action.parameters.get("input_edit") if action and "input_script" in action.applies_to else None
    patch_params = _extract_patch_params(action, actions_root)
    requires_patch = bool(action and any(t in action.applies_to for t in ["input_script", "source_patch"]))
    allowlist = policy.get("input_edit_allowlist", [])

    if requires_patch and not _is_under_repo(input_script, actions_root):
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
            job=job_snapshot,
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
        if reporter:
            reporter.verify_result(exp)
        _write_experiment(run_dir, exp)
        _append_trace(
            trace_events,
            run_trace,
            {
                "event": "error",
                "agent": "VerifierAgent",
                "run_id": run_id,
                "message": "input script outside repo; cannot apply git patch",
            },
        )
        _write_run_manifest(
            run_dir=run_dir,
            run_id=run_id,
            job=job_snapshot,
            action=action,
            env_overrides=env_overrides,
            run_args=run_args,
            git_before=None,
            git_after=None,
            patch_path=None,
            parent_run_id=parent_run_id,
            iteration=iteration,
            result=result,
            profile=profile,
            verify=None,
            run_output=None,
            repo_root=actions_root,
            build_output=None,
            binary_provenance=None,
            build_config_diff_path=None,
        )
        _write_run_trace(run_dir, run_trace)
        return exp

    ctx_mgr = (
        GitPatchContext(
            repo_root=actions_root,
            exp_id=exp_id,
            artifacts_dir=run_dir,
            input_script=input_script,
            input_edit=input_edit,
            allowlist=allowlist,
            patch_path=patch_params.get("patch_path"),
            patch_root=patch_params.get("patch_root"),
        )
        if requires_patch
        else nullcontext()
    )

    try:
        with ctx_mgr as ctx:
            if requires_patch and ctx is not None:
                patch_path = str(ctx.patch_path)
                git_before = ctx.git_commit_before
                git_after = ctx.git_commit_after
                workdir = ctx.worktree_dir
                input_script = ctx.map_to_worktree(input_script)
                job_snapshot.workdir = str(workdir)
                job_snapshot.input_script = str(input_script)
                source_root = ctx.worktree_dir
            else:
                git_before = get_git_head(actions_root)
                git_after = git_before

            if _requires_build(action):
                if not build_cfg:
                    raise RuntimeError("build config missing for build/source_patch action")
                final_build_cfg = _apply_build_config(build_cfg, action)
                build_config_diff_path = _write_build_config_diff(run_dir, build_cfg, final_build_cfg)
                build_output = build_job(final_build_cfg, source_root, run_dir)
                if not build_output.lammps_bin_path:
                    raise RuntimeError("build did not produce lammps binary")
                job_snapshot.lammps_bin = build_output.lammps_bin_path
                _append_trace(
                    trace_events,
                    run_trace,
                    {
                        "event": "build",
                        "agent": "ProfilerAgent",
                        "run_id": run_id,
                        "build_dir": build_output.build_dir,
                        "build_log": build_output.build_log_path,
                        "lammps_bin": build_output.lammps_bin_path,
                    },
                )

            binary_provenance = collect_binary_provenance(job_snapshot.lammps_bin, run_dir)

            run_output, profile = profiler.run(
                job_snapshot, run_args, env_overrides, workdir, run_dir, time_command, repeats
            )
            _append_trace(
                trace_events,
                run_trace,
                {
                    "event": "profile",
                    "agent": "ProfilerAgent",
                    "run_id": run_id,
                    "timing_breakdown": profile.timing_breakdown,
                    "system_metrics": profile.system_metrics,
                },
            )
            _append_trace(
                trace_events,
                run_trace,
                {
                    "event": "run_output_preview",
                    "agent": "ProfilerAgent",
                    "run_id": run_id,
                    "outputs": _collect_run_output_preview(run_output),
                },
            )
            if reporter:
                reporter.profile_result(run_output, profile)
            result = _build_result_ir(
                run_output, profile, baseline_runtime, runtime_agg, prior_samples
            )
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
            job=job_snapshot,
            action=action,
            patch_path=None,
            git_before=git_before,
            git_after=git_after,
            run_id=run_id,
            profile=profile,
            result=result,
            verdict=verdict,
            reasons=reasons,
        )
        if reporter:
            reporter.verify_result(exp)
        _write_experiment(run_dir, exp)
        _append_trace(
            trace_events,
            run_trace,
            {
                "event": "exception",
                "agent": "VerifierAgent",
                "run_id": run_id,
                "message": str(exc),
            },
        )
        _write_run_manifest(
            run_dir=run_dir,
            run_id=run_id,
            job=job_snapshot,
            action=action,
            env_overrides=env_overrides,
            run_args=run_args,
            git_before=git_before,
            git_after=git_after,
            patch_path=None,
            parent_run_id=parent_run_id,
            iteration=iteration,
            result=result,
            profile=profile,
            verify=None,
            run_output=None,
            repo_root=actions_root,
            build_output=build_output,
            binary_provenance=binary_provenance,
            build_config_diff_path=build_config_diff_path,
        )
        _write_run_trace(run_dir, run_trace)
        return exp

    if action and action.applies_to == ["run_config"]:
        patch_path = _write_run_config_diff(run_dir, job, run_args, env_overrides)

    verify = verifier.verify(job_snapshot, action, result, profile, gates, baseline_exp)
    result.correctness_metrics.update(verify.correctness_metrics)
    _append_trace(
        trace_events,
        run_trace,
        {
            "event": "verification",
            "agent": "VerifierAgent",
            "run_id": run_id,
            "verdict": verify.verdict,
            "reasons": verify.reasons,
            "correctness_metrics": verify.correctness_metrics,
        },
    )
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
    if reporter:
        reporter.verify_result(exp)
    _write_experiment(run_dir, exp)
    _write_run_manifest(
        run_dir=run_dir,
        run_id=run_id,
        job=job_snapshot,
        action=action,
        env_overrides=env_overrides,
        run_args=run_args,
        git_before=git_before,
        git_after=git_after,
        patch_path=patch_path,
        parent_run_id=parent_run_id,
        iteration=iteration,
        result=result,
        profile=profile,
        verify=verify,
        run_output=run_output,
        repo_root=actions_root,
        build_output=build_output,
        binary_provenance=binary_provenance,
        build_config_diff_path=build_config_diff_path,
    )
    _write_run_trace(run_dir, run_trace)
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


def _extract_patch_params(action: Optional[ActionIR], repo_root: Path) -> Dict[str, Optional[Path]]:
    if not action or "source_patch" not in action.applies_to:
        return {"patch_path": None, "patch_root": None}
    patch_path = action.parameters.get("patch_path")
    patch_root = action.parameters.get("patch_root")
    if patch_path:
        patch_path = Path(patch_path)
        if not patch_path.is_absolute():
            patch_path = (repo_root / patch_path).resolve()
    return {
        "patch_path": patch_path if patch_path else None,
        "patch_root": Path(patch_root) if patch_root else None,
    }


def _requires_build(action: Optional[ActionIR]) -> bool:
    if not action:
        return False
    return any(target in action.applies_to for target in ["build_config", "source_patch"])


def _apply_build_config(base_cfg: Dict[str, object], action: Optional[ActionIR]) -> Dict[str, object]:
    merged: Dict[str, object] = {**(base_cfg or {})}
    if not action:
        return merged
    build_params = action.parameters.get("build")
    if not isinstance(build_params, dict):
        return merged
    for key in ["source_dir", "generator", "target", "lammps_bin"]:
        if key in build_params:
            merged[key] = build_params[key]
    if "cmake_args" in build_params:
        merged["cmake_args"] = build_params["cmake_args"]
    if "build_args" in build_params:
        merged["build_args"] = build_params["build_args"]
    if "cmake_args_add" in build_params:
        merged.setdefault("cmake_args", [])
        merged["cmake_args"] = list(merged["cmake_args"]) + list(build_params["cmake_args_add"])
    if "build_args_add" in build_params:
        merged.setdefault("build_args", [])
        merged["build_args"] = list(merged["build_args"]) + list(build_params["build_args_add"])
    if "env" in build_params and isinstance(build_params["env"], dict):
        merged_env = dict(merged.get("env", {}) or {})
        merged_env.update(build_params["env"])
        merged["env"] = merged_env
    return merged


def _write_build_config_diff(
    run_dir: Path, build_before: Dict[str, object], build_after: Dict[str, object]
) -> str:
    diff = {"build_before": build_before, "build_after": build_after}
    diff_path = run_dir / "build_config.diff.json"
    diff_path.write_text(json.dumps(diff, indent=2), encoding="utf-8")
    return str(diff_path)


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


def _build_result_ir(
    run_output,
    profile: ProfileReport,
    baseline_runtime: Optional[float],
    runtime_agg: str,
    prior_samples: Optional[List[float]],
) -> ResultIR:
    timing = profile.timing_breakdown
    samples = (prior_samples or []) + (run_output.samples or [])
    aggregate = _aggregate_runtime(samples, runtime_agg) if samples else run_output.runtime_seconds
    total = timing.get("total", aggregate) or aggregate
    comm_ratio = (timing.get("comm", 0.0) / total) if total else 0.0
    derived = {"comm_ratio": comm_ratio}
    if baseline_runtime and aggregate:
        derived["speedup_vs_baseline"] = baseline_runtime / aggregate
    if len(samples) >= 2:
        mean = sum(samples) / len(samples)
        var = sum((x - mean) ** 2 for x in samples) / len(samples)
        derived["variance"] = var
        derived["variance_cv"] = (var ** 0.5) / mean if mean else 0.0
    return ResultIR(
        runtime_seconds=aggregate,
        derived_metrics=derived,
        correctness_metrics={},
        logs=[run_output.stdout_path, run_output.stderr_path, run_output.log_path],
        exit_code=run_output.exit_code,
        samples=samples,
    )


def _aggregate_runtime(samples: List[float], method: str) -> float:
    if not samples:
        return 0.0
    method_norm = (method or "mean").strip().lower()
    if method_norm == "median":
        return float(statistics.median(samples))
    return sum(samples) / len(samples)


def _evaluate_success(
    baseline_exp: ExperimentIR,
    best_exp: Optional[ExperimentIR],
    validated_exp: Optional[ExperimentIR],
    min_improvement_pct: float,
    validation_expected: bool,
) -> Dict[str, object]:
    baseline_runtime = baseline_exp.results.runtime_seconds
    candidate = validated_exp or best_exp
    success = False
    reason = "no candidate"
    improvement_pct = 0.0
    candidate_id = None
    if not candidate or candidate.action is None:
        reason = "no non-baseline candidate"
    elif baseline_runtime <= 0.0:
        reason = "invalid baseline runtime"
    else:
        candidate_id = candidate.run_id
        improvement_pct = (baseline_runtime - candidate.results.runtime_seconds) / baseline_runtime
        if validation_expected and not validated_exp:
            reason = "validation run skipped"
        elif validation_expected and validated_exp and validated_exp.verdict != "PASS":
            reason = "validation run failed"
        elif improvement_pct < min_improvement_pct:
            reason = "insufficient improvement"
        elif candidate.verdict != "PASS":
            reason = "candidate failed gates"
        else:
            success = True
            reason = "success"
    return {
        "success": success,
        "reason": reason,
        "target_improvement_pct": min_improvement_pct,
        "achieved_improvement_pct": improvement_pct,
        "candidate_run_id": candidate_id,
    }


def _write_agent_trace(artifacts_dir: Path, trace_events: List[Dict[str, object]]) -> str:
    path = artifacts_dir / "agent_trace.json"
    path.write_text(json.dumps(trace_events, indent=2), encoding="utf-8")
    return str(path)


def _write_run_trace(run_dir: Path, trace_events: List[Dict[str, object]]) -> None:
    path = run_dir / "agent_trace.json"
    path.write_text(json.dumps(trace_events, indent=2), encoding="utf-8")


def _build_llm_summary_zh(
    experiments: List[ExperimentIR],
    baseline: ExperimentIR,
    best: Optional[ExperimentIR],
    success_info: Optional[Dict[str, object]],
    llm_client: Optional[LLMClient],
) -> Optional[Dict[str, object]]:
    if not llm_client or not llm_client.config.enabled:
        return None
    payload = {
        "baseline_runtime_seconds": baseline.results.runtime_seconds,
        "best_run_id": best.run_id if best else None,
        "success_info": success_info,
        "experiments": [],
    }
    for exp in experiments:
        action_id = exp.action.action_id if exp.action else "baseline"
        family = exp.action.family if exp.action else None
        risk = exp.action.risk_level if exp.action else None
        description = exp.action.description if exp.action else None
        parameters = exp.action.parameters if exp.action else None
        expected_effect = exp.action.expected_effect if exp.action else None
        applies_to = exp.action.applies_to if exp.action else None
        correctness_skipped = exp.results.correctness_metrics.get("correctness_skipped_reason")
        payload["experiments"].append(
            {
                "run_id": exp.run_id,
                "action_id": action_id,
                "family": family,
                "risk_level": risk,
                "action_description": description,
                "action_parameters": parameters,
                "expected_effect": expected_effect,
                "applies_to": applies_to,
                "verdict": exp.verdict,
                "runtime_seconds": exp.results.runtime_seconds,
                "speedup_vs_baseline": exp.results.derived_metrics.get("speedup_vs_baseline"),
                "is_validation": exp.run_id.endswith("-validate"),
                "correctness_skipped": correctness_skipped,
                "reasons": exp.reasons,
            }
        )
    return llm_client.summarize_report_zh(payload)


def _build_llm_iteration_summary_zh(
    iteration: int,
    analysis: AnalysisResult,
    candidates: List[ActionIR],
    experiments: List[ExperimentIR],
    baseline: ExperimentIR,
    llm_client: Optional[LLMClient],
) -> Optional[Dict[str, object]]:
    if not llm_client or not llm_client.config.enabled:
        return None
    best_run_id = None
    best_runtime = None
    for exp in experiments:
        if exp.verdict != "PASS" or exp.action is None:
            continue
        if best_runtime is None or exp.results.runtime_seconds < best_runtime:
            best_runtime = exp.results.runtime_seconds
            best_run_id = exp.run_id
    payload = {
        "iteration": iteration,
        "baseline_runtime_seconds": baseline.results.runtime_seconds,
        "analysis": {
            "bottleneck": analysis.bottleneck,
            "allowed_families": analysis.allowed_families,
            "confidence": analysis.confidence,
        },
        "candidates": [action.action_id for action in candidates],
        "best_run_id": best_run_id,
        "experiments": [],
    }
    for exp in experiments:
        correctness_skipped = exp.results.correctness_metrics.get("correctness_skipped_reason")
        payload["experiments"].append(
            {
                "run_id": exp.run_id,
                "action_id": exp.action.action_id if exp.action else "baseline",
                "family": exp.action.family if exp.action else None,
                "action_description": exp.action.description if exp.action else None,
                "action_parameters": exp.action.parameters if exp.action else None,
                "expected_effect": exp.action.expected_effect if exp.action else None,
                "applies_to": exp.action.applies_to if exp.action else None,
                "verdict": exp.verdict,
                "runtime_seconds": exp.results.runtime_seconds,
                "speedup_vs_baseline": exp.results.derived_metrics.get("speedup_vs_baseline"),
                "correctness_skipped": correctness_skipped,
                "reasons": exp.reasons,
            }
        )
    return llm_client.summarize_iteration_zh(payload)


def _append_trace(
    global_trace: Optional[List[Dict[str, object]]],
    run_trace: Optional[List[Dict[str, object]]],
    event: Dict[str, object],
) -> None:
    if run_trace is not None:
        run_trace.append(event)
    if global_trace is not None:
        global_trace.append(event)


def _collect_run_output_preview(run_output) -> Dict[str, object]:
    return {
        "stdout": _read_file_preview(run_output.stdout_path, 8192),
        "stderr": _read_file_preview(run_output.stderr_path, 8192),
        "time": _read_file_preview(run_output.time_output_path, 8192),
        "log": _read_file_preview(run_output.log_path, 8192),
    }


def _read_file_preview(path: Optional[str], max_bytes: int) -> Optional[Dict[str, object]]:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    size = file_path.stat().st_size
    truncated = size > max_bytes
    if truncated:
        with file_path.open("rb") as handle:
            handle.seek(-max_bytes, 2)
            data = handle.read(max_bytes)
        text = data.decode("utf-8", errors="replace")
    else:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    return {"text": text, "size_bytes": size, "truncated": truncated}


def _select_direction_families(actions: List[ActionIR], profile: ProfileReport) -> List[str]:
    timing = profile.timing_breakdown
    total = timing.get("total", 0.0) or 0.0
    comm_ratio = (timing.get("comm", 0.0) / total) if total else 0.0
    output_ratio = (timing.get("output", 0.0) / total) if total else 0.0
    cpu = profile.system_metrics.get("cpu_percent_avg", 100.0)
    effects: set[str] = set()
    if comm_ratio > 0.2:
        effects.add("comm_reduce")
    if output_ratio > 0.2:
        effects.add("io_reduce")
    effects.add("compute_opt")
    if cpu < 70.0:
        effects.add("mem_locality")
    families = []
    for action in actions:
        if any(effect in effects for effect in action.expected_effect):
            families.append(action.family)
    if not families:
        families = [action.family for action in actions]
    return sorted(set(families))


def _write_run_manifest(
    run_dir: Path,
    run_id: str,
    job: JobIR,
    action: Optional[ActionIR],
    env_overrides: Dict[str, str],
    run_args: List[str],
    git_before: Optional[str],
    git_after: Optional[str],
    patch_path: Optional[str],
    parent_run_id: Optional[str],
    iteration: Optional[int],
    result: ResultIR,
    profile: ProfileReport,
    verify,
    run_output,
    repo_root: Path,
    build_output: Optional[BuildOutput],
    binary_provenance: Optional[Dict[str, object]],
    build_config_diff_path: Optional[str],
) -> None:
    manifest = _build_run_manifest(
        run_id=run_id,
        job=job,
        action=action,
        env_overrides=env_overrides,
        run_args=run_args,
        git_before=git_before,
        git_after=git_after,
        patch_path=patch_path,
        parent_run_id=parent_run_id,
        iteration=iteration,
        result=result,
        profile=profile,
        verify=verify,
        run_output=run_output,
        repo_root=repo_root,
        build_output=build_output,
        binary_provenance=binary_provenance,
        build_config_diff_path=build_config_diff_path,
    )
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _build_run_manifest(
    run_id: str,
    job: JobIR,
    action: Optional[ActionIR],
    env_overrides: Dict[str, str],
    run_args: List[str],
    git_before: Optional[str],
    git_after: Optional[str],
    patch_path: Optional[str],
    parent_run_id: Optional[str],
    iteration: Optional[int],
    result: ResultIR,
    profile: ProfileReport,
    verify,
    run_output,
    repo_root: Path,
    build_output: Optional[BuildOutput],
    binary_provenance: Optional[Dict[str, object]],
    build_config_diff_path: Optional[str],
) -> Dict[str, object]:
    git_status = get_git_status(repo_root)
    lammps_hash = _sha256_file(job.lammps_bin)
    input_hash = _sha256_file(job.input_script)
    artifacts = {
        "stdout": result.logs[0] if result.logs else None,
        "stderr": result.logs[1] if len(result.logs) > 1 else None,
        "log": result.logs[2] if len(result.logs) > 2 else None,
        "time": run_output.time_output_path if run_output else None,
        "patch": patch_path,
    }
    build_section = None
    if build_output:
        build_section = {
            "build_dir": build_output.build_dir,
            "build_log": build_output.build_log_path,
            "cmake_cache": build_output.cmake_cache_path,
            "compile_commands": build_output.compile_commands_path,
            "build_files": build_output.build_files,
            "lammps_bin_path": build_output.lammps_bin_path,
            "provenance_path": build_output.provenance_path,
            "provenance": build_output.provenance,
            "build_config_diff": build_config_diff_path,
        }
    verification = None
    if verify:
        verification = {
            "verdict": verify.verdict,
            "reasons": verify.reasons,
            "correctness_metrics": verify.correctness_metrics,
        }
    manifest = {
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "iteration": iteration,
        "action_id": action.action_id if action else "baseline",
        "action_family": action.family if action else None,
        "env_overrides": env_overrides,
        "env_final": job.env,
        "run_args": run_args,
        "code_version": {
            "git_commit_before": git_before,
            "git_commit_after": git_after,
            "dirty": git_status.get("dirty"),
        },
        "binary_version": {
            "path": job.lammps_bin,
            "sha256": lammps_hash,
        },
        "input": {
            "path": job.input_script,
            "sha256": input_hash,
        },
        "perf_summary": {
            "runtime_seconds": result.runtime_seconds,
            "derived_metrics": result.derived_metrics,
            "timing_breakdown": profile.timing_breakdown,
            "system_metrics": profile.system_metrics,
        },
        "verification": verification,
        "artifacts": artifacts,
        "build": build_section,
        "binary_provenance": binary_provenance,
    }
    return manifest


def _sha256_file(path_str: str) -> Optional[str]:
    try:
        path = Path(path_str)
    except TypeError:
        return None
    if not path.exists():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _append_run_index(
    artifacts_dir: Path,
    exp: ExperimentIR,
    parent_run_id: Optional[str],
    iteration: Optional[int],
) -> None:
    ledger_dir = artifacts_dir / "ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    path = ledger_dir / "run_index.jsonl"
    action_id = exp.action.action_id if exp.action else "baseline"
    speedup = exp.results.derived_metrics.get("speedup_vs_baseline")
    entry = {
        "run_id": exp.run_id,
        "parent_run_id": parent_run_id,
        "iteration": iteration,
        "action_id": action_id,
        "runtime_seconds": exp.results.runtime_seconds,
        "speedup_vs_baseline": speedup,
        "verdict": exp.verdict,
        "reasons": exp.reasons,
        "manifest_path": str((artifacts_dir / "runs" / exp.run_id / "manifest.json").resolve()),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def _write_iteration_summary(
    artifacts_dir: Path,
    iteration: int,
    analysis: AnalysisResult,
    candidates: List[ActionIR],
    experiments: List[ExperimentIR],
    baseline: ExperimentIR,
    llm_enabled: bool,
    llm_trace: Optional[Dict[str, object]],
    llm_summary_zh: Optional[Dict[str, object]],
) -> None:
    ledger_dir = artifacts_dir / "ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    summary_path = ledger_dir / f"iteration_{iteration:03d}_summary.md"
    summary_path_zh = ledger_dir / f"iteration_{iteration:03d}_summary_zh.md"

    lines = [
        f"# Iteration {iteration:03d} Summary",
        "",
        "## Analysis",
        f"- Bottleneck: {analysis.bottleneck}",
        f"- Allowed families: {', '.join(analysis.allowed_families)}",
        f"- Profiling confidence: {analysis.confidence:.2f}",
        f"- Ranking mode: {'llm' if llm_enabled else 'heuristic'}",
        "",
        "## Candidates",
    ]
    for action in candidates:
        lines.append(f"- {action.action_id} ({action.family})")
    lines.extend(["", "## Outcomes", ""])
    for exp in experiments:
        action_id = exp.action.action_id if exp.action else "baseline"
        speedup = exp.results.derived_metrics.get("speedup_vs_baseline")
        speedup_str = f"{speedup:.3f}x" if speedup else "n/a"
        reason_text = ", ".join(exp.reasons) if exp.reasons else "none"
        lines.append(
            f"- {action_id}: {exp.verdict}, {exp.results.runtime_seconds:.4f}s, speedup {speedup_str}, reasons: {reason_text}"
        )
    lines.extend(
        [
            "",
            "## Baseline",
            f"- runtime: {baseline.results.runtime_seconds:.4f}s",
        ]
    )
    if llm_trace:
        lines.extend(
            [
                "",
                "## LLM Explanation",
                llm_trace.get("explanation", ""),
                "",
            ]
        )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    summary_path_zh.write_text(
        _build_iteration_summary_zh(
            iteration=iteration,
            analysis=analysis,
            candidates=candidates,
            experiments=experiments,
            baseline=baseline,
            llm_enabled=llm_enabled,
            llm_trace=llm_trace,
            llm_summary_zh=llm_summary_zh,
        ),
        encoding="utf-8",
    )


def _build_iteration_summary_zh(
    iteration: int,
    analysis: AnalysisResult,
    candidates: List[ActionIR],
    experiments: List[ExperimentIR],
    baseline: ExperimentIR,
    llm_enabled: bool,
    llm_trace: Optional[Dict[str, object]],
    llm_summary_zh: Optional[Dict[str, object]],
) -> str:
    best_exp = None
    for exp in experiments:
        if exp.verdict != "PASS":
            continue
        if exp.action is None:
            continue
        if best_exp is None or exp.results.runtime_seconds < best_exp.results.runtime_seconds:
            best_exp = exp

    lines = [
        f"# 第 {iteration:03d} 轮摘要",
        "",
        "## 分析",
        f"- 瓶颈: {analysis.bottleneck}",
        f"- 允许动作族: {', '.join(analysis.allowed_families)}",
        f"- 画像置信度: {analysis.confidence:.2f}",
        f"- 排序模式: {'llm' if llm_enabled else 'heuristic'}",
        "",
        "## 候选动作",
    ]
    for action in candidates:
        lines.append(f"- {action.action_id} ({action.family})")
    lines.extend(["", "## 实验结果", ""])
    for exp in experiments:
        action_id = exp.action.action_id if exp.action else "baseline"
        speedup = exp.results.derived_metrics.get("speedup_vs_baseline")
        speedup_str = f"{speedup:.3f}x" if speedup else "n/a"
        reason_text = _llm_reason_for_run(llm_summary_zh, exp.run_id) if llm_summary_zh else (
            ", ".join(exp.reasons) if exp.reasons else "无"
        )
        lines.append(
            f"- {action_id}: {exp.verdict}, {exp.results.runtime_seconds:.4f}s, speedup {speedup_str}, 原因: {reason_text}"
        )
    detail_items = _collect_llm_details(llm_summary_zh, experiments) if llm_summary_zh else []
    if detail_items:
        lines.extend(["", "## 实验分析（逐条）", ""])
        for action_id, detail in detail_items:
            lines.append(f"### {action_id}")
            lines.append(detail)
    lines.extend(
        [
            "",
            "## 本轮选择",
            f"- Baseline 运行时间: {baseline.results.runtime_seconds:.4f}s",
        ]
    )
    if best_exp:
        best_action = best_exp.action.action_id if best_exp.action else "baseline"
        lines.append(
            f"- 本轮最优: {best_action} ({best_exp.results.runtime_seconds:.4f}s)"
        )
        speedup = best_exp.results.derived_metrics.get("speedup_vs_baseline")
        if speedup is not None:
            lines.append(f"- 相对基线提升: {speedup:.3f}x")
    else:
        lines.append("- 本轮最优: 无（无通过的候选）")
    summary_lines = _extract_llm_summary_lines(llm_summary_zh) if llm_summary_zh else []
    if summary_lines:
        lines.extend(["", "## 简要分析"])
        lines.extend([f"- {line}" for line in summary_lines])
    selection_reason = _extract_llm_selection_reason(llm_summary_zh)
    if selection_reason:
        lines.extend(["", "## 选择理由", f"- {selection_reason}"])
    if llm_trace:
        lines.extend(
            [
                "",
                "## LLM 解释",
                llm_trace.get("explanation", ""),
                "",
            ]
        )
    return "\n".join(lines)


def _extract_llm_summary_lines(llm_summary: Optional[Dict[str, object]]) -> List[str]:
    if not llm_summary:
        return []
    lines = llm_summary.get("summary_lines")
    if isinstance(lines, list):
        return [str(item) for item in lines]
    return []


def _llm_reason_for_run(llm_summary: Optional[Dict[str, object]], run_id: str) -> str:
    if not llm_summary:
        return "未提供原因"
    reasons = llm_summary.get("experiment_reasons", {})
    if isinstance(reasons, dict):
        return str(reasons.get(run_id, "未提供原因"))
    return "未提供原因"


def _collect_llm_details(
    llm_summary: Optional[Dict[str, object]],
    experiments: List[ExperimentIR],
) -> List[tuple[str, str]]:
    if not llm_summary:
        return []
    details = llm_summary.get("experiment_details", {})
    if not isinstance(details, dict):
        return []
    items: List[tuple[str, str]] = []
    for exp in experiments:
        action_id = exp.action.action_id if exp.action else "baseline"
        detail = details.get(exp.run_id)
        if detail:
            items.append((action_id, str(detail)))
    return items


def _extract_llm_selection_reason(llm_summary: Optional[Dict[str, object]]) -> Optional[str]:
    if not llm_summary:
        return None
    selection_reason = llm_summary.get("selection_reason")
    if selection_reason:
        return str(selection_reason)
    return None


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
