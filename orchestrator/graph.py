from __future__ import annotations

import base64
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import statistics
import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from schemas.action_ir import ActionIR, VerificationPlan
from schemas.analysis_ir import AnalysisResult
from schemas.experiment_ir import ExperimentIR
from schemas.job_ir import JobIR, Budgets
from schemas.profile_report import ProfileReport
from schemas.patch_proposal_ir import PatchProposal
from schemas.result_ir import ResultIR
from schemas.decision_ir import DecisionIR
from skills.build_local import BuildOutput, build_job, collect_binary_provenance
from skills.run_local import RunOutput
from skills.system_caps import collect_system_caps
from skills.profile_features import build_profile_features
from skills.profile_payload import build_profile_payload
from skills.action_domain import enforce_candidate_policy, select_actions_for_direction
from skills.experience_memory import ExperienceMemory
from skills.patch_triage import validate_patch_action
from skills.patch_review import review_patch_diff
from skills.lammps_templates import get_template_context
from skills.applications import apply_adapter as apply_app_adapter
from skills.applications import ensure_log_path as app_ensure_log_path
from skills.applications import input_edit_allowlist as app_input_allowlist
from skills.patch_git import (
    GitPatchContext,
    WorktreeAddError,
    _strip_patch_prefix,
    get_git_head,
    get_git_status,
)
from orchestrator.console import ConsoleUI
from orchestrator.agents import (
    ExecutorAgent,
    IdeaAgent,
    CodeSurveyAgent,
    ActionSynthAgent,
    PlannerAgent,
    ProfilerAgent,
    OptimizerAgent,
    PatchDebugAgent,
    ReporterAgent,
    CodePatchAgent,
    create_agentic_code_patch_agent,
    OrchestratorAgent,
    ParameterExplorerAgent,
    ReflectionAgent,
    PatchReviewAgent,
    ReviewerAgent,
    RouterRankerAgent,
    TriageAgent,
    VerifierAgent,
)
from schemas.code_analysis_ir import DeepCodeAnalysisResult
from schemas.opportunity_graph import (
    OpportunityGraph,
    OpportunityGraphResult,
    OpportunityStatus,
    SelectedOpportunities,
)
from schemas.ranking_ir import RankedAction
from schemas.reflection_ir import ReflectionResult
from orchestrator.agent_llm import AgentConfig
from orchestrator.errors import LLMUnavailableError
from orchestrator.memory import OptimizationMemory
from orchestrator.llm_client import LLMClient
from orchestrator.router import RuleContext, filter_actions
from orchestrator.stop import StopState, should_stop

_EXPECTED_EFFECTS = {
    "comm_reduce",
    "mem_locality",
    "compute_opt",
    "imbalance_reduce",
    "io_reduce",
}


def _slug_token(value: str, max_len: int = 64) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    if not text:
        text = "id"
    if len(text) > max_len:
        text = text[:max_len].rstrip("._-")
    return text or "id"


def _stable_compact_id(prefix: str, *parts: str, max_len: int = 120) -> str:
    raw = "|".join(str(part or "") for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    readable = ".".join(_slug_token(str(part), max_len=28) for part in parts if part)
    base = f"{prefix}.{readable}" if readable else prefix
    keep = max_len - len(digest) - 1
    if keep < 8:
        keep = 8
    if len(base) > keep:
        base = base[:keep].rstrip("._-")
    return f"{base}.{digest}"


def _iter_exp_id(iteration: int, action_id: str) -> str:
    return _stable_compact_id(f"iter{iteration}", action_id, max_len=120).replace(".", "-", 1)


def _resolve_patch_family_hint(
    patch_families: Optional[Dict[str, object]],
    family_hint: Optional[str],
    category_hint: Optional[str] = None,
) -> Optional[str]:
    families = (patch_families or {}).get("families", [])
    known_ids: List[str] = []
    for item in families:
        if isinstance(item, dict) and item.get("id"):
            known_ids.append(str(item["id"]))
    if not known_ids:
        hint = str(family_hint or category_hint or "").strip()
        return hint or None

    lookup: Dict[str, str] = {}
    for family_id in known_ids:
        lookup[family_id] = family_id
        lookup[family_id.lower()] = family_id
        lookup[_slug_token(family_id.lower(), max_len=80)] = family_id

    raw_hint = str(family_hint or category_hint or "").strip()
    if not raw_hint:
        return None
    norm_hint = _slug_token(raw_hint.lower(), max_len=80)
    if raw_hint in lookup:
        return lookup[raw_hint]
    if raw_hint.lower() in lookup:
        return lookup[raw_hint.lower()]
    if norm_hint in lookup:
        return lookup[norm_hint]

    text = f"{raw_hint} {category_hint or ''}".lower()
    alias_map: List[Tuple[str, str]] = [
        ("vector", "vectorization_hints"),
        ("simd", "vectorization_hints"),
        ("unroll", "loop_unroll"),
        ("fusion", "loop_fusion"),
        ("fission", "loop_fission"),
        ("hoist", "hoist_invariant"),
        ("invariant", "hoist_invariant"),
        ("cache", "cache_local_pointers"),
        ("pointer", "cache_local_pointers"),
        ("array", "array_packing"),
        ("layout", "array_packing"),
        ("soa", "array_packing"),
    ]
    for needle, family_id in alias_map:
        if needle in text and family_id in lookup:
            return lookup[family_id]

    for family_id in known_ids:
        if family_id.lower() in text:
            return family_id
    return None


def _patch_family_effects(
    patch_families: Optional[Dict[str, object]], family_id: Optional[str]
) -> List[str]:
    """Look up patch_tags for a family and return those that are valid expected_effects."""
    if not patch_families or not family_id:
        return []
    for item in patch_families.get("families", []):
        if isinstance(item, dict) and item.get("id") == family_id:
            tags = item.get("patch_tags", [])
            return [t for t in tags if t in _EXPECTED_EFFECTS]
    return []


def _patch_family_gates(
    patch_families: Optional[Dict[str, object]], family_id: Optional[str]
) -> List[str]:
    """Return mandatory gates for a patch family, if any."""
    if not patch_families or not family_id:
        return []
    for item in patch_families.get("families", []):
        if isinstance(item, dict) and item.get("id") == family_id:
            return list(item.get("mandatory_gates", []) or [])
    return []


def _best_pass_exp(experiments: List[ExperimentIR]) -> Optional[ExperimentIR]:
    best = None
    for exp in experiments:
        if exp.verdict != "PASS":
            continue
        if best is None or exp.results.runtime_seconds < best.results.runtime_seconds:
            best = exp
    return best


def _resolve_app_repo_root(
    repo_root: Path,
    job: JobIR,
    adapter_cfg: Optional[Dict[str, object]] = None,
) -> Path:
    """Resolve the repository root used by app-specific agents.

    Priority:
    1) adapter.patch_root (if provided)
    2) legacy third_party/<app> directory (if it exists)
    3) repository root fallback
    """
    if isinstance(adapter_cfg, dict):
        patch_root = adapter_cfg.get("patch_root")
        if isinstance(patch_root, str) and patch_root.strip():
            candidate = (repo_root / patch_root).resolve()
            if candidate.exists():
                return candidate
    legacy = (repo_root / "third_party" / str(job.app)).resolve()
    if legacy.exists():
        return legacy
    return repo_root


def _latest_backend_exp(experiments: List[ExperimentIR]) -> Optional[ExperimentIR]:
    latest = None
    for exp in experiments:
        if exp.verdict != "PASS" or not exp.action:
            continue
        if exp.action.family == "runtime_backend_select":
            latest = exp
            continue
        params = exp.action.parameters or {}
        if params.get("backend_enable"):
            latest = exp
    return latest


def _run_args_has_backend(run_args: List[str]) -> bool:
    if not run_args:
        return False
    for idx, arg in enumerate(run_args):
        if arg == "-sf" and idx + 1 < len(run_args) and run_args[idx + 1] == "omp":
            return True
    return False


def _backend_from_args(run_args: List[str]) -> Optional[str]:
    if not run_args:
        return None
    for idx, arg in enumerate(run_args):
        if arg == "-sf" and idx + 1 < len(run_args):
            return str(run_args[idx + 1])
    return None


def _run_args_has_flag(run_args: List[str], flag: str) -> bool:
    return any(str(arg) == flag for arg in (run_args or []))


def _is_pthread_cli_model(
    run_args: List[str],
    actions: List[ActionIR],
) -> bool:
    if _run_args_has_backend(run_args):
        return False
    if not _run_args_has_flag(run_args, "-t"):
        return False
    has_pthread_family = any(action.family == "parallel_pthread" for action in actions)
    return has_pthread_family


def _runtime_family_preference(
    run_args: List[str],
    actions: List[ActionIR],
) -> List[str]:
    if _is_pthread_cli_model(run_args, actions):
        return ["parallel_pthread", "affinity_tune", "wait_policy", "sched_granularity", "parallel_omp"]
    if _backend_from_args(run_args) == "omp":
        return ["parallel_omp", "affinity_tune", "wait_policy", "sched_granularity", "parallel_pthread"]
    return ["parallel_omp", "parallel_pthread", "affinity_tune", "wait_policy", "sched_granularity"]


def _is_parallel_threads_action(action: Optional[ActionIR]) -> bool:
    if not action or not action.parameters:
        return False
    env = action.parameters.get("env", {})
    if not isinstance(env, dict):
        return False
    return "OMP_NUM_THREADS" in env


def _best_for_target(experiments: List[ExperimentIR], target: str) -> Optional[ExperimentIR]:
    best = None
    for exp in experiments:
        if exp.action is None or exp.verdict != "PASS":
            continue
        applies_to = set(exp.action.applies_to or [])
        if target == "runtime_tier":
            if not ({"run_config", "input_script"} & applies_to):
                continue
        elif target not in applies_to:
            continue
        if best is None or exp.results.runtime_seconds < best.results.runtime_seconds:
            best = exp
    return best


def _best_effective_for_target(
    experiments: List[ExperimentIR],
    target: str,
    baseline_runtime: float,
    min_improvement_pct: float,
) -> Optional[ExperimentIR]:
    best = None
    for exp in experiments:
        if exp.action is None or exp.verdict != "PASS":
            continue
        applies_to = set(exp.action.applies_to or [])
        if target == "runtime_tier":
            if not ({"run_config", "input_script"} & applies_to):
                continue
        elif target not in applies_to:
            continue
        if baseline_runtime <= 0:
            continue
        improvement = (baseline_runtime - exp.results.runtime_seconds) / baseline_runtime
        if improvement < min_improvement_pct:
            continue
        if best is None or exp.results.runtime_seconds < best.results.runtime_seconds:
            best = exp
    return best


def _best_effective_action_set(
    experiments: List[ExperimentIR],
    baseline_runtime: float,
    min_improvement_pct: float,
) -> Dict[str, str]:
    targets = ["run_config", "input_script", "build_config", "source_patch"]
    best_actions: Dict[str, str] = {}
    for target in targets:
        best = _best_effective_for_target(
            experiments, target, baseline_runtime, min_improvement_pct
        )
        if best and best.action:
            best_actions[target] = best.action.action_id
    return best_actions


def _best_action_chain(
    experiments: List[ExperimentIR],
    best_exp: Optional[ExperimentIR],
) -> List[str]:
    if not best_exp:
        return []
    by_run: Dict[str, ExperimentIR] = {exp.run_id: exp for exp in experiments}
    chain: List[str] = []
    current = best_exp
    seen: set[str] = set()
    while current and current.run_id not in seen:
        seen.add(current.run_id)
        if current.action:
            chain.append(current.action.action_id)
        if not current.base_run_id:
            break
        current = by_run.get(current.base_run_id)
    chain.reverse()
    return chain


def _select_actions_from_decision(
    eligible_pool: List[ActionIR],
    decision: DecisionIR,
    top_k: int,
    action_by_cid: Optional[Dict[int, ActionIR]] = None,
) -> List[ActionIR]:
    if not decision:
        return []
    ordered_cids = list(decision.candidate_cids or decision.ranking_cids or [])
    if not ordered_cids and not (decision.candidates or decision.ranking):
        return []
    action_map = {action.action_id: action for action in eligible_pool}
    action_map_lower = {action_id.lower(): action_id for action_id in action_map}
    action_map_canonical = {
        _canonical_action_id(action_id): action_id for action_id in action_map
    }
    selected: List[ActionIR] = []
    seen: set[str] = set()
    if ordered_cids:
        cid_map = action_by_cid or {}
        for cid in ordered_cids:
            try:
                cid_int = int(cid)
            except (TypeError, ValueError):
                continue
            action = cid_map.get(cid_int)
            if not action:
                continue
            resolved_id = action.action_id
            if resolved_id in seen or resolved_id not in action_map:
                continue
            selected.append(action_map[resolved_id])
            seen.add(resolved_id)

    # Backward-compatible fallback (legacy decision format by action_id string).
    if not selected:
        ordered_ids = list(decision.candidates or decision.ranking or [])
        for requested_id in ordered_ids:
            resolved_id = _resolve_decision_action_id(
                str(requested_id),
                eligible_pool,
                action_map,
                action_map_lower,
                action_map_canonical,
            )
            if not resolved_id or resolved_id in seen:
                continue
            action = action_map.get(resolved_id)
            if action:
                selected.append(action)
                seen.add(resolved_id)
    limit = decision.max_candidates if decision.max_candidates is not None else top_k
    if limit and len(selected) > limit:
        selected = selected[:limit]
    return selected


def _canonical_action_id(action_id: str) -> str:
    lowered = action_id.strip().lower()
    lowered = lowered.replace(":", "_")
    lowered = lowered.replace("-", "_")
    lowered = re.sub(r"[^a-z0-9._]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered)
    return lowered


def _thread_count_from_action(action: ActionIR) -> Optional[int]:
    params = action.parameters or {}
    env = params.get("env")
    if isinstance(env, dict):
        raw = env.get("OMP_NUM_THREADS")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass
    run_args_cfg = params.get("run_args")
    if isinstance(run_args_cfg, dict):
        for entry in run_args_cfg.get("set_flags", []) or []:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("flag", "")) != "-t":
                continue
            values = entry.get("values", [])
            if not isinstance(values, list) or not values:
                continue
            try:
                return int(values[0])
            except (TypeError, ValueError):
                continue
    match = re.search(r"\.t(\d+)(?:_|$)", action.action_id)
    if match:
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None
    return None


def _resolve_decision_action_id(
    requested_id: str,
    eligible_pool: List[ActionIR],
    action_map: Dict[str, ActionIR],
    action_map_lower: Dict[str, str],
    action_map_canonical: Dict[str, str],
) -> Optional[str]:
    if requested_id in action_map:
        return requested_id

    rid_lower = requested_id.lower().strip()
    if rid_lower in action_map_lower:
        return action_map_lower[rid_lower]

    rid_canonical = _canonical_action_id(requested_id)
    if rid_canonical in action_map_canonical:
        return action_map_canonical[rid_canonical]

    # Handle thread-template aliases produced by LLMs, e.g.
    # "parallel_pthread.template_threads:4" or "parallel_omp.template_close_cores:16".
    tmpl_match = re.match(
        r"^(?P<family>[a-z0-9_]+)\.template_(?P<template>[a-z0-9_]+):(?P<threads>\d+)$",
        rid_lower,
    )
    if tmpl_match:
        family = tmpl_match.group("family")
        template = tmpl_match.group("template")
        threads = int(tmpl_match.group("threads"))
        by_family_threads = [
            action
            for action in eligible_pool
            if action.family == family and _thread_count_from_action(action) == threads
        ]
        if by_family_threads:
            if template not in {"threads", "template"}:
                for action in by_family_threads:
                    params = action.parameters or {}
                    tid = str(params.get("template_id", "")).lower()
                    aid = action.action_id.lower()
                    if tid == template or aid.endswith(f"_{template}") or f"_{template}." in aid:
                        return action.action_id
            return by_family_threads[0].action_id
    return None


def _tail_text(path: Path, max_bytes: int = 4096, max_lines: int = 40) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    lines = data.splitlines()
    tail_lines = lines[-max_lines:] if max_lines > 0 else lines
    text = "\n".join(tail_lines)
    if len(text) > max_bytes:
        text = text[-max_bytes:]
    return text


def _has_tau_sigtrap(run_dir: Path) -> bool:
    for path in run_dir.glob("stderr*.log"):
        if path.name.endswith(".pre_tau_retry"):
            continue
        text = _tail_text(path, max_bytes=8192, max_lines=80)
        if "Trace/BPT trap" in text or "SIGTRAP" in text:
            return True
    return False


def _bwa_run_complete(run_dir: Path) -> bool:
    """BWA run completeness check.

    BWA prints a final summary line with "[main] Real time:" at the end of stderr/time output.
    Missing this line indicates the run terminated early (often due to unstable sampling).
    """
    logs = sorted(run_dir.glob("stderr*.log"))
    if not logs:
        return True
    for path in logs:
        if path.name.endswith(".pre_tau_retry"):
            continue
        text = _tail_text(path, max_bytes=20000, max_lines=200)
        if "[main] Real time" not in text:
            return False
    return True


def _bwa_per_repeat_complete(run_dir: Path, n_repeats: int) -> List[bool]:
    """Per-repeat BWA completeness check.

    Returns a list of booleans, one per repeat, indicating whether that
    repeat produced a complete BWA output (has '[main] Real time' in stderr).
    """
    result: List[bool] = []
    for idx in range(n_repeats):
        if n_repeats == 1:
            stderr_path = run_dir / "stderr.log"
        else:
            stderr_path = run_dir / f"stderr_{idx}.log"
        if not stderr_path.is_file():
            result.append(True)  # no file → can't tell, assume ok
            continue
        text = _tail_text(stderr_path, max_bytes=20000, max_lines=200)
        result.append("[main] Real time" in text)
    return result


def _backup_run_logs(run_dir: Path, suffix: str) -> None:
    for pattern in ("stdout*.log", "stderr*.log", "time*.log"):
        for path in run_dir.glob(pattern):
            if not path.is_file():
                continue
            backup = path.with_name(f"{path.name}.{suffix}")
            try:
                path.rename(backup)
            except OSError:
                continue


def _tau_retry_env(env: Dict[str, str]) -> Dict[str, str]:
    updated = dict(env)
    updated["TAU_SAMPLING"] = "1"
    updated.setdefault("TAU_EBS_SOURCE", "itimer")
    updated.setdefault("TAU_EBS_UNWIND", "0")
    period = 0
    try:
        period = int(updated.get("TAU_EBS_PERIOD", "0") or 0)
    except ValueError:
        period = 0
    if period <= 0:
        period = 100000
    updated["TAU_EBS_PERIOD"] = str(max(period, 200000))
    return updated


def _collect_failure_feedback(
    exp: ExperimentIR, artifacts_dir: Path
) -> Optional[Dict[str, object]]:
    if exp.verdict != "FAIL":
        return None
    run_dir = artifacts_dir / "runs" / exp.run_id
    stderr_tail = _tail_text(run_dir / "stderr.log")
    stdout_tail = _tail_text(run_dir / "stdout.log")
    log_tail = ""
    for log_entry in reversed(exp.results.logs or []):
        candidate = Path(str(log_entry))
        if not candidate.is_absolute():
            candidate = (run_dir / candidate).resolve()
        if candidate.is_file():
            log_tail = _tail_text(candidate)
            break
    if not log_tail:
        log_tail = _tail_text(run_dir / "log.lammps")
    return {
        "action_id": exp.action.action_id if exp.action else "baseline",
        "run_id": exp.run_id,
        "exit_code": exp.results.exit_code,
        "reasons": exp.reasons,
        "stderr_tail": stderr_tail,
        "stdout_tail": stdout_tail,
        "log_tail": log_tail,
    }


def _effective_action_combo(
    experiments: List[ExperimentIR],
    best_exp: Optional[ExperimentIR],
) -> List[str]:
    if not best_exp:
        return []
    by_run: Dict[str, ExperimentIR] = {exp.run_id: exp for exp in experiments}
    applied: Dict[str, str] = {}
    current = best_exp
    seen: set[str] = set()
    while current and current.run_id not in seen:
        seen.add(current.run_id)
        if current.action and current.action.applies_to:
            for target in current.action.applies_to:
                if target not in applied:
                    applied[target] = current.action.action_id
        if not current.base_run_id:
            break
        current = by_run.get(current.base_run_id)
    ordered = ["run_config", "input_script", "build_config", "source_patch"]
    combo: List[str] = []
    for key in ordered:
        if key in applied:
            combo.append(f"{key}={applied[key]}")
    for key in sorted(k for k in applied.keys() if k not in ordered):
        combo.append(f"{key}={applied[key]}")
    return combo


def _extract_variance_cv(exp: ExperimentIR) -> Optional[float]:
    if not exp or not exp.results:
        return None
    cv = exp.results.derived_metrics.get("variance_cv")
    if isinstance(cv, (int, float)):
        return float(cv)
    cv = exp.results.correctness_metrics.get("variance_cv")
    if isinstance(cv, (int, float)):
        return float(cv)
    return None


def _action_env_overrides(action: Optional[ActionIR]) -> Dict[str, str]:
    if not action or not action.parameters:
        return {}
    params = action.parameters or {}
    env: Dict[str, str] = {}
    for key in ("env", "env_overrides", "env_vars"):
        value = params.get(key)
        if isinstance(value, dict):
            env.update({str(k): str(v) for k, v in value.items()})
    return env


def _is_volatile_action(action: Optional[ActionIR]) -> bool:
    env = _action_env_overrides(action)
    if not env:
        return False
    dynamic = env.get("OMP_DYNAMIC")
    if isinstance(dynamic, str) and dynamic.lower() in {"1", "true", "yes", "on"}:
        return True
    nested = env.get("OMP_NESTED")
    if isinstance(nested, str) and nested.lower() in {"1", "true", "yes", "on"}:
        return True
    max_levels = env.get("OMP_MAX_ACTIVE_LEVELS")
    if isinstance(max_levels, str):
        try:
            if int(max_levels) > 1:
                return True
        except ValueError:
            pass
    return False


def _merge_run_args_cfg(
    base: Optional[Dict[str, object]],
    extra: Optional[Dict[str, object]],
) -> Dict[str, object]:
    merged: Dict[str, object] = {}
    if isinstance(base, dict):
        merged.update(base)
    if not isinstance(extra, dict):
        return merged
    set_flags = list(merged.get("set_flags", []))
    seen = {item.get("flag") for item in set_flags if isinstance(item, dict)}
    for item in extra.get("set_flags", []):
        if not isinstance(item, dict):
            continue
        flag = item.get("flag")
        if flag in seen:
            continue
        set_flags.append(item)
        seen.add(flag)
    if set_flags:
        merged["set_flags"] = set_flags
    return merged


def _merge_action_params(actions: List[ActionIR]) -> Dict[str, object]:
    params: Dict[str, object] = {}
    env: Dict[str, str] = {}
    run_args_cfg: Dict[str, object] = {}
    for action in actions:
        for key, value in (action.parameters or {}).items():
            if key == "env" and isinstance(value, dict):
                env.update({str(k): str(v) for k, v in value.items()})
                continue
            if key == "run_args" and isinstance(value, dict):
                run_args_cfg = _merge_run_args_cfg(run_args_cfg, value)
                continue
            params[key] = value
    if env:
        params["env"] = env
    if run_args_cfg:
        params["run_args"] = run_args_cfg
    return params


def _actions_conflict(a: ActionIR, b: ActionIR) -> bool:
    """Check if two actions conflict (cannot be composed).

    Two actions conflict if they touch overlapping resources:
    - Same family (e.g. two parallel_omp variants)
    - Both are build_config
    - source_patch touching the same files
    - Overlapping env variables
    - Overlapping run_args flags
    """
    # Same family always conflicts
    if a.family == b.family:
        return True
    # build_config cannot compose with another build_config
    if "build_config" in (a.applies_to or []) and "build_config" in (b.applies_to or []):
        return True
    # source_patch: conflict if touching same files
    a_is_patch = "source_patch" in (a.applies_to or [])
    b_is_patch = "source_patch" in (b.applies_to or [])
    if a_is_patch and b_is_patch:
        files_a = set((a.parameters or {}).get("patch_files", []))
        files_b = set((b.parameters or {}).get("patch_files", []))
        target_a = (a.parameters or {}).get("target_file", "")
        target_b = (b.parameters or {}).get("target_file", "")
        if target_a:
            files_a.add(target_a)
        if target_b:
            files_b.add(target_b)
        if files_a & files_b:
            return True
    # env variable overlap
    env_a = set((a.parameters or {}).get("env", {}).keys()) if isinstance((a.parameters or {}).get("env"), dict) else set()
    env_b = set((b.parameters or {}).get("env", {}).keys()) if isinstance((b.parameters or {}).get("env"), dict) else set()
    if env_a & env_b:
        return True
    # run_args flag overlap
    ra_a = (a.parameters or {}).get("run_args", {})
    ra_b = (b.parameters or {}).get("run_args", {})
    flags_a = {f.get("flag") for f in (ra_a.get("set_flags", []) if isinstance(ra_a, dict) else []) if isinstance(f, dict)}
    flags_b = {f.get("flag") for f in (ra_b.get("set_flags", []) if isinstance(ra_b, dict) else []) if isinstance(f, dict)}
    if flags_a & flags_b:
        return True
    return False


def _select_non_conflicting(
    experiments: List[ExperimentIR],
    baseline_runtime: float,
    min_improvement_pct: float,
    variance_cfg: Optional[Dict[str, object]] = None,
    variance_repeats: int = 2,
    baseline_run_id: Optional[str] = None,
) -> List[ExperimentIR]:
    """Select the best non-conflicting set of improvements from an iteration.

    Returns a list of experiments sorted by runtime (fastest first), where no
    two experiments have conflicting actions. Uses a greedy approach: start
    with the best, add remaining if they don't conflict.
    """
    runtime_lookup: Dict[str, float] = {}
    for exp in experiments:
        if exp and exp.results and exp.results.runtime_seconds > 0:
            runtime_lookup[exp.run_id] = exp.results.runtime_seconds
    if baseline_run_id and baseline_runtime > 0:
        runtime_lookup.setdefault(baseline_run_id, baseline_runtime)
    if not runtime_lookup and baseline_runtime <= 0:
        return []

    cv_max = 1.0
    if isinstance(variance_cfg, dict):
        cv_max = float(variance_cfg.get("cv_max", 1.0) or 1.0)

    improved: List[ExperimentIR] = []
    for exp in experiments:
        if exp.verdict != "PASS" or not exp.action:
            continue
        base_runtime = baseline_runtime
        if exp.base_run_id and exp.base_run_id in runtime_lookup:
            base_runtime = runtime_lookup[exp.base_run_id]
        if base_runtime <= 0:
            continue
        gain = (base_runtime - exp.results.runtime_seconds) / base_runtime
        if gain < min_improvement_pct:
            continue
        # Skip unstable volatile actions
        if _is_volatile_action(exp.action):
            variance_cv = _extract_variance_cv(exp)
            samples = exp.results.samples or []
            if variance_cv is None or len(samples) < variance_repeats or variance_cv > cv_max:
                continue
        improved.append(exp)

    # Sort by runtime (fastest first)
    improved.sort(key=lambda e: e.results.runtime_seconds)

    if len(improved) <= 1:
        return improved

    # Greedy: start with best, add non-conflicting
    selected = [improved[0]]
    for exp in improved[1:]:
        conflicts = False
        for sel in selected:
            if _actions_conflict(exp.action, sel.action):
                conflicts = True
                break
        if not conflicts:
            selected.append(exp)

    return selected


def _build_final_composite_action(
    components: List[ExperimentIR],
    gates: Dict[str, object],
) -> Optional[ActionIR]:
    if len(components) < 2:
        return None
    actions = [exp.action for exp in components if exp.action]
    if len(actions) < 2:
        return None
    applies: List[str] = sorted({t for action in actions for t in (action.applies_to or [])})
    expected = sorted({e for action in actions for e in (action.expected_effect or [])})
    params = _merge_action_params(actions)
    if "source_patch" in applies:
        stacked_paths: List[str] = []
        for action in actions:
            if "source_patch" not in (action.applies_to or []):
                continue
            raw_paths = (action.parameters or {}).get("patch_paths")
            if isinstance(raw_paths, list):
                for item in raw_paths:
                    if isinstance(item, str) and item and item not in stacked_paths:
                        stacked_paths.append(item)
            raw_path = (action.parameters or {}).get("patch_path")
            if isinstance(raw_path, str) and raw_path and raw_path not in stacked_paths:
                stacked_paths.append(raw_path)
        if stacked_paths:
            params["patch_paths"] = stacked_paths
            params.pop("patch_path", None)
    variance_cfg = gates.get("variance", {}) if isinstance(gates, dict) else {}
    thresholds = {}
    if isinstance(variance_cfg, dict) and variance_cfg.get("cv_max") is not None:
        thresholds["variance_cv_max"] = float(variance_cfg.get("cv_max"))
    risk = "high" if any("source_patch" in action.applies_to for action in actions) else "medium"
    return ActionIR(
        action_id="final_composite",
        family="final_composite",
        description="Composite of best effective run/input/build/source_patch actions.",
        applies_to=applies,
        parameters=params,
        expected_effect=expected,
        risk_level=risk,
        verification_plan=VerificationPlan(
            gates=["runtime", "correctness", "variance"],
            thresholds=thresholds,
        ),
    )


def _opportunities_to_actions(
    analysis: DeepCodeAnalysisResult,
    patch_families: Optional[Dict[str, object]],
    repo_root: Optional[Path] = None,
    patch_root: str = "",
) -> List[ActionIR]:
    """Convert deep analysis opportunities to ActionIR for Phase 2 consumption."""
    _EFFECTS = {"comm_reduce", "mem_locality", "compute_opt", "imbalance_reduce", "io_reduce"}

    def _norm(raw: str) -> str:
        """Normalize a path to repo-relative if repo_root is available."""
        if not repo_root or not raw:
            return raw
        normed = _normalise_path_to_repo_rel(raw, repo_root, patch_root)
        return normed if normed else raw

    # Respect recommended_sequence if available, else sort by priority_rank
    if analysis.recommended_sequence:
        opp_map = {op.opportunity_id: op for op in analysis.opportunities}
        ordered = [opp_map[oid] for oid in analysis.recommended_sequence if oid in opp_map]
        remaining = [
            op for op in analysis.opportunities
            if op.opportunity_id not in set(analysis.recommended_sequence)
        ]
        ordered.extend(remaining)
    else:
        ordered = sorted(analysis.opportunities, key=lambda op: op.priority_rank)

    actions: List[ActionIR] = []
    for op in ordered:
        resolved_family = _resolve_patch_family_hint(
            patch_families,
            op.family_hint,
            op.category,
        )
        effects = [e for e in op.expected_effect if e in _EFFECTS]
        if not effects and resolved_family:
            effects = _patch_family_effects(patch_families, resolved_family)
        if not effects:
            effects = ["compute_opt"]

        gates = _patch_family_gates(patch_families, resolved_family) if resolved_family else []
        if not gates:
            gates = ["runtime", "correctness", "variance"]

        norm_target_files = [_norm(tf) for tf in op.target_files]
        norm_target_file = norm_target_files[0] if norm_target_files else ""

        action = ActionIR(
            action_id=_stable_compact_id("deep", op.opportunity_id, max_len=96),
            family="source_patch",
            description=f"{op.title}: {op.mechanism}" if op.mechanism else op.title,
            applies_to=["source_patch"],
            parameters={
                "patch_family": resolved_family or op.family_hint or op.category,
                "target_file": norm_target_file,
                "code_context": op.code_context,
                "reference_code": op.reference_code,
                "compiler_gap": op.compiler_gap,
                "diagnosis": op.diagnosis,
                "mechanism": op.mechanism,
                "assembly_evidence": op.assembly_evidence,
                "origin": "deep_code_analysis",
                "deep_analysis_id": op.opportunity_id,
                "implementation_complexity": op.implementation_complexity,
                "depends_on": list(op.depends_on),
                "conflicts_with": list(op.conflicts_with),
                "composable_with": list(op.composable_with),
                "target_files": norm_target_files,
            },
            expected_effect=effects,
            risk_level=op.risk_level or "medium",
            verification_plan=VerificationPlan(gates=gates),
        )
        actions.append(action)
    return actions


def _deep_analysis_next_step(
    status: OpportunityStatus,
    retry_count: int,
    max_context_retries: int = 1,
) -> str:
    if status == OpportunityStatus.OK:
        return "PROCEED"
    if status == OpportunityStatus.NEED_MORE_CONTEXT:
        if retry_count < max_context_retries:
            return "RETRY_CONTEXT"
        return "ERROR_NEED_MORE_CONTEXT"
    if status == OpportunityStatus.NEED_MORE_PROFILE:
        return "ERROR_NEED_MORE_PROFILE"
    return "ERROR_NO_ACTIONABLE"


def _collect_missing_context_snippets(
    repo_root: Path,
    missing: List[str],
    patch_root: str = "",
    max_items: int = 8,
    max_chars: int = 5000,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not missing:
        return out
    root = (repo_root / patch_root) if patch_root else repo_root
    for item in missing:
        token = str(item or "").strip()
        if not token:
            continue
        if len(out) >= max_items:
            break
        candidates: List[Path] = []
        token_path = Path(token)
        if token_path.is_absolute():
            candidates.append(token_path)
        else:
            candidates.append(root / token)
            if patch_root:
                candidates.append(repo_root / token)
        for path in candidates:
            if not path.exists() or not path.is_file():
                continue
            try:
                rel = str(path.relative_to(repo_root))
            except Exception:
                rel = str(path)
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            out[rel] = text[:max_chars]
            break
    return out


def _opportunity_graph_summary(graph: OpportunityGraph) -> Dict[str, object]:
    nodes = list(graph.opportunities or [])
    invalid = [node for node in nodes if getattr(node, "invalid", False)]
    valid = [node for node in nodes if not getattr(node, "invalid", False)]
    return {
        "graph_id": graph.graph_id,
        "total_nodes": len(nodes),
        "valid_nodes": len(valid),
        "invalid_nodes": len(invalid),
        "mechanisms": sorted(
            {
                str(getattr(getattr(node, "mechanism", None), "value", getattr(node, "mechanism", "")))
                for node in valid
                if getattr(node, "mechanism", None)
            }
        ),
    }


def _run_phase2_deep_analysis_init(
    *,
    use_two_phase: bool,
    phase: str,
    planner_cfg: Optional[Dict[str, object]],
    reporter: Optional[ConsoleUI],
    job: JobIR,
    best_chain_exp: ExperimentIR,
    baseline_exp: ExperimentIR,
    memory: OptimizationMemory,
    repo_root: Path,
    adapter_cfg: Optional[Dict[str, object]],
    build_cfg: Optional[Dict[str, object]],
    experience_memory: ExperienceMemory,
    system_caps: Dict[str, object],
    patch_families: Optional[Dict[str, object]],
    patch_planner: object,
    artifacts_dir: Path,
    trace_events: List[Dict[str, object]],
) -> DeepAnalysisInitOutput:
    deep_analysis_opportunities: List[ActionIR] = []
    deep_analysis_result: Optional[DeepCodeAnalysisResult] = None
    opportunity_graph: Optional[OpportunityGraph] = None
    selected_opportunities: Optional[SelectedOpportunities] = None
    if not (use_two_phase and phase == "PATCH"):
        return DeepAnalysisInitOutput(
            opportunities=deep_analysis_opportunities,
            deep_analysis_result=deep_analysis_result,
            opportunity_graph=opportunity_graph,
            selected_opportunities=selected_opportunities,
        )
    da_cfg = (planner_cfg or {}).get("two_phase", {})
    da_cfg = da_cfg.get("deep_analysis", {}) if isinstance(da_cfg, dict) else {}
    use_deep_analysis = isinstance(da_cfg, dict) and da_cfg.get("enabled", False)
    strict_da = bool(da_cfg.get("strict_required", False)) if isinstance(da_cfg, dict) else False
    if not use_deep_analysis:
        return DeepAnalysisInitOutput(
            opportunities=deep_analysis_opportunities,
            deep_analysis_result=deep_analysis_result,
            opportunity_graph=opportunity_graph,
            selected_opportunities=selected_opportunities,
        )

    from orchestrator.agents.code_analysis_agent import DeepCodeAnalysisAgent

    if reporter:
        reporter._section("Phase 2 Init: Deep Code Analysis")
    da_agent_config = AgentConfig(
        enabled=True,
        api_key_env=da_cfg.get("api_key_env", "DEEPSEEK_API_KEY"),
        base_url=da_cfg.get("base_url", "https://api.deepseek.com"),
        model=da_cfg.get("model", "deepseek-chat"),
        temperature=float(da_cfg.get("temperature", 0.3)),
        max_tokens=int(da_cfg.get("max_tokens", 4096)),
        max_turns=int(da_cfg.get("max_turns", 50)),
        max_tool_calls_per_turn=int(da_cfg.get("max_tool_calls_per_turn", 5)),
    )
    da_input_script = best_chain_exp.job.input_script or ""
    da_input_text = _safe_read(Path(da_input_script)) if da_input_script else ""
    da_patch_root = (adapter_cfg or {}).get("patch_root", "") if isinstance(adapter_cfg, dict) else ""
    da_tau = _merge_function_hotspots(
        repo_root=repo_root,
        patch_root=da_patch_root,
        baseline_exp=baseline_exp,
        best_exp=best_chain_exp,
        experiments=memory.experiments,
        max_entries=int(da_cfg.get("max_hotspots", 160) or 160),
    )
    da_hotspot = _hotspot_map(
        da_input_text,
        repo_root,
        best_chain_exp.job.run_args or [],
        patch_root=da_patch_root,
        function_hotspots=da_tau,
    )
    da_hotspot_files = da_hotspot.get("hotspot_files", [])
    if not da_hotspot_files and isinstance(adapter_cfg, dict):
        hg = adapter_cfg.get("hotspot_globs", {})
        if isinstance(hg, dict):
            seen_hg: set[str] = set()
            for category_globs in hg.values():
                for gp in (category_globs if isinstance(category_globs, list) else []):
                    import glob as _glob_mod

                    for match in sorted(_glob_mod.glob(str(repo_root / gp))):
                        rel = str(Path(match).relative_to(repo_root))
                        if rel not in seen_hg:
                            seen_hg.add(rel)
                            da_hotspot_files.append(rel)
            if da_hotspot_files:
                trace_events.append(
                    {
                        "event": "hotspot_globs_fallback",
                        "agent": "DeepCodeAnalysisAgent",
                        "num_files": len(da_hotspot_files),
                        "source": "adapter.hotspot_globs",
                    }
                )

    da_backend = _backend_from_args(best_chain_exp.job.run_args or [])
    da_exp_hints = (
        experience_memory.format_hints_for_prompt(app=job.app, backend=da_backend)
        if experience_memory.config.enabled
        else []
    )
    da_profile = (
        {
            "timing_breakdown": best_chain_exp.profile_report.timing_breakdown,
            "system_metrics": best_chain_exp.profile_report.system_metrics,
            "notes": best_chain_exp.profile_report.notes,
        }
        if best_chain_exp.profile_report
        else {}
    )
    if da_tau:
        da_profile["tau_hotspots"] = da_tau

    # Keep deep-analysis tools rooted at repository root so file paths like
    # "third_party/bwa/bwt.c" resolve consistently.
    da_repo_root = repo_root
    da_build_dir = repo_root
    if build_cfg and isinstance(build_cfg, dict):
        src_raw = str(build_cfg.get("source_dir") or ".")
        src_path = Path(src_raw)
        da_build_dir = src_path if src_path.is_absolute() else (repo_root / src_path)
    da_agent = DeepCodeAnalysisAgent(
        config=da_agent_config,
        repo_root=da_repo_root,
        build_dir=da_build_dir,
        experience_db=experience_memory,
    )

    context_retry = 0
    max_context_retry = int(da_cfg.get("max_context_retry", 1) or 1)
    supplemental_context: Dict[str, str] = {}
    graph_result: Optional[OpportunityGraphResult] = None
    last_discovery: Optional[object] = None
    while True:
        try:
            graph_result = da_agent.discover_opportunity_graph(
                profile=da_profile,
                hotspot_files=da_hotspot_files,
                system_caps=system_caps,
                patch_families=patch_families,
                experience_hints=da_exp_hints,
                backend_variant=da_backend,
                input_script_path=job.input_script or None,
                supplemental_context=supplemental_context,
            )
            last_discovery = da_agent.last_discovery_run
        except LLMUnavailableError:
            raise
        except Exception as exc:
            graph_result = OpportunityGraphResult(
                status=OpportunityStatus.NO_ACTIONABLE,
                graph=None,
                missing=[],
                needs_profile=[],
                rationale=str(exc),
                suggestions=["fix deep analysis call failure"],
            )
            last_discovery = None
            trace_events.append(
                {
                    "event": "deep_analysis_error",
                    "agent": "DeepCodeAnalysisAgent",
                    "error": str(exc),
                }
            )

        trace_events.append(
            {
                "event": "deep_analysis_discovery",
                "agent": "DeepCodeAnalysisAgent",
                "status": graph_result.status.value if graph_result else "NO_ACTIONABLE",
                "retry_count": context_retry,
                "missing": list(graph_result.missing if graph_result else []),
                "needs_profile": list(graph_result.needs_profile if graph_result else []),
                "rationale": graph_result.rationale if graph_result else "",
            }
        )
        if reporter and last_discovery:
            reporter.agent_conversation("DeepCodeAnalysisAgent", last_discovery.conversation_log)
        if last_discovery:
            conv_path = artifacts_dir / "deep_analysis_conversation.json"
            try:
                conv_path.write_text(
                    json.dumps(last_discovery.conversation_log, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
            deep_analysis_result = (
                last_discovery.analysis if last_discovery.analysis else None
            )

        if not graph_result:
            break
        next_step = _deep_analysis_next_step(
            graph_result.status,
            retry_count=context_retry,
            max_context_retries=max_context_retry,
        )
        if next_step == "RETRY_CONTEXT":
            supplemental_context = _collect_missing_context_snippets(
                repo_root=repo_root,
                missing=list(graph_result.missing or []),
                patch_root=da_patch_root,
                max_items=8,
                max_chars=4000,
            )
            context_retry += 1
            trace_events.append(
                {
                    "event": "deep_analysis_retry_context",
                    "agent": "Orchestrator",
                    "retry_count": context_retry,
                    "collected_files": sorted(list(supplemental_context.keys())),
                    "missing": list(graph_result.missing or []),
                }
            )
            if supplemental_context:
                continue
        break

    if graph_result and graph_result.status == OpportunityStatus.OK and graph_result.graph:
        opportunity_graph = graph_result.graph
        select_k = int(da_cfg.get("top_k", (planner_cfg or {}).get("max_candidates", 3)) or 3)
        selected_opportunities = da_agent.select_topk_from_graph(
            graph=opportunity_graph,
            k=select_k,
            budget={
                "max_iters": job.budgets.max_iters,
                "max_runs": job.budgets.max_runs,
            },
            experience_hints=da_exp_hints,
            selection_policy=(
                da_cfg.get("selection_policy", {})
                if isinstance(da_cfg.get("selection_policy"), dict)
                else None
            ),
        )
        deep_analysis_opportunities = patch_planner.plan_from_opportunity_selection(
            selected_opportunities,
            patch_families=patch_families,
            existing_action_ids=[],
        )
        if strict_da and not deep_analysis_opportunities:
            raise RuntimeError(
                "OpportunityGraph selection produced zero actionable source_patch actions"
            )
        trace_events.append(
            {
                "event": "opportunity_graph",
                "agent": "DeepCodeAnalysisAgent",
                "summary": _opportunity_graph_summary(opportunity_graph),
                "graph": opportunity_graph.model_dump(),
                "selection": selected_opportunities.model_dump(),
            }
        )
        graph_path = artifacts_dir / "opportunity_graph.json"
        select_path = artifacts_dir / "opportunity_selection.json"
        try:
            graph_path.write_text(
                opportunity_graph.model_dump_json(indent=2),
                encoding="utf-8",
            )
            select_path.write_text(
                selected_opportunities.model_dump_json(indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        if reporter:
            reporter._print(
                f"Deep analysis graph: {len(opportunity_graph.opportunities)} nodes, "
                f"selected {len(selected_opportunities.selected)} actions"
            )
    else:
        status = graph_result.status if graph_result else OpportunityStatus.NO_ACTIONABLE
        rationale = graph_result.rationale if graph_result else "unknown"
        missing = list(graph_result.missing if graph_result else [])
        needs_profile = list(graph_result.needs_profile if graph_result else [])
        if reporter:
            reporter._print(
                f"Deep analysis failed: status={status.value} rationale={rationale}"
            )
        if strict_da:
            if status == OpportunityStatus.NEED_MORE_CONTEXT:
                raise RuntimeError(
                    "Deep analysis NEED_MORE_CONTEXT after one retry; "
                    f"missing={missing}; rationale={rationale}"
                )
            if status == OpportunityStatus.NEED_MORE_PROFILE:
                raise RuntimeError(
                    "Deep analysis NEED_MORE_PROFILE; "
                    f"required_profile={needs_profile}; rationale={rationale}"
                )
            raise RuntimeError(
                "Deep analysis NO_ACTIONABLE; "
                f"rationale={rationale}; suggestions={graph_result.suggestions if graph_result else []}"
            )

    return DeepAnalysisInitOutput(
        opportunities=deep_analysis_opportunities,
        deep_analysis_result=deep_analysis_result,
        opportunity_graph=opportunity_graph,
        selected_opportunities=selected_opportunities,
    )


def _select_opportunity_batch(
    opportunities: List[ActionIR],
    min_batch: int = 3,
    max_batch: int = 5,
    succeeded_ids: Optional[set] = None,
    failed_ids: Optional[set] = None,
) -> List[ActionIR]:
    """Select a batch of compatible opportunities for one iteration.

    Algorithm:
    1. Filter out opportunities whose dependencies are unmet or that have failed.
    2. Build conflict pairs from ``conflicts_with``.
    3. Greedily pick highest-priority opportunities that don't conflict with
       those already selected, preferring composable partners.

    The function does NOT check for file-level overlap (``_actions_conflict``)
    because within one iteration actions compete independently (all built on
    ``best_chain_exp``).  File-level conflicts are resolved later by
    ``_select_non_conflicting``.
    """
    succeeded = succeeded_ids or set()
    failed = failed_ids or set()

    # --- 1. Filter ---
    eligible: List[ActionIR] = []
    for opp in opportunities:
        params = opp.parameters or {}
        da_id = params.get("deep_analysis_id", "")
        if da_id in failed:
            continue
        deps = params.get("depends_on", [])
        if deps and not all(d in succeeded for d in deps):
            continue
        eligible.append(opp)

    if not eligible:
        return []

    # --- 2. Conflict pairs ---
    conflict_map: Dict[str, set] = {}
    for opp in eligible:
        params = opp.parameters or {}
        da_id = params.get("deep_analysis_id", "")
        conflicts = set(params.get("conflicts_with", []))
        conflict_map[da_id] = conflicts

    # --- 3. Composability affinity ---
    composable_map: Dict[str, set] = {}
    for opp in eligible:
        params = opp.parameters or {}
        da_id = params.get("deep_analysis_id", "")
        composable_map[da_id] = set(params.get("composable_with", []))

    # --- 4. Greedy selection ---
    selected: List[ActionIR] = []
    selected_ids: set = set()

    for opp in eligible:
        if len(selected) >= max_batch:
            break
        da_id = (opp.parameters or {}).get("deep_analysis_id", "")
        # Check conflict with already-selected
        dominated = False
        for sid in selected_ids:
            if da_id in conflict_map.get(sid, set()) or sid in conflict_map.get(da_id, set()):
                dominated = True
                break
        if dominated:
            continue
        selected.append(opp)
        selected_ids.add(da_id)

    # If below min_batch but more eligible exist, don't force — return what we have.
    return selected


def _seed_graph_actions_for_iteration(
    deep_analysis_opportunities: List[ActionIR],
    tested_actions: List[str],
    use_batch_selection: bool,
    batch_min: int,
    batch_max: int,
    succeeded_ids: set[str],
    failed_ids: set[str],
    iteration: int,
    trace_events: List[Dict[str, object]],
    blocked_action_ids: Optional[set[str]] = None,
) -> List[ActionIR]:
    blocked = blocked_action_ids or set()
    pending_graph_actions = [
        action
        for action in deep_analysis_opportunities
        if action.action_id not in tested_actions
        and action.action_id not in blocked
        and ((action.parameters or {}).get("deep_analysis_id") not in failed_ids)
    ]
    generated: List[ActionIR] = []
    if use_batch_selection and len(pending_graph_actions) > 1:
        batch = _select_opportunity_batch(
            pending_graph_actions,
            min_batch=batch_min,
            max_batch=batch_max,
            succeeded_ids=succeeded_ids,
            failed_ids=failed_ids,
        )
        if batch:
            generated.extend(batch)
            trace_events.append(
                {
                    "event": "deep_analysis_batch",
                    "agent": "DeepCodeAnalysisAgent",
                    "iteration": iteration,
                    "batch_size": len(batch),
                    "action_ids": [a.action_id for a in batch],
                    "deep_analysis_ids": [
                        (a.parameters or {}).get("deep_analysis_id") for a in batch
                    ],
                    "evidence_ids": {
                        a.action_id: list((a.parameters or {}).get("evidence_ids", []))
                        for a in batch
                    },
                }
            )
        return generated
    gen_action = pending_graph_actions[0] if pending_graph_actions else None
    if gen_action:
        generated.append(gen_action)
        trace_events.append(
            {
                "event": "deep_analysis_action",
                "agent": "DeepCodeAnalysisAgent",
                "iteration": iteration,
                "action_id": gen_action.action_id,
                "deep_analysis_id": gen_action.parameters.get("deep_analysis_id"),
                "evidence_ids": list((gen_action.parameters or {}).get("evidence_ids", [])),
            }
        )
    return generated


def _maybe_stop_on_opportunity_graph_exhausted(
    phase: str,
    opportunity_graph_mode: bool,
    generated_actions: List[ActionIR],
    deep_analysis_opportunities: List[ActionIR],
    tested_actions: List[str],
    iteration: int,
    trace_events: List[Dict[str, object]],
    reporter: Optional[ConsoleUI],
    failed_ids: Optional[set[str]] = None,
    blocked_action_ids: Optional[set[str]] = None,
) -> bool:
    if not (phase == "PATCH" and opportunity_graph_mode and not generated_actions):
        return False
    failed = failed_ids or set()
    blocked = blocked_action_ids or set()
    remaining_graph_actions = [
        action
        for action in deep_analysis_opportunities
        if action.action_id not in tested_actions
        and action.action_id not in blocked
        and ((action.parameters or {}).get("deep_analysis_id") not in failed)
    ]
    if remaining_graph_actions:
        return False
    trace_events.append(
        {
            "event": "opportunity_graph_exhausted",
            "agent": "Orchestrator",
            "iteration": iteration,
            "reason": "no untested graph-derived source_patch actions remaining",
        }
    )
    if reporter:
        reporter.stop("source_patch 阶段完成: OpportunityGraph 已耗尽")
    return True


def _build_iteration_base_actions(
    actions: List[ActionIR],
    generated_actions: List[ActionIR],
    phase: str,
    opportunity_graph_mode: bool,
) -> List[ActionIR]:
    base_actions_for_iteration = list(actions) + generated_actions
    if phase == "PATCH" and opportunity_graph_mode:
        return [
            action for action in base_actions_for_iteration if action.family != "source_patch"
        ] + [action for action in generated_actions if action.family == "source_patch"]
    return base_actions_for_iteration


def _sanitize_param_explorer_env(candidates: List[ActionIR]) -> None:
    """Remove thread/binding env overrides from non-thread actions.

    This ensures that once a good thread count is found, subsequent
    non-thread actions inherit it from the base job instead of resetting
    to OMP_NUM_THREADS=1 in their own env overrides.
    """
    thread_env_keys = {
        "OMP_NUM_THREADS",
        "OMP_PROC_BIND",
        "OMP_PLACES",
        "OMP_DYNAMIC",
    }
    thread_families = {
        "parallel_omp",
        "affinity_tune",
        "parallel_mpi",
        "mpi_omp_hybrid",
        "runtime_backend_select",
    }
    for cand in candidates:
        if not cand.parameters or "env" not in cand.parameters:
            continue
        if cand.family in thread_families:
            continue
        env = dict(cand.parameters.get("env") or {})
        if not env:
            continue
        changed = False
        for key in thread_env_keys:
            if key in env:
                env.pop(key, None)
                changed = True
        if changed:
            if env:
                cand.parameters["env"] = env
            else:
                cand.parameters.pop("env", None)


def _run_parameter_exploration_phase(
    job: JobIR,
    actions: List[ActionIR],
    baseline_exp: ExperimentIR,
    executor: "ExecutorAgent",
    profiler: "ProfilerAgent",
    verifier: "VerifierAgent",
    policy: Dict[str, object],
    gates: Dict[str, object],
    artifacts_dir: Path,
    time_command: Optional[str],
    profiling_cfg: Optional[Dict[str, object]],
    wrappers_cfg: Optional[List[Dict[str, object]]],
    build_cfg: Optional[Dict[str, object]],
    build_packs: Optional[Dict[str, object]],
    adapter_cfg: Optional[Dict[str, object]],
    planner_cfg: Optional[Dict[str, object]],
    experience_memory: "ExperienceMemory",
    memory: "OptimizationMemory",
    state: StopState,
    trace_events: List[Dict[str, object]],
    reporter: Optional[ConsoleUI],
    repo_root: Path,
    system_caps: Dict[str, object],
    chain_min_improvement_pct: float,
    arg_rules: Optional[List[Dict[str, object]]] = None,
) -> Tuple[Optional[ExperimentIR], List[ExperimentIR]]:
    """Phase 1: agentic parameter exploration + single-batch evaluation.

    Returns:
        (best_param_exp, all_experiments) — best_param_exp is None if exploration
        failed or produced no improvements.
    """
    two_phase_cfg = (planner_cfg or {}).get("two_phase", {})
    explorer_cfg = two_phase_cfg.get("parameter_explorer", {}) if isinstance(two_phase_cfg, dict) else {}

    if reporter:
        reporter._section("Phase 1: Agentic Parameter Exploration")

    # Build AgentConfig from explorer settings
    _explorer_max_turns = int(explorer_cfg.get("max_turns", 20))
    _explorer_max_tool_calls = int(explorer_cfg.get("max_tool_calls_per_turn", 5))

    agent_config = AgentConfig(
        enabled=True,
        api_key_env=explorer_cfg.get("api_key_env", "DEEPSEEK_API_KEY"),
        base_url=explorer_cfg.get("base_url", "https://api.deepseek.com"),
        model=explorer_cfg.get("model", "deepseek-chat"),
        temperature=float(explorer_cfg.get("temperature", 0.3)),
        max_tokens=4096,
        max_turns=_explorer_max_turns,
        max_tool_calls_per_turn=_explorer_max_tool_calls,
    )

    input_script_path = Path(job.input_script) if job.input_script else None

    agent = ParameterExplorerAgent(
        config=agent_config,
        repo_root=_resolve_app_repo_root(repo_root, job, adapter_cfg),
        input_script_path=input_script_path,
        experience_db=experience_memory,
    )

    # Build profile dict
    profile_data = {}
    if baseline_exp.profile_report:
        profile_data = build_profile_payload(baseline_exp.profile_report)

    # Reconstruct action_space dict for the agent
    action_space = {
        "families": [],
        "actions": [],
    }
    seen_families = set()
    for a in actions:
        if a.family not in seen_families:
            seen_families.add(a.family)
            action_space["families"].append({
                "id": a.family,
                "description": "",
                "expected_effect": list(a.expected_effect) if a.expected_effect else [],
            })
        action_space["actions"].append(a.model_dump())

    job_context = {
        "app": job.app,
        "case_id": job.case_id,
        "tags": job.tags or [],
        "system_caps": system_caps,
        "run_args": job.run_args,
        "env": job.env,
    }

    trace_events.append({
        "event": "phase1_start",
        "agent": "ParameterExplorerAgent",
    })

    # Run exploration
    result = agent.explore(
        profile=profile_data,
        action_space=action_space,
        job_context=job_context,
    )

    trace_events.append({
        "event": "phase1_exploration_done",
        "agent": "ParameterExplorerAgent",
        "status": result.status,
        "num_candidates": len(result.candidates),
        "total_turns": result.total_turns,
        "total_tokens": result.total_tokens,
        "platform_summary": result.platform_summary,
        "rationale": result.rationale[:500] if result.rationale else "",
    })
    if reporter:
        reporter.agent_conversation("ParameterExplorerAgent", result.conversation_log)

    if result.status != "OK" or not result.candidates:
        if reporter:
            reporter._print(
                f"Phase 1 exploration: {result.status}, "
                f"{len(result.candidates)} candidates — falling back to iterative mode"
            )
        return None, []

    if reporter:
        reporter._print(
            f"Phase 1: {len(result.candidates)} parameter candidates proposed "
            f"({result.total_turns} turns, {result.total_tokens} tokens)"
        )

    # Auto-inject backend_enable for parallel_omp actions so the LAMMPS
    # adapter adds ``-sf omp`` to run_args.  Without this, setting
    # OMP_NUM_THREADS alone has no effect because LAMMPS pair styles
    # default to serial kernels.  BWA does not use this mechanism.
    if job.app == "lammps":
        _PARALLEL_FAMILIES = {"parallel_omp", "affinity_tune"}
        for cand in result.candidates:
            needs_backend = (
                _is_parallel_threads_action(cand)
                or (cand.family and cand.family in _PARALLEL_FAMILIES)
            )
            if needs_backend:
                params = dict(cand.parameters or {})
                if not params.get("backend_enable"):
                    params["backend_enable"] = "openmp"
                    cand.parameters = params

    _sanitize_param_explorer_env(result.candidates)

    # Ensure app-specific adapter logic runs for agent-proposed actions.
    result.candidates = _apply_adapter(result.candidates, adapter_cfg, job)

    # Filter out MPI launcher actions when the binary lacks real MPI support.
    result.candidates = _filter_mpi_actions(result.candidates, job, reporter)

    # Execute all candidates in a single batch
    all_experiments: List[ExperimentIR] = []
    baseline_runtime = baseline_exp.results.runtime_seconds
    current_base_runtime = baseline_runtime
    base_run_id = baseline_exp.run_id
    base_action_id = baseline_exp.action.action_id if baseline_exp.action else None
    base_job = baseline_exp.job

    for i, candidate in enumerate(result.candidates):
        exp_id = f"phase1-{candidate.action_id}"
        if reporter:
            reporter._print(f"  [{i + 1}/{len(result.candidates)}] {candidate.action_id}")

        try:
            exp = executor.execute(
                exp_id=exp_id,
                job=job,
                base_job=base_job,
                base_run_id=base_run_id,
                base_action_id=base_action_id,
                action=candidate,
                actions_root=repo_root,
                policy=policy,
                gates=gates,
                profiler=profiler,
                verifier=verifier,
                artifacts_dir=artifacts_dir,
                time_command=time_command,
                profiling_cfg=profiling_cfg,
                wrappers_cfg=wrappers_cfg,
                build_cfg=build_cfg or {},
                build_packs=build_packs,
                adapter_cfg=adapter_cfg,
                repeats=1,
                runtime_agg="mean",
                baseline_exp=baseline_exp,
                baseline_exp_for_verify=baseline_exp,
                baseline_runtime=baseline_runtime,
                prior_samples=None,
                trace_events=trace_events,
                parent_run_id=None,
                iteration=0,
                llm_trace=None,
                reporter=reporter,
                arg_rules=arg_rules,
            )
            all_experiments.append(exp)
            memory.record(exp)
            state.run_count += 1

            experience_memory.record_experiment(exp, baseline_exp)
            if exp.verdict == "PASS" and exp.results and exp.results.runtime_seconds > 0:
                improvement = (
                    (current_base_runtime - exp.results.runtime_seconds) / current_base_runtime
                    if current_base_runtime > 0
                    else 0.0
                )
                if improvement >= chain_min_improvement_pct:
                    current_base_runtime = exp.results.runtime_seconds
                    base_run_id = exp.run_id
                    base_action_id = exp.action.action_id if exp.action else "baseline"
                    base_job = exp.job
                    trace_events.append(
                        {
                            "event": "phase1_base_update",
                            "agent": "Orchestrator",
                            "run_id": exp.run_id,
                            "action_id": base_action_id,
                            "improvement": improvement,
                        }
                    )

        except LLMUnavailableError:
            raise
        except Exception as exc:
            trace_events.append({
                "event": "phase1_candidate_error",
                "agent": "ExecutorAgent",
                "action_id": candidate.action_id,
                "error": str(exc),
            })
            continue

    # Select best non-conflicting combination
    winners = _select_non_conflicting(
        all_experiments,
        baseline_runtime,
        chain_min_improvement_pct,
        baseline_run_id=baseline_exp.run_id,
    )

    if not winners:
        if reporter:
            reporter._print("Phase 1: No candidates improved over baseline")
        trace_events.append({
            "event": "phase1_no_winners",
            "agent": "Orchestrator",
        })
        return None, all_experiments

    best_exp = winners[0]

    if len(winners) > 1:
        # Build composite action and validate
        composite_action = _build_final_composite_action(winners, gates)
        if composite_action:
            comp_exp_id = "phase1-composite"
            try:
                comp_exp = executor.execute(
                    exp_id=comp_exp_id,
                    job=job,
                    base_job=base_job,
                    base_run_id=base_run_id,
                    base_action_id=base_action_id,
                    action=composite_action,
                    actions_root=repo_root,
                    policy=policy,
                    gates=gates,
                    profiler=profiler,
                    verifier=verifier,
                    artifacts_dir=artifacts_dir,
                    time_command=time_command,
                    profiling_cfg=profiling_cfg,
                    wrappers_cfg=wrappers_cfg,
                    build_cfg=build_cfg or {},
                    build_packs=build_packs,
                    adapter_cfg=adapter_cfg,
                    repeats=1,
                    runtime_agg="mean",
                    baseline_exp=baseline_exp,
                    baseline_exp_for_verify=baseline_exp,
                    baseline_runtime=baseline_runtime,
                    prior_samples=None,
                    trace_events=trace_events,
                    parent_run_id=None,
                    iteration=0,
                    llm_trace=None,
                    reporter=reporter,
                    arg_rules=arg_rules,
                )
                all_experiments.append(comp_exp)
                memory.record(comp_exp)
                state.run_count += 1

                if (
                    comp_exp.verdict == "PASS"
                    and comp_exp.results.runtime_seconds < best_exp.results.runtime_seconds
                ):
                    best_exp = comp_exp
                    if reporter:
                        reporter._print("Phase 1: Composite outperforms single best")
            except LLMUnavailableError:
                raise
            except Exception as exc:
                trace_events.append({
                    "event": "phase1_composite_error",
                    "agent": "ExecutorAgent",
                    "error": str(exc),
                })

    speedup = baseline_runtime / best_exp.results.runtime_seconds if best_exp.results.runtime_seconds > 0 else 1.0
    if reporter:
        reporter._print(
            f"Phase 1 result: {best_exp.run_id} — "
            f"{best_exp.results.runtime_seconds:.3f}s "
            f"({speedup:.2f}x vs baseline)"
        )

    trace_events.append({
        "event": "phase1_done",
        "agent": "Orchestrator",
        "best_run_id": best_exp.run_id,
        "best_runtime": best_exp.results.runtime_seconds,
        "speedup": speedup,
        "total_candidates": len(result.candidates),
        "total_experiments": len(all_experiments),
        "winners": len(winners),
    })

    return best_exp, all_experiments


def _phase1_cache_enabled(planner_cfg: Optional[Dict[str, object]]) -> bool:
    two_phase_cfg = (planner_cfg or {}).get("two_phase", {})
    cache_cfg = two_phase_cfg.get("phase1_cache", {}) if isinstance(two_phase_cfg, dict) else {}
    if not isinstance(cache_cfg, dict):
        return True
    return bool(cache_cfg.get("enabled", True))


def _phase1_cache_path(
    artifacts_dir: Path,
    planner_cfg: Optional[Dict[str, object]],
) -> Path:
    two_phase_cfg = (planner_cfg or {}).get("two_phase", {})
    cache_cfg = two_phase_cfg.get("phase1_cache", {}) if isinstance(two_phase_cfg, dict) else {}
    path_raw = cache_cfg.get("path") if isinstance(cache_cfg, dict) else None
    if isinstance(path_raw, str) and path_raw:
        path = Path(path_raw)
        if not path.is_absolute():
            path = artifacts_dir.parent / path
        return path
    return artifacts_dir.parent / "knowledge" / "phase1_cache.json"


def _phase1_cache_key(job: JobIR) -> str:
    return f"{job.app}:{job.case_id}"


def _load_phase1_cache(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"version": 1, "entries": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "entries": {}}
    if not isinstance(payload, dict):
        return {"version": 1, "entries": {}}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        payload["entries"] = {}
    payload.setdefault("version", 1)
    return payload


def _save_phase1_cache(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _phase1_improvement_pct(baseline_runtime: float, tuned_runtime: float) -> float:
    if baseline_runtime <= 0.0 or tuned_runtime <= 0.0:
        return 0.0
    return (baseline_runtime - tuned_runtime) / baseline_runtime * 100.0


def _lookup_phase1_cached_action(
    actions: List[ActionIR],
    job: JobIR,
    cache_payload: Dict[str, object],
) -> Tuple[Optional[ActionIR], Optional[Dict[str, object]]]:
    entries = cache_payload.get("entries", {})
    if not isinstance(entries, dict):
        return None, None
    entry = entries.get(_phase1_cache_key(job))
    if not isinstance(entry, dict):
        return None, None
    action_blob = entry.get("best_action")
    if not isinstance(action_blob, dict):
        return None, entry
    cached_action_id = action_blob.get("action_id")
    if isinstance(cached_action_id, str) and cached_action_id:
        for action in actions:
            if action.action_id == cached_action_id:
                return action.model_copy(deep=True), entry
    try:
        return ActionIR.model_validate(action_blob), entry
    except Exception:
        return None, entry


def _record_phase1_cache_entry(
    cache_path: Path,
    cache_payload: Dict[str, object],
    job: JobIR,
    baseline_exp: ExperimentIR,
    best_param_exp: ExperimentIR,
) -> None:
    if not best_param_exp.action:
        return
    applies = set(best_param_exp.action.applies_to or [])
    if not ({"run_config", "input_script", "build_config"} & applies):
        return
    baseline_runtime = baseline_exp.results.runtime_seconds
    tuned_runtime = best_param_exp.results.runtime_seconds
    improvement_pct = _phase1_improvement_pct(baseline_runtime, tuned_runtime)
    speedup = baseline_runtime / tuned_runtime if baseline_runtime > 0 and tuned_runtime > 0 else 1.0
    entries = cache_payload.setdefault("entries", {})
    if not isinstance(entries, dict):
        cache_payload["entries"] = {}
        entries = cache_payload["entries"]
    entries[_phase1_cache_key(job)] = {
        "app": job.app,
        "case_id": job.case_id,
        "best_action": best_param_exp.action.model_dump(),
        "best_run_id": best_param_exp.run_id,
        "baseline_runtime_s": baseline_runtime,
        "best_runtime_s": tuned_runtime,
        "improvement_pct": improvement_pct,
        "speedup": speedup,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _save_phase1_cache(cache_path, cache_payload)


def _bootstrap_phase1_cache_from_history(
    artifacts_dir: Path,
    actions: List[ActionIR],
    job: JobIR,
    cache_path: Path,
    cache_payload: Dict[str, object],
) -> Tuple[Optional[ActionIR], Optional[Dict[str, object]]]:
    sessions_root = artifacts_dir.parent
    if not sessions_root.exists():
        return None, None
    best_entry: Optional[Dict[str, object]] = None
    for session in sorted(sessions_root.iterdir(), reverse=True):
        if not session.is_dir():
            continue
        if session.name == "knowledge":
            continue
        best_state = session / "best_state.json"
        if not best_state.exists():
            continue
        try:
            payload = json.loads(best_state.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("case_id") != job.case_id:
            continue
        action_blob = payload.get("best_run_action") or payload.get("best_action")
        if not isinstance(action_blob, dict):
            continue
        try:
            action = ActionIR.model_validate(action_blob)
        except Exception:
            continue
        applies = set(action.applies_to or [])
        if not ({"run_config", "input_script", "build_config"} & applies):
            continue
        best_entry = {
            "app": job.app,
            "case_id": job.case_id,
            "best_action": action.model_dump(),
            "best_run_id": payload.get("best_run_id"),
            "baseline_runtime_s": None,
            "best_runtime_s": None,
            "improvement_pct": None,
            "speedup": None,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_session": session.name,
            "source": "best_state_bootstrap",
        }
        break
    if best_entry is None:
        return None, None
    entries = cache_payload.setdefault("entries", {})
    if not isinstance(entries, dict):
        cache_payload["entries"] = {}
        entries = cache_payload["entries"]
    entries[_phase1_cache_key(job)] = best_entry
    _save_phase1_cache(cache_path, cache_payload)
    return _lookup_phase1_cached_action(actions, job, cache_payload)


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


def _build_cost_model(experiments: List[ExperimentIR]) -> Dict[str, float]:
    run_samples = [
        exp.results.runtime_seconds
        for exp in experiments
        if exp.results and exp.results.runtime_seconds > 0.0
    ]
    build_samples = [exp.build_seconds for exp in experiments if exp.build_seconds]
    avg_run = statistics.mean(run_samples) if run_samples else 0.0
    avg_build = statistics.mean(build_samples) if build_samples else 0.0
    return {"avg_run_seconds": avg_run, "avg_build_seconds": avg_build}


def _collect_iteration_summaries(
    experiments: List[ExperimentIR],
    baseline_runtime: float,
) -> List[Dict[str, object]]:
    import re

    summaries: Dict[int, Dict[str, object]] = {}
    for exp in experiments:
        if exp.action is None:
            continue
        exp_id = exp.exp_id or ""
        if exp_id.endswith("-validate"):
            continue
        match = re.match(r"^iter(\d+)-", exp_id)
        if not match:
            continue
        iteration = int(match.group(1))
        summary = summaries.setdefault(
            iteration,
            {
                "iteration": iteration,
                "attempts": 0,
                "fails": 0,
                "best_runtime": None,
                "best_action": None,
            },
        )
        summary["attempts"] += 1
        if exp.verdict == "FAIL":
            summary["fails"] += 1
            continue
        runtime = exp.results.runtime_seconds
        if summary["best_runtime"] is None or runtime < summary["best_runtime"]:
            summary["best_runtime"] = runtime
            summary["best_action"] = exp.action.action_id
    results: List[Dict[str, object]] = []
    for iteration in sorted(summaries.keys()):
        summary = summaries[iteration]
        best_runtime = summary.get("best_runtime")
        speedup = None
        improvement_pct = None
        if best_runtime and best_runtime > 0 and baseline_runtime > 0:
            speedup = baseline_runtime / best_runtime
            improvement_pct = (baseline_runtime - best_runtime) / baseline_runtime
        results.append(
            {
                **summary,
                "speedup": speedup,
                "improvement_pct": improvement_pct,
            }
        )
    return results


def _remaining_candidates_by_family(
    actions: List[ActionIR],
    tested: set[str],
) -> Dict[str, int]:
    remaining: Dict[str, int] = {}
    for action in actions:
        if action.action_id in tested:
            continue
        remaining[action.family] = remaining.get(action.family, 0) + 1
    return remaining


def _best_summary(
    baseline: ExperimentIR,
    best: Optional[ExperimentIR],
) -> Dict[str, object]:
    base = baseline.results.runtime_seconds
    if not best:
        return {"baseline_runtime": base}
    best_runtime = best.results.runtime_seconds
    speedup = base / best_runtime if base > 0 and best_runtime > 0 else None
    improvement_pct = (base - best_runtime) / base if base > 0 and best_runtime > 0 else None
    return {
        "baseline_runtime": base,
        "best_runtime": best_runtime,
        "best_action": best.action.action_id if best.action else "baseline",
        "speedup": speedup,
        "improvement_pct": improvement_pct,
    }


def _extract_phase_transitions(trace_events: List[Dict[str, object]]) -> List[Dict[str, object]]:
    transitions: List[Dict[str, object]] = []
    for event in trace_events:
        if event.get("event") != "phase_transition":
            continue
        transitions.append(
            {
                "iteration": event.get("iteration"),
                "from_phase": event.get("from_phase"),
                "to_phase": event.get("to_phase"),
                "frozen_run_id": event.get("frozen_run_id"),
                "frozen_build_id": event.get("frozen_build_id"),
                "reason": event.get("reason"),
            }
        )
    return transitions


def _phase_targets(phase: str) -> set[str]:
    if phase == "RUN_TUNE":
        return {"run_config", "input_script"}
    if phase == "BUILD_TUNE":
        return {"build_config"}
    if phase == "RUN_RETUNE":
        return {"run_config", "input_script"}
    if phase == "PATCH":
        return {"source_patch"}
    return {"run_config"}


def _coverage_floor_families(
    phase: str,
    eligible_pool: List[ActionIR],
    remaining_by_family: Dict[str, int],
) -> List[str]:
    families_by_target: Dict[str, set[str]] = {
        "run_config": set(),
        "input_script": set(),
        "build_config": set(),
        "source_patch": set(),
    }
    for action in eligible_pool:
        if remaining_by_family.get(action.family, 0) <= 0:
            continue
        for target in action.applies_to or []:
            if target in families_by_target:
                families_by_target[target].add(action.family)
    if phase in {"RUN_TUNE", "RUN_RETUNE"}:
        return sorted(families_by_target["run_config"] | families_by_target["input_script"])
    if phase == "BUILD_TUNE":
        return sorted(families_by_target["build_config"])
    if phase == "PATCH":
        return sorted(families_by_target["source_patch"])
    return []


def _families_by_target(
    eligible_pool: List[ActionIR],
    remaining_by_family: Dict[str, int],
) -> Dict[str, List[str]]:
    families: Dict[str, set[str]] = {
        "run_config": set(),
        "input_script": set(),
        "build_config": set(),
        "source_patch": set(),
    }
    for action in eligible_pool:
        if remaining_by_family.get(action.family, 0) <= 0:
            continue
        for target in action.applies_to or []:
            if target in families:
                families[target].add(action.family)
    return {key: sorted(vals) for key, vals in families.items()}


def _inject_family(
    chosen: List[str],
    candidates: List[str],
    prefer: List[str],
    max_families: int = 2,
) -> List[str]:
    if not candidates:
        return chosen
    if any(fam in chosen for fam in candidates):
        return chosen
    pick = None
    for fam in prefer:
        if fam in candidates:
            pick = fam
            break
    if pick is None:
        pick = candidates[0]
    if pick in chosen:
        return chosen
    if len(chosen) >= max_families:
        chosen = chosen[:-1] + [pick]
    else:
        chosen = chosen + [pick]
    # keep unique order
    deduped: List[str] = []
    for fam in chosen:
        if fam not in deduped:
            deduped.append(fam)
    return deduped


def _dedupe_actions(actions: List[ActionIR]) -> List[ActionIR]:
    deduped: List[ActionIR] = []
    seen: set[str] = set()
    for action in actions:
        if action.action_id in seen:
            continue
        seen.add(action.action_id)
        deduped.append(action)
    return deduped


def _best_non_patch_experiment(experiments: List[ExperimentIR]) -> Optional[ExperimentIR]:
    best: Optional[ExperimentIR] = None
    for exp in experiments:
        if exp.verdict != "PASS":
            continue
        if exp.action and exp.action.family == "source_patch":
            continue
        if exp.results is None:
            continue
        if best is None or exp.results.runtime_seconds < best.results.runtime_seconds:
            best = exp
    return best


def _load_patch_replay_actions(
    artifacts_root: Path,
    case_id: str,
    app_name: Optional[str],
    backend: Optional[str],
    min_gain_pct: float,
    max_actions: int,
    existing_action_ids: set[str],
    patch_families_cfg: Dict[str, object],
) -> List[ActionIR]:
    sessions_root = artifacts_root / "sessions"
    if not sessions_root.exists() or max_actions <= 0:
        return []
    families_meta: Dict[str, Dict[str, object]] = {}
    for item in (patch_families_cfg or {}).get("families", []):
        if isinstance(item, dict) and item.get("id"):
            families_meta[item["id"]] = item
    exact_case_candidates: List[Tuple[float, ExperimentIR]] = []
    same_app_candidates: List[Tuple[float, ExperimentIR]] = []
    for session in sessions_root.iterdir():
        runs_dir = session / "runs"
        if not runs_dir.is_dir():
            continue
        for run_dir in runs_dir.iterdir():
            exp_path = run_dir / "experiment.json"
            if not exp_path.exists():
                continue
            try:
                exp = ExperimentIR.model_validate_json(exp_path.read_text())
            except Exception:
                continue
            if exp.action is None or exp.action.family != "source_patch":
                continue
            params = exp.action.parameters or {}
            if str(exp.action.action_id).startswith("replay.") or params.get("origin") == "memory_replay":
                continue
            if exp.verdict != "PASS" or exp.results is None:
                continue
            if exp.job.case_id == case_id:
                bucket = exact_case_candidates
            elif app_name and exp.job.app == app_name:
                bucket = same_app_candidates
            else:
                continue
            if backend and exp.job.run_args:
                if _backend_from_args(exp.job.run_args) not in (backend, None):
                    continue
            derived = exp.results.derived_metrics or {}
            speedup = derived.get("speedup_vs_baseline_check") or derived.get("speedup_vs_baseline")
            try:
                speedup = float(speedup)
            except (TypeError, ValueError):
                speedup = None
            if speedup is None:
                continue
            improvement_pct = (speedup - 1.0) * 100.0
            if improvement_pct < min_gain_pct:
                continue
            patch_path = exp.patch_path
            if not patch_path or not Path(patch_path).exists():
                continue
            bucket.append((improvement_pct, exp))
    replay_candidates = exact_case_candidates if exact_case_candidates else same_app_candidates
    replay_candidates.sort(key=lambda item: item[0], reverse=True)
    actions: List[ActionIR] = []
    seen_patch_paths: set[str] = set()
    for improvement_pct, exp in replay_candidates:
        action = exp.action
        if action is None:
            continue
        if exp.patch_path and exp.patch_path in seen_patch_paths:
            continue
        patch_family = (action.parameters or {}).get("patch_family")
        meta = families_meta.get(patch_family, {})
        expected_effect = meta.get("patch_tags") if isinstance(meta, dict) else None
        if not expected_effect:
            expected_effect = action.expected_effect or ["compute_opt"]
        if isinstance(expected_effect, str):
            expected_effect = [expected_effect]
        expected_effect = [eff for eff in (expected_effect or []) if eff in _EXPECTED_EFFECTS]
        if not expected_effect:
            expected_effect = ["compute_opt"]
        action_id = _stable_compact_id("replay", str(action.action_id), exp.run_id, max_len=110)
        if action_id in existing_action_ids:
            continue
        params = dict(action.parameters or {})
        params["patch_path"] = exp.patch_path
        # Replay actions must bypass stale-path clearing logic.
        params["origin"] = "memory_replay"
        params.setdefault("replay_run_id", exp.run_id)
        replay = ActionIR(
            action_id=action_id,
            family="source_patch",
            description=f"Replay patch from {exp.run_id}.",
            applies_to=["source_patch"],
            parameters=params,
            preconditions=action.preconditions or [],
            constraints=action.constraints or [],
            expected_effect=expected_effect,
            risk_level=action.risk_level or meta.get("risk", "medium"),
            verification_plan=action.verification_plan,
        )
        actions.append(replay)
        existing_action_ids.add(action_id)
        if exp.patch_path:
            seen_patch_paths.add(exp.patch_path)
        if len(actions) >= max_actions:
            break
    return actions


def _collect_source_patch_chain_paths(
    tip_exp: Optional[ExperimentIR],
    experiments: List[ExperimentIR],
    max_depth: int = 128,
) -> List[str]:
    if not tip_exp:
        return []
    if tip_exp.action and "source_patch" in (tip_exp.action.applies_to or []):
        tip_params = tip_exp.action.parameters or {}
        tip_paths_raw = tip_params.get("patch_paths")
        if isinstance(tip_paths_raw, list):
            ordered_tip: List[str] = []
            for item in tip_paths_raw:
                if not isinstance(item, str):
                    continue
                norm = item.strip()
                if not norm or norm in ordered_tip:
                    continue
                ordered_tip.append(norm)
            tip_patch = tip_params.get("patch_path")
            if isinstance(tip_patch, str) and tip_patch.strip() and tip_patch.strip() not in ordered_tip:
                ordered_tip.append(tip_patch.strip())
            if ordered_tip:
                return ordered_tip
    by_run_id = {exp.run_id: exp for exp in experiments if exp and exp.run_id}
    seen_runs: set[str] = set()
    collected_rev: List[str] = []
    cursor = tip_exp
    depth = 0
    while cursor and depth < max_depth:
        depth += 1
        if cursor.run_id in seen_runs:
            break
        seen_runs.add(cursor.run_id)
        if cursor.action and "source_patch" in (cursor.action.applies_to or []):
            params = cursor.action.parameters or {}
            patch_path_raw = params.get("patch_path")
            if isinstance(patch_path_raw, str) and patch_path_raw:
                collected_rev.append(patch_path_raw)
            elif cursor.patch_path:
                collected_rev.append(cursor.patch_path)
        if not cursor.base_run_id:
            break
        cursor = by_run_id.get(cursor.base_run_id)
    # Walked tip -> root; reverse to apply root -> tip and keep first occurrence.
    ordered: List[str] = []
    seen_paths: set[str] = set()
    for path in reversed(collected_rev):
        norm = str(path).strip()
        if not norm or norm in seen_paths:
            continue
        seen_paths.add(norm)
        ordered.append(norm)
    return ordered


def _attach_patch_stack_to_action(
    action: Optional[ActionIR],
    base_patch_paths: List[str],
) -> None:
    if not action or "source_patch" not in (action.applies_to or []):
        return
    params = action.parameters or {}
    stacked: List[str] = []
    for path in base_patch_paths:
        if isinstance(path, str) and path and path not in stacked:
            stacked.append(path)
    patch_paths_raw = params.get("patch_paths")
    if isinstance(patch_paths_raw, list):
        for item in patch_paths_raw:
            if isinstance(item, str) and item and item not in stacked:
                stacked.append(item)
    patch_path = params.get("patch_path")
    if isinstance(patch_path, str) and patch_path and patch_path not in stacked:
        stacked.append(patch_path)
    if stacked:
        params["patch_paths"] = stacked
    action.parameters = params


def _build_source_patch_replay_action(
    best_exp: Optional[ExperimentIR],
    experiments: List[ExperimentIR],
) -> Optional[ActionIR]:
    if not best_exp or not best_exp.action:
        return None
    if "source_patch" not in (best_exp.action.applies_to or []):
        return None
    chain_paths = _collect_source_patch_chain_paths(best_exp, experiments)
    if not chain_paths:
        return None
    params = dict(best_exp.action.parameters or {})
    params["patch_paths"] = chain_paths
    params.pop("patch_path", None)
    return ActionIR(
        action_id="profile_probe.source_patch_replay",
        family="source_patch",
        description="Replay current best source patch chain for profiling probe.",
        applies_to=["source_patch"],
        parameters=params,
        preconditions=list(best_exp.action.preconditions or []),
        constraints=list(best_exp.action.constraints or []),
        expected_effect=list(best_exp.action.expected_effect or []),
        risk_level=best_exp.action.risk_level,
        verification_plan=best_exp.action.verification_plan,
    )


def _summarize_input_script(text: str) -> Dict[str, str]:
    summary: Dict[str, str] = {}
    keys = {
        "pair_style": "pair_style",
        "kspace_style": "kspace_style",
        "kspace_modify": "kspace_modify",
        "neighbor": "neighbor",
        "neigh_modify": "neigh_modify",
        "thermo": "thermo",
        "dump": "dump",
        "comm_modify": "comm_modify",
        "newton": "newton",
    }
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for prefix, key in keys.items():
            if line.startswith(prefix) and key not in summary:
                summary[key] = line
                break
    return summary


def _normalise_path_to_repo_rel(
    raw: str, repo_root: Path, patch_root: str
) -> Optional[str]:
    """Resolve a file path to repo-relative, adding patch_root prefix if needed.

    Standalone version usable outside the iteration-loop closure.
    """
    p = (raw or "").replace("\\", "/")
    if not p:
        return None
    if Path(p).is_absolute():
        try:
            p = Path(p).relative_to(repo_root).as_posix()
        except ValueError:
            # Best-effort remap for stale absolute paths after repo relocation.
            # Example: /old/root/hpc-agent-platform/third_party/bwa/ksw.c
            #      ->  third_party/bwa/ksw.c
            repo_marker = f"/{repo_root.name}/"
            idx = p.rfind(repo_marker)
            if idx != -1:
                candidate = p[idx + len(repo_marker):]
                if (repo_root / candidate).is_file():
                    p = candidate
                else:
                    remapped = _resolve_stale_abs_path_by_suffix(
                        str(repo_root), patch_root, p
                    )
                    if remapped:
                        p = remapped
                    else:
                        return None
            else:
                remapped = _resolve_stale_abs_path_by_suffix(
                    str(repo_root), patch_root, p
                )
                if remapped:
                    p = remapped
                else:
                    return None
    if patch_root and not p.startswith(patch_root):
        candidate = f"{patch_root}/{p}"
        if (repo_root / candidate).is_file():
            return candidate
    if (repo_root / p).is_file():
        return p
    remapped = _resolve_stale_abs_path_by_suffix(str(repo_root), patch_root, p)
    if remapped:
        return remapped
    return None


def _suffix_overlap_len(candidate_parts: List[str], raw_parts: List[str]) -> int:
    score = 0
    i = len(candidate_parts) - 1
    j = len(raw_parts) - 1
    while i >= 0 and j >= 0 and candidate_parts[i] == raw_parts[j]:
        score += 1
        i -= 1
        j -= 1
    return score


@lru_cache(maxsize=1024)
def _resolve_stale_abs_path_by_suffix(
    repo_root_str: str, patch_root: str, raw_path: str
) -> Optional[str]:
    """Map stale absolute paths to current repo via robust suffix matching."""
    repo_root = Path(repo_root_str)
    search_root = (repo_root / patch_root) if patch_root else repo_root
    if not search_root.exists():
        search_root = repo_root
    filename = Path(raw_path).name
    if not filename:
        return None

    raw_parts = [seg for seg in raw_path.replace("\\", "/").split("/") if seg]
    best_rel: Optional[str] = None
    best_score = 0
    best_depth = 0
    tie = False

    for match in search_root.rglob(filename):
        if not match.is_file():
            continue
        rel = match.relative_to(repo_root).as_posix()
        rel_parts = rel.split("/")
        score = _suffix_overlap_len(rel_parts, raw_parts)
        depth = len(rel_parts)
        if score > best_score or (score == best_score and depth > best_depth):
            best_rel = rel
            best_score = score
            best_depth = depth
            tie = False
        elif score == best_score and depth == best_depth and rel != best_rel:
            tie = True

    if best_rel and not tie and best_score >= 2:
        return best_rel
    return None


def _tau_hotspot_files(
    tau_hotspots: List[Dict[str, object]],
    repo_root: Path,
    patch_root: str = "",
) -> List[str]:
    """Extract source file paths from TAU profiling entries.

    Returns deduplicated repo-relative paths sorted by aggregate exclusive
    time (descending).  Files that cannot be resolved are silently dropped.
    """
    file_time: Dict[str, float] = {}
    for entry in tau_hotspots:
        src = entry.get("file", "")
        if not src or not isinstance(src, str):
            continue
        norm = _normalise_path_to_repo_rel(src, repo_root, patch_root)
        if norm:
            file_time[norm] = file_time.get(norm, 0.0) + float(entry.get("exclusive_us", 0))
    return sorted(file_time, key=lambda f: file_time[f], reverse=True)


def _hotspot_weight(entry: Dict[str, object]) -> float:
    """Compute a comparable hotspot weight across profiler schemas."""
    exclusive_us = float(entry.get("exclusive_us", 0.0) or 0.0)
    exclusive_ms = float(entry.get("exclusive_ms", 0.0) or 0.0)
    inclusive_us = float(entry.get("inclusive_us", 0.0) or 0.0)
    inclusive_ms = float(entry.get("inclusive_ms", 0.0) or 0.0)
    weight = 0.0
    weight += exclusive_us
    weight += exclusive_ms * 1000.0
    # Inclusive time is weaker evidence than exclusive time.
    weight += inclusive_us * 0.25
    weight += inclusive_ms * 250.0
    if weight <= 0.0:
        calls = float(entry.get("calls", 0.0) or 0.0)
        weight = calls
    return max(weight, 0.0)


def _merge_function_hotspots(
    repo_root: Path,
    patch_root: str,
    baseline_exp: Optional[ExperimentIR],
    best_exp: Optional[ExperimentIR],
    experiments: List[ExperimentIR],
    max_entries: int = 120,
) -> List[Dict[str, object]]:
    """Merge hotspots across baseline/best/history for robust patch guidance."""
    candidates: List[ExperimentIR] = []
    seen_run_ids: set[str] = set()
    for exp in [best_exp, baseline_exp]:
        if exp and exp.run_id not in seen_run_ids:
            seen_run_ids.add(exp.run_id)
            candidates.append(exp)
    pass_exps = sorted(
        [
            exp for exp in experiments
            if exp.verdict == "PASS" and exp.profile_report and exp.profile_report.tau_hotspots
        ],
        key=lambda item: item.results.runtime_seconds if item.results.runtime_seconds > 0 else 1.0e18,
    )
    for exp in pass_exps:
        if exp.run_id in seen_run_ids:
            continue
        seen_run_ids.add(exp.run_id)
        candidates.append(exp)
        if len(candidates) >= 16:
            break

    merged: Dict[Tuple[str, str, int], Tuple[float, Dict[str, object]]] = {}
    for exp in candidates:
        hotspots = (exp.profile_report.tau_hotspots if exp.profile_report else None) or []
        for raw in hotspots:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            file_raw = item.get("file", "")
            file_norm = ""
            if isinstance(file_raw, str) and file_raw:
                file_norm = _normalise_path_to_repo_rel(file_raw, repo_root, patch_root) or file_raw
                item["file"] = file_norm
            fn = str(item.get("name", "") or "")
            try:
                line = int(item.get("line", 0) or 0)
            except (TypeError, ValueError):
                line = 0
            key = (file_norm, fn, line)
            score = _hotspot_weight(item)
            prev = merged.get(key)
            if prev is None or score > prev[0]:
                item["_source_run_id"] = exp.run_id
                merged[key] = (score, item)

    ranked = sorted(merged.values(), key=lambda pair: pair[0], reverse=True)
    out: List[Dict[str, object]] = []
    for _, item in ranked[:max_entries]:
        item.pop("_source_run_id", None)
        out.append(item)
    return out


def _hotspot_map(
    input_text: str,
    repo_root: Path,
    run_args: List[str],
    patch_root: str = "",
    deep_analysis_result: Optional["DeepCodeAnalysisResult"] = None,
    function_hotspots: Optional[List[Dict[str, object]]] = None,
    tau_hotspots: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    pair_files = _pair_style_files(input_text, repo_root, run_args)
    hotspot_files = list(pair_files)
    # Priority: deep_analysis > TAU > pair_style_files
    # `tau_hotspots` is a legacy field name. On macOS this list can contain
    # xctrace-derived hotspots normalized into the same schema.
    if function_hotspots is None:
        function_hotspots = tau_hotspots
    if function_hotspots:
        tau_files = _tau_hotspot_files(function_hotspots, repo_root, patch_root)
        for tf in reversed(tau_files):
            if tf not in hotspot_files:
                hotspot_files.insert(0, tf)
    if deep_analysis_result and deep_analysis_result.hotspot_files:
        for hf in deep_analysis_result.hotspot_files:
            norm = _normalise_path_to_repo_rel(hf, repo_root, patch_root)
            if norm and norm not in hotspot_files:
                hotspot_files.insert(0, norm)
    return {
        "pair_style_files": pair_files,
        "hotspot_files": hotspot_files,
    }


def _adapter_hints(job: JobIR, adapter_cfg: Optional[Dict[str, object]]) -> Dict[str, object]:
    base_keys = [
        "target_file",
        "target_anchor",
        "snippet_tag",
        "origin",
        "wrapper_id",
    ]
    allowlist = app_input_allowlist(job.app)
    if job.app != "lammps":
        hints: Dict[str, object] = {"parameter_keys": base_keys}
        if allowlist:
            hints["input_edit_allowlist"] = allowlist
        return hints
    return {
        "input_edit_allowlist": allowlist,
        "parameter_keys": [
            *base_keys,
            "neighbor_skin",
            "neighbor_every",
            "output_thermo_every",
            "output_dump_every",
            "comm_cutoff",
            "comm_mode",
            "newton_setting",
            "kspace_accuracy",
            "kspace_style",
            "backend_enable",
            "backend_threads",
            "env",
            "run_args",
            "runtime_env",
            "lib_env",
            "mpi_env",
            "io_env",
        ],
    }


def _build_planner_context(
    job: JobIR,
    input_summary: Dict[str, str],
    profile: ProfileReport,
    profile_features: Dict[str, object],
    hotspot_map: Dict[str, object],
    system_caps: Dict[str, object],
) -> Dict[str, object]:
    return {
        "job": {
            "app": job.app,
            "case_id": job.case_id,
            "tags": job.tags,
        },
        "input_summary": input_summary,
        "profile": {
            "timing_breakdown": profile.timing_breakdown,
            "system_metrics": profile.system_metrics,
        },
        "profile_features": profile_features,
        "hotspot_map": hotspot_map,
        "system_caps": system_caps,
    }


def _action_space_summary(
    actions: List[ActionIR],
    patch_families: Optional[Dict[str, object]],
) -> Dict[str, object]:
    families = sorted({action.family for action in actions})
    patch_family_ids: List[str] = []
    patch_family_defs: List[Dict[str, object]] = []
    for item in (patch_families or {}).get("families", []):
        if not isinstance(item, dict):
            continue
        family_id = item.get("id")
        if not family_id:
            continue
        patch_family_ids.append(str(family_id))
        patch_family_defs.append(
            {
                "id": str(family_id),
                "description": item.get("description", ""),
                "transform_types": item.get("transform_types", []),
                "patch_tags": item.get("patch_tags", []),
                "risk": item.get("risk", ""),
                "requires_caps": item.get("requires_caps", []),
                "reference_file": item.get("reference_file", ""),
                "profiling_trigger": item.get("profiling_trigger", []),
                "compiler_covered": bool(item.get("compiler_covered")),
                "tier": item.get("tier", "algorithm"),
            }
        )
    patch_actions = [action.action_id for action in actions if action.family == "source_patch"]
    if len(patch_actions) > 50:
        patch_actions = patch_actions[:50]
    return {
        "families": families,
        "patch_families": patch_family_ids,
        "patch_family_defs": patch_family_defs,
        "source_patch_actions": patch_actions,
    }


def _summarize_code_survey(
    code_survey_payload: Optional[Dict[str, object]],
) -> Dict[str, object]:
    summary: Dict[str, object] = {"count": 0, "files": [], "patch_families": []}
    if not code_survey_payload:
        return summary
    opportunities = code_survey_payload.get("opportunities", [])
    if not isinstance(opportunities, list):
        return summary
    files: List[str] = []
    families: List[str] = []
    for item in opportunities:
        if not isinstance(item, dict):
            continue
        file_path = item.get("file_path")
        patch_family = item.get("patch_family")
        if isinstance(file_path, str):
            files.append(file_path)
        if isinstance(patch_family, str):
            families.append(patch_family)
    summary["count"] = len(opportunities)
    summary["files"] = sorted(set(files))
    summary["patch_families"] = sorted(set(families))
    return summary


def _actions_from_opportunities(
    opportunities: List[Dict[str, object]],
    patch_families: Optional[Dict[str, object]],
    max_actions: int = 4,
) -> List[ActionIR]:
    if not opportunities or not patch_families:
        return []
    families_cfg = patch_families.get("families")
    if not isinstance(families_cfg, list):
        return []
    family_meta: Dict[str, Dict[str, object]] = {}
    for entry in families_cfg:
        if isinstance(entry, dict) and entry.get("id"):
            family_meta[str(entry["id"])] = entry
    normalized: List[Dict[str, object]] = []
    for item in opportunities:
        if not isinstance(item, dict):
            continue
        patch_family = item.get("patch_family")
        file_path = item.get("file_path")
        if not isinstance(patch_family, str) or patch_family not in family_meta:
            continue
        if not isinstance(file_path, str):
            continue
        normalized.append(item)
    normalized.sort(
        key=lambda item: float(item.get("confidence") or 0.0),
        reverse=True,
    )
    actions: List[ActionIR] = []
    for item in normalized[:max_actions]:
        patch_family = str(item.get("patch_family"))
        meta = family_meta.get(patch_family, {})
        tag_list = meta.get("patch_tags") or []
        expected_effect = [tag for tag in tag_list if tag in _EXPECTED_EFFECTS]
        action_id = f"generated.patch.{patch_family}.{item.get('opportunity_id')}"
        params: Dict[str, object] = {
            "patch_family": patch_family,
            "target_file": item.get("file_path"),
            "target_anchor": item.get("anchor_hint"),
            "snippet_tag": item.get("snippet_tag"),
            "origin": "code_survey",
        }
        evidence = item.get("evidence")
        if isinstance(evidence, list) and evidence:
            params["evidence"] = evidence
        actions.append(
            ActionIR(
                action_id=str(action_id),
                family="source_patch",
                description=str(meta.get("description") or patch_family),
                applies_to=["source_patch"],
                parameters=params,
                expected_effect=expected_effect,
                risk_level=str(meta.get("risk") or "medium"),
                verification_plan=VerificationPlan(
                    gates=list(meta.get("mandatory_gates") or ["runtime"]),
                    thresholds={},
                ),
            )
        )
    return actions


def _write_generated_actions(
    run_dir: Path,
    generated_actions: List[ActionIR],
    ideas: Optional[Dict[str, object]] = None,
    code_survey: Optional[Dict[str, object]] = None,
) -> None:
    payload = {
        "actions": [action.model_dump() for action in generated_actions],
        "ideas": ideas or {},
        "code_survey": code_survey or {},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "generated_actions.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )


def _patch_rules(
    adapter_cfg: Optional[Dict[str, object]],
    policy: Optional[Dict[str, object]],
) -> Dict[str, object]:
    rules: Dict[str, object] = {}
    if isinstance(policy, dict):
        base = policy.get("source_patch_rules")
        if isinstance(base, dict):
            rules.update(base)
    if isinstance(adapter_cfg, dict):
        adapter_rules = adapter_cfg.get("patch_rules")
        if isinstance(adapter_rules, dict):
            rules.update(adapter_rules)
    # Do not enforce file-scope restrictions from patch_root/allowed_globs.
    # Keep the rest of patch rules (limits/forbidden patterns/etc.).
    rules.pop("patch_root", None)
    rules.pop("allowed_globs", None)
    return rules


def _patch_scope_config(policy: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not isinstance(policy, dict):
        return {}
    scope = policy.get("source_patch_scope")
    if not isinstance(scope, dict):
        return {}
    levels = scope.get("levels")
    if not isinstance(levels, list):
        levels = []
    start_level = scope.get("start_level")
    if isinstance(start_level, str) and start_level not in levels:
        start_level = None
    return {
        "levels": levels,
        "start_level": start_level or (levels[0] if levels else None),
        "promote": scope.get("promote", {}) if isinstance(scope.get("promote"), dict) else {},
        "coverage": scope.get("coverage", {}) if isinstance(scope.get("coverage"), dict) else {},
    }


def _apply_patch_scope(
    patch_rules: Dict[str, object],
    scope_id: Optional[str],
) -> Dict[str, object]:
    if not scope_id:
        return patch_rules
    scope_levels = patch_rules.get("scope_levels")
    if not isinstance(scope_levels, dict):
        return patch_rules
    scope_cfg = scope_levels.get(scope_id)
    if not isinstance(scope_cfg, dict):
        return patch_rules
    merged = dict(patch_rules)
    for key in (
        "hotspot_only",
        "max_lines_changed",
        "max_files_changed",
        "max_snippets",
        "max_context_chars",
    ):
        if key in scope_cfg:
            merged[key] = scope_cfg[key]
    merged["active_scope"] = scope_id
    return merged


def _untried_patch_families(
    experiments: List[ExperimentIR],
    patch_families: Optional[Dict[str, object]],
) -> set[str]:
    if not patch_families:
        return set()
    families = patch_families.get("families")
    if not isinstance(families, list):
        return set()
    all_families = {
        str(entry["id"])
        for entry in families
        if isinstance(entry, dict) and entry.get("id")
    }
    tried: set[str] = set()
    for exp in experiments:
        action = exp.action
        if not action or action.family != "source_patch":
            continue
        patch_family = (action.parameters or {}).get("patch_family")
        if isinstance(patch_family, str):
            tried.add(patch_family)
    return all_families - tried


def _exhausted_patch_families(
    experiments: List[ExperimentIR],
    max_failures_per_family: int = 2,
) -> set[str]:
    """Return patch families that have been tried and failed too many times."""
    family_failures: Dict[str, int] = {}
    family_passes: set[str] = set()
    for exp in experiments:
        action = exp.action
        if not action or action.family != "source_patch":
            continue
        patch_family = (action.parameters or {}).get("patch_family")
        if not isinstance(patch_family, str):
            continue
        if exp.verdict == "PASS":
            family_passes.add(patch_family)
        else:
            family_failures[patch_family] = family_failures.get(patch_family, 0) + 1
    # A family is exhausted if it has failed enough times AND never passed
    return {
        fam for fam, count in family_failures.items()
        if count >= max_failures_per_family and fam not in family_passes
    }


def _filter_source_patch_for_coverage(
    actions: List[ActionIR],
    untried_families: set[str],
) -> Tuple[List[ActionIR], bool]:
    if not untried_families:
        return actions, False
    filtered: List[ActionIR] = []
    for action in actions:
        if action.family != "source_patch":
            filtered.append(action)
            continue
        patch_family = (action.parameters or {}).get("patch_family")
        if isinstance(patch_family, str) and patch_family in untried_families:
            filtered.append(action)
    if any(action.family == "source_patch" for action in filtered):
        return filtered, True
    return actions, False


def _select_distinct_patch_families(
    ranked: List[RankedAction],
    limit: int,
) -> List[RankedAction]:
    selected: List[RankedAction] = []
    seen_families: set[str] = set()
    for item in ranked:
        action = item.action
        if action.family == "source_patch":
            patch_family = (action.parameters or {}).get("patch_family") or action.action_id
            if patch_family in seen_families:
                continue
            seen_families.add(patch_family)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def _expand_allowed_files(repo_root: Path, rules: Dict[str, object]) -> List[str]:
    allowed = rules.get("allowed_globs")
    if not isinstance(allowed, list):
        return []
    files: List[str] = []
    for pattern in allowed:
        if not isinstance(pattern, str):
            continue
        for path in repo_root.glob(pattern):
            if path.is_file():
                rel = path.relative_to(repo_root).as_posix()
                files.append(rel)
    return sorted(set(files))


def _collect_repo_source_files(
    repo_root: Path,
    patch_root: str = "",
    max_files: int = 1200,
) -> List[str]:
    """Collect a broad set of source files for agentic exploration."""
    search_root = (repo_root / patch_root) if patch_root else repo_root
    if not search_root.exists():
        search_root = repo_root
    max_files = max(1, int(max_files or 1200))
    allowed_exts = {
        ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
        ".cu", ".cuh", ".inc", ".ipp", ".py", ".f", ".f90", ".f95",
    }
    skip_dirs = {
        ".git", "artifacts", "build", "__pycache__", ".pytest_cache",
        ".idea", ".vscode", ".venv", "venv", "node_modules",
    }
    files: List[str] = []
    for root, dirnames, filenames in os.walk(search_root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for name in filenames:
            if len(files) >= max_files:
                return files
            ext = Path(name).suffix.lower()
            if ext and ext not in allowed_exts:
                continue
            path = Path(root) / name
            if not path.is_file():
                continue
            rel = path.relative_to(repo_root).as_posix()
            files.append(rel)
    return files


def _build_navigation_hints(
    repo_root: Path,
    files: List[str],
    max_hints: int,
) -> List[Dict[str, object]]:
    """Build lightweight navigation hints for files the agent should examine.

    Returns file pointers with optional hotspot locations.  The agentic
    patcher uses its own ``read_file`` / ``grep`` tools to explore code —
    these hints save the initial 3-4 tool calls needed to find the
    starting point.
    """
    hints: List[Dict[str, object]] = []
    if max_hints <= 0:
        return hints
    for rel in files:
        if len(hints) >= max_hints:
            break
        path = repo_root / rel
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        hint: Dict[str, object] = {
            "path": rel,
            "total_lines": len(lines),
        }
        marker_line, _ = _find_marker_line(lines)
        if marker_line is not None:
            hint["hotspot_line"] = marker_line + 1  # 1-indexed
            func_start = _find_function_start(lines, marker_line)
            if func_start is not None:
                hint["function_start"] = func_start + 1
                sig = lines[func_start].strip()[:120]
                hint["function_signature"] = sig
        hints.append(hint)
    return hints


def _collect_code_snippets(
    repo_root: Path,
    files: List[str],
    max_snippets: int,
    max_chars: int,
) -> List[Dict[str, object]]:
    snippets: List[Dict[str, object]] = []
    if max_snippets <= 0 or max_chars <= 0:
        return snippets
    budget_per = max(800, max_chars // max_snippets)
    for rel in files:
        if len(snippets) >= max_snippets:
            break
        path = repo_root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        blocks = _build_snippet_blocks(text, budget_per)
        seen = set()
        for block in blocks:
            if len(snippets) >= max_snippets:
                break
            snippet = block["snippet"]
            if not snippet or snippet in seen:
                continue
            seen.add(snippet)
            anchor_hints = _anchor_hints_for_snippet(text, snippet)
            features = _snippet_features(snippet)
            snippets.append(
                {
                    "path": rel,
                    "snippet": snippet,
                    "tag": block.get("tag"),
                    "start_line": block.get("start_line"),
                    "end_line": block.get("end_line"),
                    "anchor_hints": anchor_hints,
                    "features": features,
                }
            )
    return snippets


def _infer_target_anchor_for_action(
    action: ActionIR,
    repo_root: Path,
    code_snippets: List[Dict[str, object]],
) -> Optional[str]:
    params = action.parameters if isinstance(action.parameters, dict) else {}
    raw_anchor = params.get("target_anchor")
    if isinstance(raw_anchor, str) and raw_anchor.strip():
        return raw_anchor.strip()
    target_file = params.get("target_file")
    target_file = target_file if isinstance(target_file, str) and target_file else None
    raw_funcs = params.get("target_functions")
    target_functions: List[str] = []
    if isinstance(raw_funcs, list):
        target_functions = [
            str(item).strip() for item in raw_funcs if isinstance(item, str) and str(item).strip()
        ]
    if target_file and target_functions:
        path = repo_root / target_file
        if path.is_file():
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for func in target_functions:
                pat = re.compile(rf"\b{re.escape(func)}\s*\(")
                for line in lines:
                    if pat.search(line):
                        return line.strip()
    for snippet_item in code_snippets:
        snippet_path = snippet_item.get("path")
        if target_file and snippet_path != target_file:
            continue
        hints = snippet_item.get("anchor_hints")
        if isinstance(hints, list):
            for hint in hints:
                if isinstance(hint, str) and hint.strip():
                    return hint.strip()
        snippet_text = str(snippet_item.get("snippet") or "")
        for func in target_functions:
            pat = re.compile(rf"\b{re.escape(func)}\s*\(")
            for line in snippet_text.splitlines():
                if pat.search(line):
                    return line.strip()
    if target_functions:
        return f"{target_functions[0]}("
    return None


def _fallback_reference_template_for_action(
    action: ActionIR,
    target_file: Optional[str],
    target_anchor: Optional[str],
) -> Dict[str, str]:
    params = action.parameters if isinstance(action.parameters, dict) else {}
    hypothesis = params.get("hypothesis")
    hypothesis_text = hypothesis if isinstance(hypothesis, str) else ""
    return {
        "reference_file": target_file or "",
        "description": (
            "No canonical reference template is available for this patch family. "
            "Derive a minimal, semantics-preserving transformation directly from action "
            "hypothesis and code_snippets."
        ),
        "before": (
            f"target_file={target_file or ''}\n"
            f"target_anchor={target_anchor or ''}"
        ),
        "after": (
            f"expected_effect={','.join(action.expected_effect or [])}\n"
            f"hypothesis={hypothesis_text}"
        ),
        "full_reference": "",
    }


def _timing_key_to_hotspot(key: str) -> Optional[str]:
    lowered = key.lower()
    if "pair" in lowered:
        return "pair"
    if "neigh" in lowered:
        return "neigh"
    if "comm" in lowered:
        return "comm"
    if "kspace" in lowered:
        return "kspace"
    if "compute" in lowered or "modify" in lowered:
        return "compute"
    return None


def _slice_around_compute(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    lines = text.splitlines()
    marker_line, _ = _find_marker_line(lines)
    if marker_line is None:
        return _slice_line_block(lines, 0, min(len(lines), 200), max_chars)[0]
    start = max(0, marker_line - 40)
    end = min(len(lines), marker_line + 120)
    return _slice_line_block(lines, start, end, max_chars)[0]


def _find_marker_line(lines: List[str]) -> Tuple[Optional[int], str]:
    markers = [
        "for (int jj",
        "for (jj",
        "for (int ii",
        "for (ii",
        "eval(",
        "compute<",
        "compute(",
        "compute ",
    ]
    for marker in markers:
        for idx, line in enumerate(lines):
            if marker in line:
                return idx, marker
    return None, ""


def _find_function_start(lines: List[str], from_idx: int) -> Optional[int]:
    for idx in range(from_idx, -1, -1):
        line = lines[idx]
        if "compute" in line and "(" in line:
            if "{" in line:
                return idx
            if idx + 1 < len(lines) and "{" in lines[idx + 1]:
                return idx
        if "::" in line and "(" in line and "{" in line:
            return idx
    return None


def _extract_function_block(
    lines: List[str],
    start_idx: int,
    max_chars: int,
) -> Tuple[str, int, int]:
    if start_idx < 0 or start_idx >= len(lines):
        return "", start_idx, start_idx
    brace_count = 0
    started = False
    end_idx: Optional[int] = None
    for idx in range(start_idx, len(lines)):
        for ch in lines[idx]:
            if ch == "{":
                brace_count += 1
                started = True
            elif ch == "}":
                if not started:
                    continue
                brace_count -= 1
                if brace_count == 0:
                    end_idx = idx + 1
                    break
        if end_idx is not None:
            break
    if end_idx is None:
        end_idx = min(len(lines), start_idx + 320)
    return _slice_line_block(lines, start_idx, end_idx, max_chars)


def _slice_line_block(
    lines: List[str],
    start: int,
    end: int,
    max_chars: int,
) -> Tuple[str, int, int]:
    if start < 0:
        start = 0
    if end > len(lines):
        end = len(lines)
    if start >= end:
        return "", start, end
    total = 0
    out_lines = []
    actual_end = start
    for idx in range(start, end):
        line = lines[idx]
        line_len = len(line) + 1
        if total + line_len > max_chars and out_lines:
            break
        out_lines.append(line)
        total += line_len
        actual_end = idx + 1
    return "\n".join(out_lines), start, actual_end


def _build_snippet_blocks(text: str, max_chars: int) -> List[Dict[str, object]]:
    lines = text.splitlines()
    marker_line, _ = _find_marker_line(lines)
    if marker_line is None:
        snippet, start, end = _slice_line_block(lines, 0, min(len(lines), 200), max_chars)
        return [{"tag": "head", "snippet": snippet, "start_line": start + 1, "end_line": end}]
    blocks: List[Dict[str, object]] = []
    hot_start = max(0, marker_line - 5)
    hot_end = min(len(lines), marker_line + 200)
    snippet, start, end = _slice_line_block(lines, hot_start, hot_end, max_chars)
    if snippet:
        blocks.append({"tag": "hotspot", "snippet": snippet, "start_line": start + 1, "end_line": end})
    decl_start = max(0, marker_line - 120)
    decl_end = min(len(lines), marker_line + 5)
    snippet, start, end = _slice_line_block(lines, decl_start, decl_end, max_chars)
    if snippet:
        blocks.append({"tag": "declarations", "snippet": snippet, "start_line": start + 1, "end_line": end})
    func_start = _find_function_start(lines, marker_line)
    if func_start is not None:
        snippet, start, end = _extract_function_block(lines, func_start, max_chars)
        if snippet:
            blocks.append({"tag": "function_full", "snippet": snippet, "start_line": start + 1, "end_line": end})
        helper_end = min(len(lines), func_start + 140)
        snippet, start, end = _slice_line_block(lines, func_start, helper_end, max_chars)
        if snippet:
            blocks.append({"tag": "function_header", "snippet": snippet, "start_line": start + 1, "end_line": end})
    extra_funcs = _find_named_functions(
        lines,
        ["compute_inner", "compute_middle", "compute_outer"],
    )
    for name, idx in extra_funcs:
        snippet, start, end = _slice_line_block(lines, idx, min(len(lines), idx + 180), max_chars)
        if snippet:
            blocks.append(
                {
                    "tag": f"function_{name}",
                    "snippet": snippet,
                    "start_line": start + 1,
                    "end_line": end,
                }
            )
    return blocks


def _find_named_functions(lines: List[str], names: List[str]) -> List[Tuple[str, int]]:
    found: List[Tuple[str, int]] = []
    for idx, line in enumerate(lines):
        for name in names:
            if name in line and "(" in line:
                found.append((name, idx))
    return found


def _anchor_hints_for_snippet(
    full_text: str,
    snippet: str,
    max_hints: int = 5,
) -> List[str]:
    lines = snippet.splitlines()
    hints: List[str] = []
    if len(lines) < 3:
        return hints
    for window in (5, 4, 3):
        if len(hints) >= max_hints:
            break
        if len(lines) < window:
            continue
        for idx in range(0, len(lines) - window + 1):
            block = "\n".join(lines[idx : idx + window])
            if not block.strip():
                continue
            if _count_occurrences(full_text, block) == 1:
                hints.append(block)
            if len(hints) >= max_hints:
                break
    return hints


_LOOP_HEADER_RE = re.compile(r"\b(for|while)\s*\(")
_FOR_COND_RE = re.compile(r"\bfor\s*\(([^;]*);([^;]*);([^)]+)\)")
_IF_RE = re.compile(r"\bif\s*\(")
_FLAG_COND_RE = re.compile(r"\bif\s*\([^)]*\b[A-Z0-9_]*FLAG\b")


def _loop_signature(line: str) -> str:
    match = _FOR_COND_RE.search(line)
    if not match:
        return re.sub(r"\s+", "", line.strip())
    cond = match.group(2)
    return re.sub(r"\s+", "", cond.strip())


def _snippet_features(snippet: str) -> Dict[str, object]:
    lines = snippet.splitlines()
    loop_headers: List[str] = []
    loop_lines: List[int] = []
    loop_signatures: List[str] = []
    for idx, line in enumerate(lines):
        if _LOOP_HEADER_RE.search(line):
            loop_headers.append(line.strip())
            loop_lines.append(idx)
            loop_signatures.append(_loop_signature(line))
    has_adjacent = False
    adjacent_same_signature = False
    for idx in range(1, len(loop_lines)):
        if loop_lines[idx] - loop_lines[idx - 1] <= 8:
            has_adjacent = True
            if loop_signatures[idx] == loop_signatures[idx - 1]:
                adjacent_same_signature = True
            break
    conditional_count = 0
    has_flag_condition = False
    for line in lines:
        if _IF_RE.search(line):
            conditional_count += 1
        if _FLAG_COND_RE.search(line):
            has_flag_condition = True
    return {
        "loop_count": len(loop_headers),
        "loop_headers": loop_headers[:4],
        "loop_signatures": loop_signatures[:4],
        "has_adjacent_loops": has_adjacent,
        "adjacent_loop_signature": adjacent_same_signature,
        "conditional_count": conditional_count,
        "has_flag_condition": has_flag_condition,
    }


def _snippet_feature_map(
    snippets: List[Dict[str, object]],
) -> Dict[Tuple[str, Optional[str]], Dict[str, object]]:
    mapping: Dict[Tuple[str, Optional[str]], Dict[str, object]] = {}
    aggregate: Dict[str, Dict[str, object]] = {}
    for item in snippets:
        path = item.get("path")
        tag = item.get("tag")
        features = item.get("features")
        if isinstance(path, str) and isinstance(features, dict):
            mapping[(path, tag if isinstance(tag, str) else None)] = features
            agg = aggregate.setdefault(
                path,
                {
                    "loop_count": 0,
                    "loop_headers": [],
                    "loop_signatures": [],
                    "has_adjacent_loops": False,
                    "adjacent_loop_signature": False,
                    "conditional_count": 0,
                    "has_flag_condition": False,
                },
            )
            agg["loop_count"] = max(
                int(agg.get("loop_count", 0) or 0),
                int(features.get("loop_count", 0) or 0),
            )
            if features.get("loop_headers"):
                agg["loop_headers"] = list(dict.fromkeys(agg["loop_headers"] + list(features["loop_headers"])))
            if features.get("loop_signatures"):
                agg["loop_signatures"] = list(
                    dict.fromkeys(agg["loop_signatures"] + list(features["loop_signatures"]))
                )
            agg["has_adjacent_loops"] = bool(agg["has_adjacent_loops"] or features.get("has_adjacent_loops"))
            agg["adjacent_loop_signature"] = bool(
                agg["adjacent_loop_signature"] or features.get("adjacent_loop_signature")
            )
            agg["conditional_count"] = max(
                int(agg.get("conditional_count", 0) or 0),
                int(features.get("conditional_count", 0) or 0),
            )
            agg["has_flag_condition"] = bool(
                agg["has_flag_condition"] or features.get("has_flag_condition")
            )
    for path, agg in aggregate.items():
        mapping[(path, None)] = agg
    return mapping


def _structural_allowed(patch_family: str, features: Dict[str, object]) -> bool:
    # Allow LLM autonomy; structural signals are soft evidence, not hard filters.
    return True


def _filter_opportunities_by_structure(
    opportunities: List[Dict[str, object]],
    snippet_features: Dict[Tuple[str, Optional[str]], Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not opportunities or not snippet_features:
        return opportunities, []
    kept: List[Dict[str, object]] = []
    dropped: List[Dict[str, object]] = []
    for item in opportunities:
        if not isinstance(item, dict):
            continue
        patch_family = item.get("patch_family")
        file_path = item.get("file_path")
        snippet_tag = item.get("snippet_tag")
        if not isinstance(patch_family, str) or not isinstance(file_path, str):
            kept.append(item)
            continue
        features = snippet_features.get(
            (file_path, snippet_tag if isinstance(snippet_tag, str) else None)
        ) or snippet_features.get((file_path, None))
        if not isinstance(features, dict):
            kept.append(item)
            continue
        if _structural_allowed(patch_family, features):
            kept.append(item)
        else:
            dropped.append(
                {
                    "opportunity_id": item.get("opportunity_id"),
                    "patch_family": patch_family,
                    "file_path": file_path,
                    "snippet_tag": snippet_tag,
                }
            )
    return kept, dropped


def _filter_generated_actions_by_structure(
    actions: List[ActionIR],
    snippet_features: Dict[Tuple[str, Optional[str]], Dict[str, object]],
) -> Tuple[List[ActionIR], List[Dict[str, object]]]:
    if not actions or not snippet_features:
        return actions, []
    kept: List[ActionIR] = []
    dropped: List[Dict[str, object]] = []
    for action in actions:
        if action.family != "source_patch":
            kept.append(action)
            continue
        params = dict(action.parameters or {})
        patch_family = params.get("patch_family")
        target_file = params.get("target_file")
        snippet_tag = params.get("snippet_tag")
        if not isinstance(patch_family, str) or not isinstance(target_file, str):
            kept.append(action)
            continue
        features = snippet_features.get(
            (target_file, snippet_tag if isinstance(snippet_tag, str) else None)
        ) or snippet_features.get((target_file, None))
        if not isinstance(features, dict):
            kept.append(action)
            continue
        if _structural_allowed(patch_family, features):
            kept.append(action)
        else:
            dropped.append(
                {
                    "action_id": action.action_id,
                    "patch_family": patch_family,
                    "target_file": target_file,
                    "snippet_tag": snippet_tag,
                }
            )
    return kept, dropped


def _retry_patch_after_review(
    patch_debugger: PatchDebugAgent,
    patch_reviewer: PatchReviewAgent,
    action: ActionIR,
    profile: ProfileReport,
    patch_rules: Dict[str, object],
    allowed_files: List[str],
    code_snippets: List[Dict[str, object]],
    repo_root: Path,
    patch_proposal: PatchProposal,
    review_reasons: List[str],
    debug_max_attempts: int,
    snippet_files: List[str],
    iteration: int,
    trace_events: List[Dict[str, object]],
    reporter: Optional[ConsoleUI],
) -> Optional[PatchProposal]:
    if debug_max_attempts <= 0:
        return None
    reason_text = "\n".join(review_reasons).strip() or "review_failed"
    debug_snippets = list(code_snippets)
    debug_allowed_files = list(allowed_files)
    for attempt in range(debug_max_attempts):
        debug_proposal = patch_debugger.repair(
            action=action,
            profile=profile,
            patch_rules=patch_rules,
            allowed_files=debug_allowed_files,
            code_snippets=debug_snippets,
            repo_root=repo_root,
            patch_diff=patch_proposal.patch_diff,
            build_log=reason_text,
            feedback="review_failed",
        )
        if reporter:
            note = reason_text
            if debug_proposal and debug_proposal.missing_fields:
                note = "; ".join(debug_proposal.missing_fields)
            reporter.patch_debug(
                action.action_id,
                attempt + 1,
                debug_proposal.status if debug_proposal else "NO_RESPONSE",
                note,
            )
        trace_events.append(
            {
                "event": "patch_debug_review",
                "agent": "PatchDebugAgent",
                "iteration": iteration,
                "action_id": action.action_id,
                "attempt": attempt + 1,
                "proposal": debug_proposal.model_dump() if debug_proposal else None,
            }
        )
        if not debug_proposal:
            continue
        if debug_proposal.status == "NEED_MORE_CONTEXT":
            base_context = int(patch_rules.get("max_context_chars", 0) or 0)
            if base_context <= 0:
                base_context = 60000
            expand_budget = base_context
            expand_cap = max(base_context * 4, 240000)
            max_expand_rounds = int(patch_rules.get("max_context_expand_rounds", 0) or 3)
            expand_round = 0
            last_fingerprint = _snippet_fingerprint(debug_snippets)
            while debug_proposal and debug_proposal.status == "NEED_MORE_CONTEXT":
                expand_round += 1
                if expand_round > max_expand_rounds:
                    break
                expand_budget = min(expand_budget + base_context, expand_cap)
                expanded = _expand_snippets_for_missing_context(
                    debug_proposal.missing_fields,
                    repo_root,
                    snippet_files,
                    patch_rules,
                    max_chars_override=expand_budget,
                )
                if not expanded:
                    break
                new_fingerprint = _snippet_fingerprint(expanded)
                if new_fingerprint == last_fingerprint:
                    break
                last_fingerprint = new_fingerprint
                snippet_paths = [item.get("path") for item in expanded if item.get("path")]
                debug_allowed_files = snippet_paths or debug_allowed_files
                debug_snippets = expanded
                trace_events.append(
                    {
                        "event": "code_snippets_expanded",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "action_id": action.action_id,
                        "round": expand_round,
                        "snippets": [
                            {
                                "path": item.get("path"),
                                "tag": item.get("tag"),
                                "start_line": item.get("start_line"),
                                "end_line": item.get("end_line"),
                            }
                            for item in expanded
                        ],
                    }
                )
                debug_proposal = patch_debugger.repair(
                    action=action,
                    profile=profile,
                    patch_rules=patch_rules,
                    allowed_files=debug_allowed_files,
                    code_snippets=debug_snippets,
                    repo_root=repo_root,
                    patch_diff=patch_proposal.patch_diff,
                    build_log=reason_text,
                    feedback="review_failed",
                )
                if reporter:
                    note = reason_text
                    if debug_proposal and debug_proposal.missing_fields:
                        note = "; ".join(debug_proposal.missing_fields)
                    reporter.patch_debug(
                        action.action_id,
                        attempt + 1,
                        debug_proposal.status if debug_proposal else "NO_RESPONSE",
                        note,
                    )
                trace_events.append(
                    {
                        "event": "patch_debug_review_retry",
                        "agent": "PatchDebugAgent",
                        "iteration": iteration,
                        "action_id": action.action_id,
                        "attempt": attempt + 1,
                        "proposal": debug_proposal.model_dump() if debug_proposal else None,
                    }
                )
                if not debug_proposal or debug_proposal.status != "NEED_MORE_CONTEXT":
                    break
            if not debug_proposal or debug_proposal.status != "OK":
                continue
        if debug_proposal.status != "OK":
            continue
        patch_proposal = debug_proposal
        format_ok, format_reason = _check_patch_format(repo_root, patch_proposal.patch_diff)
        if not format_ok:
            continue
        det_ok, _, _ = review_patch_diff(patch_proposal.patch_diff, repo_root, patch_rules)
        if not det_ok:
            continue
        # Skip LLM review by default (deterministic review is sufficient).
        llm_review = None
        _skip_llm = patch_rules.get("skip_llm_review", True)
        if not _skip_llm:
            llm_review = patch_reviewer.review(
                patch_diff=patch_proposal.patch_diff,
                patch_rules=patch_rules,
                context={
                    "action": action.action_id,
                    "family": action.family,
                    "risk_level": action.risk_level,
                    "expected_effect": action.expected_effect,
                    "profile": profile.model_dump(),
                },
            )
        if llm_review:
            trace_events.append(
                {
                    "event": "patch_review_llm",
                    "agent": "PatchReviewAgent",
                    "iteration": iteration,
                    "action_id": action.action_id,
                    "review": llm_review.model_dump(),
                }
            )
            if llm_review.status != "OK" or llm_review.verdict != "PASS":
                continue
        return patch_proposal
    return None


def _build_memory_hints(
    experience_memory: ExperienceMemory,
    case_id: str,
    app: str,
    backend: Optional[str],
    limit: int = 6,
) -> List[Dict[str, object]]:
    if not experience_memory.config.enabled or not experience_memory.records:
        return []
    cfg = experience_memory.config
    hints: List[Dict[str, object]] = []
    for record in experience_memory.records:
        score = float(record.weight or 0.0)
        if backend and record.backend and backend != record.backend:
            score *= cfg.backend_mismatch_penalty
        if record.case_id and record.case_id == case_id:
            score *= cfg.case_match_boost
        if record.app:
            if record.app == app:
                score *= cfg.app_match_boost
            else:
                score *= cfg.app_mismatch_penalty
        if score == 0.0:
            continue
        hints.append(
            {
                "action_id": record.action_id,
                "family": record.family,
                "patch_family": record.patch_family,
                "target_file": record.target_file,
                "improvement_pct": record.improvement_pct,
                "strength": record.strength,
                "score": score,
            }
        )
    hints.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return hints[:limit]


def _count_occurrences(text: str, block: str) -> int:
    return len(re.findall(re.escape(block), text))


def _expand_snippets_for_anchor(
    repo_root: Path,
    file_path: str,
    anchor: str,
    max_snippets: int,
    max_chars: int,
) -> List[Dict[str, object]]:
    if max_snippets <= 0 or max_chars <= 0:
        return []
    path = (repo_root / file_path).resolve()
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    indices = [m.start() for m in re.finditer(re.escape(anchor), text)]
    if not indices:
        return []
    lines = text.splitlines()
    snippets: List[Dict[str, object]] = []
    budget_per = max(800, max_chars // max_snippets)
    seen = set()
    for idx in indices:
        line_no = text[:idx].count("\n")
        for block in _build_anchor_snippet_blocks(lines, line_no, budget_per):
            if len(snippets) >= max_snippets:
                break
            snippet = block["snippet"]
            if not snippet or snippet in seen:
                continue
            seen.add(snippet)
            anchor_hints = _anchor_hints_for_snippet(text, snippet)
            snippets.append(
                {
                    "path": file_path,
                    "snippet": snippet,
                    "tag": block.get("tag"),
                    "start_line": block.get("start_line"),
                    "end_line": block.get("end_line"),
                    "anchor_hints": anchor_hints,
                }
            )
        if len(snippets) >= max_snippets:
            break
    return snippets


def _build_anchor_snippet_blocks(
    lines: List[str],
    anchor_line: int,
    max_chars: int,
) -> List[Dict[str, object]]:
    blocks: List[Dict[str, object]] = []
    decl_start = max(0, anchor_line - 80)
    decl_end = min(len(lines), anchor_line + 10)
    snippet, start, end = _slice_line_block(lines, decl_start, decl_end, max_chars)
    if snippet:
        blocks.append({"tag": "anchor_declarations", "snippet": snippet, "start_line": start + 1, "end_line": end})
    ctx_start = max(0, anchor_line - 20)
    ctx_end = min(len(lines), anchor_line + 140)
    snippet, start, end = _slice_line_block(lines, ctx_start, ctx_end, max_chars)
    if snippet:
        blocks.append({"tag": "anchor_context", "snippet": snippet, "start_line": start + 1, "end_line": end})
    func_start = _find_function_start(lines, anchor_line)
    if func_start is not None:
        snippet, start, end = _extract_function_block(lines, func_start, max_chars)
        if snippet:
            blocks.append({"tag": "function_full", "snippet": snippet, "start_line": start + 1, "end_line": end})
        func_end = min(len(lines), func_start + 140)
        snippet, start, end = _slice_line_block(lines, func_start, func_end, max_chars)
        if snippet:
            blocks.append({"tag": "function_header", "snippet": snippet, "start_line": start + 1, "end_line": end})
    return blocks


def _parse_edit_failure_anchor(
    missing_fields: List[str],
) -> Optional[Tuple[str, str]]:
    for item in missing_fields:
        if not item.startswith("edit_apply_failed:"):
            continue
        payload = item.split("edit_apply_failed:", 1)[1].strip()
        parts = payload.split(":")
        if len(parts) < 4:
            continue
        error_kind = parts[0]
        if error_kind not in {"anchor_not_unique", "anchor_not_found", "old_text_not_found"}:
            continue
        file_path = parts[1]
        if parts[2] != "b64":
            continue
        encoded = ":".join(parts[3:])
        try:
            anchor = base64.b64decode(encoded.encode("ascii")).decode("utf-8")
        except Exception:
            continue
        return file_path, anchor
    return None


def _parse_need_more_context_targets(
    missing_fields: List[str],
    repo_root: Path,
) -> List[Tuple[str, Optional[str]]]:
    targets: List[Tuple[str, Optional[str]]] = []
    for item in missing_fields:
        if "NEED_MORE_CONTEXT" not in item:
            continue
        files: List[str] = []
        for match in re.finditer(r"file=([\\w./\\-*]+)", item):
            path = match.group(1).strip(" ,.;")
            files.append(path)
        for match in re.finditer(r"([\\w./-]+\\.(?:c|cc|cpp|h))", item):
            files.append(match.group(1))
        anchor = _guess_context_anchor(item)
        for path in files:
            if "*" in path:
                for cand in repo_root.glob(path):
                    if cand.is_file():
                        rel = cand.relative_to(repo_root).as_posix()
                        targets.append((rel, anchor))
            else:
                targets.append((path, anchor))
    deduped = []
    seen = set()
    for file_path, anchor in targets:
        key = (file_path, anchor)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((file_path, anchor))
    return deduped


def _expand_snippets_for_missing_context(
    missing_fields: Optional[List[str]],
    repo_root: Path,
    snippet_files: List[str],
    patch_rules: Dict[str, object],
    max_chars_override: Optional[int] = None,
) -> List[Dict[str, object]]:
    expanded: List[Dict[str, object]] = []
    if not missing_fields:
        return expanded
    max_snippets = int(patch_rules.get("max_snippets", 0) or 0)
    max_context = int(patch_rules.get("max_context_chars", 0) or 0)
    if max_chars_override is not None:
        expand_budget = max_chars_override
    else:
        expand_budget = max_context if max_context > 0 else 0
        if expand_budget:
            expand_budget = min(expand_budget + max_context, 240000)
    targets = _parse_need_more_context_targets(missing_fields, repo_root)
    edit_anchor = _parse_edit_failure_anchor(missing_fields)
    if edit_anchor:
        targets.append(edit_anchor)
    if not targets:
        for file_path in snippet_files:
            expanded.extend(
                _expand_snippets_for_file(
                    repo_root=repo_root,
                    file_path=file_path,
                    max_snippets=max_snippets,
                    max_chars=expand_budget,
                )
            )
    for file_path, anchor in targets:
        if anchor:
            expanded.extend(
                _expand_snippets_for_anchor(
                    repo_root=repo_root,
                    file_path=file_path,
                    anchor=anchor,
                    max_snippets=max_snippets,
                    max_chars=expand_budget,
                )
            )
        if not expanded:
            expanded.extend(
                _expand_snippets_for_file(
                    repo_root=repo_root,
                    file_path=file_path,
                    max_snippets=max_snippets,
                    max_chars=expand_budget,
                )
            )
    return expanded


def _snippet_fingerprint(snippets: List[Dict[str, object]]) -> Tuple[Tuple[str, int, int], ...]:
    items: List[Tuple[str, int, int]] = []
    for item in snippets:
        path = str(item.get("path") or "")
        start = int(item.get("start_line") or 0)
        end = int(item.get("end_line") or 0)
        items.append((path, start, end))
    return tuple(sorted(set(items)))


def _guess_context_anchor(text: str) -> Optional[str]:
    lowered = text.lower()
    if "#pragma omp" in lowered:
        return "#pragma omp"
    if "for (int jj" in lowered or "for(int jj" in lowered:
        return "for (int jj"
    if "for (int ii" in lowered or "for(int ii" in lowered:
        return "for (int ii"
    if "inner loop" in lowered and "neighbor" in lowered:
        return "for (jj"
    if "for (jj" in lowered or "for(jj" in lowered:
        return "for (jj"
    if "for (ii" in lowered or "for(ii" in lowered:
        return "for (ii"
    if "eval" in lowered:
        return "eval"
    if "compute_inner" in lowered:
        return "compute_inner"
    if "compute_middle" in lowered:
        return "compute_middle"
    if "compute_outer" in lowered:
        return "compute_outer"
    if "compute" in lowered:
        return "compute("
    return None


def _expand_snippets_for_file(
    repo_root: Path,
    file_path: str,
    max_snippets: int,
    max_chars: int,
) -> List[Dict[str, object]]:
    if max_snippets <= 0 or max_chars <= 0:
        return []
    path = (repo_root / file_path).resolve()
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = _build_snippet_blocks(text, max(800, max_chars // max_snippets))
    snippets: List[Dict[str, object]] = []
    seen = set()
    for block in blocks:
        if len(snippets) >= max_snippets:
            break
        snippet = block.get("snippet")
        if not snippet or snippet in seen:
            continue
        seen.add(snippet)
        snippets.append(
            {
                "path": file_path,
                "snippet": snippet,
                "tag": block.get("tag"),
                "start_line": block.get("start_line"),
                "end_line": block.get("end_line"),
                "anchor_hints": _anchor_hints_for_snippet(text, snippet),
            }
        )
    return snippets


def _parse_pair_style(input_text: str) -> Optional[List[str]]:
    for raw in input_text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        lower = line.lower()
        if not lower.startswith("pair_style"):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        style = parts[1]
        if style.startswith("hybrid"):
            return [token for token in parts[2:] if not _looks_numeric(token)]
        return [style]
    return None


def _looks_numeric(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _pair_style_files(
    input_text: str,
    repo_root: Path,
    run_args: Optional[List[str]] = None,
) -> List[str]:
    styles = _parse_pair_style(input_text)
    if not styles:
        return []
    files: List[str] = []
    use_omp = _run_args_enable_omp(run_args or [])
    use_opt = _run_args_enable_opt(run_args or [])
    for style in styles:
        cleaned = style.replace("/", "_").replace("-", "_")
        candidate = repo_root / "third_party" / "lammps" / "src" / f"pair_{cleaned}.cpp"
        if candidate.is_file():
            files.append(candidate.relative_to(repo_root).as_posix())
            if use_omp:
                omp_candidate = candidate.with_name(f"pair_{cleaned}_omp.cpp")
                if omp_candidate.is_file():
                    files.append(omp_candidate.relative_to(repo_root).as_posix())
                omp_subdir = (
                    repo_root
                    / "third_party"
                    / "lammps"
                    / "src"
                    / "OPENMP"
                    / f"pair_{cleaned}_omp.cpp"
                )
                if omp_subdir.is_file():
                    files.append(omp_subdir.relative_to(repo_root).as_posix())
            if use_opt:
                opt_subdir = (
                    repo_root
                    / "third_party"
                    / "lammps"
                    / "src"
                    / "OPT"
                    / f"pair_{cleaned}_opt.cpp"
                )
                if opt_subdir.is_file():
                    files.append(opt_subdir.relative_to(repo_root).as_posix())
            continue
        header = candidate.with_suffix(".h")
        if header.is_file():
            files.append(header.relative_to(repo_root).as_posix())
            if use_omp:
                omp_header = header.with_name(f"pair_{cleaned}_omp.h")
                if omp_header.is_file():
                    files.append(omp_header.relative_to(repo_root).as_posix())
                omp_header_subdir = (
                    repo_root
                    / "third_party"
                    / "lammps"
                    / "src"
                    / "OPENMP"
                    / f"pair_{cleaned}_omp.h"
                )
                if omp_header_subdir.is_file():
                    files.append(omp_header_subdir.relative_to(repo_root).as_posix())
            if use_opt:
                opt_header_subdir = (
                    repo_root
                    / "third_party"
                    / "lammps"
                    / "src"
                    / "OPT"
                    / f"pair_{cleaned}_opt.h"
                )
                if opt_header_subdir.is_file():
                    files.append(opt_header_subdir.relative_to(repo_root).as_posix())
    return files


def _expand_globs(repo_root: Path, patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        if not isinstance(pattern, str):
            continue
        for path in repo_root.glob(pattern):
            if path.is_file():
                files.append(path.relative_to(repo_root).as_posix())
    return files


def _variant_matches(
    when_cfg: Dict[str, object],
    run_args: List[str],
    input_text: str,
    env: Dict[str, str],
) -> bool:
    args_text = " ".join(str(arg) for arg in run_args)
    run_contains = when_cfg.get("run_args_contains", [])
    if isinstance(run_contains, list) and run_contains:
        matched = False
        for token in run_contains:
            if isinstance(token, str) and token in args_text:
                matched = True
                break
        if not matched:
            return False
    input_contains = when_cfg.get("input_contains", [])
    if isinstance(input_contains, list) and input_contains:
        matched = False
        for token in input_contains:
            if isinstance(token, str) and token in input_text:
                matched = True
                break
        if not matched:
            return False
    input_regex = when_cfg.get("input_regex", [])
    if isinstance(input_regex, list) and input_regex:
        matched = False
        for pattern in input_regex:
            if not isinstance(pattern, str):
                continue
            try:
                if re.search(pattern, input_text, flags=re.MULTILINE):
                    matched = True
                    break
            except re.error:
                continue
        if not matched:
            return False
    env_has = when_cfg.get("env_has", [])
    if isinstance(env_has, list) and env_has:
        for key in env_has:
            if not isinstance(key, str):
                continue
            if key not in env:
                return False
    env_equals = when_cfg.get("env_equals", {})
    if isinstance(env_equals, dict) and env_equals:
        for key, value in env_equals.items():
            if key not in env:
                return False
            if value is not None and str(env.get(key)) != str(value):
                return False
    return True


def _adapter_variant_files(
    adapter_cfg: Optional[Dict[str, object]],
    repo_root: Path,
    run_args: List[str],
    input_text: str,
    env: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    if not isinstance(adapter_cfg, dict):
        return [], []
    variants = adapter_cfg.get("variants")
    if not isinstance(variants, list):
        return [], []
    matched_ids: List[str] = []
    preferred: List[str] = []
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        when_cfg = variant.get("when")
        if not isinstance(when_cfg, dict):
            continue
        if not _variant_matches(when_cfg, run_args, input_text, env):
            continue
        matched_ids.append(str(variant.get("id", "variant")))
        globs = variant.get("prefer_globs", [])
        if isinstance(globs, list):
            preferred.extend(_expand_globs(repo_root, globs))
    return sorted(set(preferred)), matched_ids


def _run_args_enable_omp(run_args: List[str]) -> bool:
    args = [str(arg) for arg in run_args]
    for idx, arg in enumerate(args):
        if arg == "-sf" and idx + 1 < len(args) and args[idx + 1] == "omp":
            return True
        if arg == "-suffix" and idx + 1 < len(args) and args[idx + 1] == "omp":
            return True
        if arg == "-pk" and idx + 1 < len(args) and args[idx + 1] == "omp":
            return True
    return False


def _run_args_enable_opt(run_args: List[str]) -> bool:
    args = [str(arg) for arg in run_args]
    for idx, arg in enumerate(args):
        if arg == "-sf" and idx + 1 < len(args) and args[idx + 1] == "opt":
            return True
        if arg == "-suffix" and idx + 1 < len(args) and args[idx + 1] == "opt":
            return True
    return False


def _select_hotspot_categories(timing_breakdown: Optional[Dict[str, float]]) -> List[str]:
    if not timing_breakdown:
        return []
    ordered = sorted(
        timing_breakdown.items(),
        key=lambda item: item[1] if isinstance(item[1], (int, float)) else 0.0,
        reverse=True,
    )
    categories: List[str] = []
    for key, _value in ordered:
        category = _timing_key_to_hotspot(str(key))
        if category and category not in categories:
            categories.append(category)
        if len(categories) >= 2:
            break
    return categories


def _select_snippet_files(
    repo_root: Path,
    patch_rules: Dict[str, object],
    timing_breakdown: Optional[Dict[str, float]],
    fallback_files: List[str],
    preferred_files: List[str],
    max_snippets: int,
) -> List[str]:
    if max_snippets <= 0:
        return []
    hotspot_globs = patch_rules.get("hotspot_globs")
    hotspot_only = bool(patch_rules.get("hotspot_only", False))

    def _is_hotspot(path: str) -> bool:
        if not isinstance(hotspot_globs, dict):
            return False
        for patterns in hotspot_globs.values():
            if not isinstance(patterns, list):
                continue
            for pattern in patterns:
                if isinstance(pattern, str) and fnmatch.fnmatch(path, pattern):
                    return True
        return False
    if not isinstance(hotspot_globs, dict):
        return preferred_files or ([] if hotspot_only else fallback_files)
    files: List[str] = []
    for rel in preferred_files:
        if hotspot_only and not _is_hotspot(rel):
            continue
        if rel not in files:
            files.append(rel)
        if len(files) >= max_snippets:
            return files
    categories = _select_hotspot_categories(timing_breakdown)
    for category in categories:
        patterns = hotspot_globs.get(category, [])
        if not isinstance(patterns, list):
            continue
        for pattern in patterns:
            if not isinstance(pattern, str):
                continue
            for path in repo_root.glob(pattern):
                if not path.is_file():
                    continue
                rel = path.relative_to(repo_root).as_posix()
                if rel not in files:
                    files.append(rel)
            if len(files) >= max_snippets:
                break
        if len(files) >= max_snippets:
            break
    if files:
        return files
    if hotspot_only:
        return []
    return preferred_files or fallback_files


def _target_with_related_files(
    repo_root: Path,
    target_file: str,
    candidate_files: List[str],
    max_related: int = 8,
) -> List[str]:
    related: List[str] = []
    if not target_file:
        return related
    allowed = set(candidate_files or [])
    if target_file in allowed:
        related.append(target_file)
    else:
        related.append(target_file)
    if max_related <= 1:
        return related[:1]

    repo_root_resolved = repo_root.resolve()
    target_path = (repo_root / target_file).resolve()

    def _add(rel_path: str) -> None:
        if rel_path in related:
            return
        if allowed and rel_path not in allowed:
            return
        related.append(rel_path)

    if target_file.endswith(".c"):
        _add(target_file[:-2] + ".h")

    if target_path.is_file():
        try:
            include_text = target_path.read_text(encoding="utf-8", errors="replace")
            include_matches = re.findall(
                r'^\s*#\s*include\s*"([^"]+)"',
                include_text,
                flags=re.MULTILINE,
            )
            for inc in include_matches:
                inc_path = (target_path.parent / inc).resolve()
                try:
                    rel = inc_path.relative_to(repo_root_resolved).as_posix()
                except ValueError:
                    continue
                _add(rel)
                if len(related) >= max_related:
                    break
        except Exception:
            pass
    return related[:max_related]


def _available_hotspots_for_files(
    snippet_files: List[str],
    patch_rules: Dict[str, object],
) -> set[str]:
    available: set[str] = set()
    hotspot_globs = patch_rules.get("hotspot_globs")
    if not isinstance(hotspot_globs, dict):
        return available
    for category, patterns in hotspot_globs.items():
        if not isinstance(patterns, list):
            continue
        for rel in snippet_files:
            if category in available:
                break
            for pattern in patterns:
                if not isinstance(pattern, str):
                    continue
                if fnmatch.fnmatch(rel, pattern):
                    available.add(category)
                    break
    return available


def _filter_source_patch_by_hotspot(
    actions: List[ActionIR],
    patch_families: Optional[Dict[str, object]],
    available_hotspots: set[str],
) -> List[ActionIR]:
    if not actions or not patch_families:
        return actions
    families = patch_families.get("families")
    if not isinstance(families, list):
        return actions
    family_map: Dict[str, Dict[str, object]] = {}
    for entry in families:
        if isinstance(entry, dict) and "id" in entry:
            family_map[str(entry["id"])] = entry
    filtered: List[ActionIR] = []
    for action in actions:
        if action.family != "source_patch":
            filtered.append(action)
            continue
        patch_family = (action.parameters or {}).get("patch_family")
        if not isinstance(patch_family, str):
            filtered.append(action)
            continue
        meta = family_map.get(patch_family, {})
        required = set(meta.get("requires_hotspot") or [])
        if required and not (required & available_hotspots):
            continue
        filtered.append(action)
    return filtered


def _quick_diff_stats(patch_text: str) -> Optional[Dict[str, object]]:
    """Extract quick file/line stats from a unified diff for console display."""
    if not patch_text:
        return None
    files: List[str] = []
    lines_changed = 0
    for line in patch_text.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            path = line[4:].split("\t", 1)[0].strip()
            if path != "/dev/null":
                if path.startswith(("a/", "b/")):
                    path = path[2:]
                if path and path not in files:
                    files.append(path)
        elif line.startswith("+") or line.startswith("-"):
            if not line.startswith(("+++", "---")):
                lines_changed += 1
    if not files and lines_changed == 0:
        return None
    return {"files": files, "lines_changed": lines_changed}


def _check_patch_format(repo_root: Path, patch_text: str) -> Tuple[bool, str]:
    if not patch_text.strip():
        return False, "empty patch"
    result = subprocess.run(
        ["git", "-C", str(repo_root), "apply", "--check", "-"],
        input=patch_text,
        text=True,
        capture_output=True,
    )
    if result.returncode == 0:
        return True, ""
    stderr = (result.stderr or "").strip()
    if "corrupt patch" in stderr:
        return False, stderr
    return True, ""


def _check_patch_apply(
    repo_root: Path,
    patch_text: str,
    patch_root: Optional[Path],
) -> Tuple[bool, str]:
    check_root = repo_root
    adjusted = patch_text
    if patch_root:
        prefix = patch_root.as_posix().rstrip("/") + "/"
        adjusted = _strip_patch_prefix(patch_text, prefix)
        check_root = (repo_root / patch_root).resolve()
    result = subprocess.run(
        ["git", "-C", str(check_root), "apply", "--check", "-"],
        input=adjusted,
        text=True,
        capture_output=True,
    )
    if result.returncode == 0:
        return True, ""
    stderr = (result.stderr or result.stdout or "").strip()
    return False, stderr or "patch does not apply"


def _compose_patch_from_debug_delta(
    worktree_root: Path,
    debug_patch_diff: str,
    patch_root: Optional[Path],
) -> Tuple[bool, str]:
    if not debug_patch_diff.strip():
        return False, "empty debug patch diff"
    compose_root = worktree_root
    adjusted = debug_patch_diff
    if patch_root:
        compose_root = (worktree_root / patch_root).resolve()
        try:
            compose_root.relative_to(worktree_root.resolve())
        except ValueError:
            return False, f"patch_root escapes worktree: {patch_root}"
        prefix = patch_root.as_posix().rstrip("/") + "/"
        adjusted = _strip_patch_prefix(debug_patch_diff, prefix)
    apply_result = subprocess.run(
        ["git", "-C", str(compose_root), "apply", "-"],
        input=adjusted,
        text=True,
        capture_output=True,
    )
    if apply_result.returncode != 0:
        stderr = (apply_result.stderr or apply_result.stdout or "").strip()
        return False, stderr or "debug patch delta does not apply in debug context"
    diff_result = subprocess.run(
        ["git", "-C", str(compose_root), "diff", "--binary"],
        text=True,
        capture_output=True,
    )
    if diff_result.returncode != 0:
        stderr = (diff_result.stderr or diff_result.stdout or "").strip()
        return False, stderr or "failed to compose debug patch"
    composed = diff_result.stdout
    if not composed.strip():
        return False, "no changes after composing debug patch"
    return True, composed


def _summarize_build_log(log_path: Path, max_lines: int = 120, max_chars: int = 4000) -> str:
    if not log_path.exists():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    error_lines = [
        line
        for line in lines
        if "error:" in line
        or "fatal error" in line
        or "undefined reference" in line
        or "no such file or directory" in line
    ]
    if error_lines:
        lines = error_lines[-max_lines:]
    else:
        lines = lines[-max_lines:]
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def _preflight_patch_compile(
    repo_root: Path,
    exp_id: str,
    patch_path: Path,
    patch_root: Optional[Path],
    input_script: Path,
    allowlist: list[str],
    build_cfg: Dict[str, object],
    build_packs: Optional[Dict[str, object]],
    action: ActionIR,
    run_dir: Path,
    worktree_retries: int = 2,
) -> Tuple[bool, str, Optional[str]]:
    preflight_dir = run_dir / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    build_log_path = preflight_dir / "build.log"
    try:
        with GitPatchContext(
            repo_root=repo_root,
            exp_id=f"{exp_id}-preflight",
            artifacts_dir=preflight_dir,
            input_script=input_script,
            input_edit=None,
            allowlist=allowlist,
            patch_path=patch_path,
            patch_root=patch_root,
            worktree_retries=worktree_retries,
        ) as ctx:
            source_root = ctx.worktree_dir
            final_build_cfg = _apply_build_config(build_cfg, action, build_packs)
            build_job(final_build_cfg, source_root, preflight_dir)
        return True, "", str(build_log_path)
    except Exception as exc:
        log_summary = _summarize_build_log(build_log_path)
        reason = log_summary or str(exc)
        return False, reason, str(build_log_path)


def _phase_order(hierarchical_cfg: Optional[Dict[str, object]]) -> List[str]:
    phases_cfg = hierarchical_cfg.get("phases", {}) if isinstance(hierarchical_cfg, dict) else {}
    order = phases_cfg.get("default_order")
    if isinstance(order, list) and all(isinstance(item, str) for item in order):
        return [str(item) for item in order]
    return ["RUN_TUNE", "BUILD_TUNE", "PATCH"]


def _next_phase_in_order(
    current: str,
    order: List[str],
    has_build: bool,
    has_patch: bool,
) -> str:
    if current not in order:
        return current
    idx = order.index(current) + 1
    while idx < len(order):
        candidate = order[idx]
        if candidate == "BUILD_TUNE" and not has_build:
            idx += 1
            continue
        if candidate == "PATCH" and not has_patch:
            idx += 1
            continue
        return candidate
    return current


def _freeze_thresholds(
    hierarchical_cfg: Optional[Dict[str, object]],
    phase: str,
) -> Dict[str, object]:
    freeze_cfg = hierarchical_cfg.get("freeze", {}) if isinstance(hierarchical_cfg, dict) else {}
    if phase == "RUN_TUNE":
        return freeze_cfg.get("run_config", {}) if isinstance(freeze_cfg, dict) else {}
    if phase == "BUILD_TUNE":
        return freeze_cfg.get("build_config", {}) if isinstance(freeze_cfg, dict) else {}
    return {}


def _phase_freeze_decision(
    phase: str,
    baseline_exp: ExperimentIR,
    best_exp: Optional[ExperimentIR],
    thresholds: Dict[str, object],
    state: StopState,
) -> Tuple[bool, str]:
    if not best_exp:
        return False, ""
    baseline = baseline_exp.results.runtime_seconds
    best = best_exp.results.runtime_seconds
    if baseline <= 0 or best <= 0:
        return False, ""
    min_gain = float(thresholds.get("min_relative_gain", 0.0) or 0.0)
    max_cv = thresholds.get("max_variance_cv")
    patience = int(thresholds.get("patience_rounds", 0) or 0)
    improvement = (baseline - best) / baseline
    variance_cv = _extract_variance_cv(best_exp)
    if improvement >= min_gain and (variance_cv is None or variance_cv <= float(max_cv)):
        return True, f"{phase} freeze: gain {improvement:.3f} meets threshold"
    if patience and state.no_improve_iters >= patience:
        return True, f"{phase} freeze: stagnation for {state.no_improve_iters} rounds"
    return False, ""


def _filter_actions_by_targets(actions: List[ActionIR], targets: set[str]) -> List[ActionIR]:
    return [action for action in actions if any(t in targets for t in action.applies_to)]


def _family_targets(actions: List[ActionIR]) -> Dict[str, set[str]]:
    mapping: Dict[str, set[str]] = {}
    for action in actions:
        fam = action.family
        if fam not in mapping:
            mapping[fam] = set()
        for target in action.applies_to or []:
            mapping[fam].add(target)
    return mapping


def _select_with_evidence(
    ranked_items: List[RankedAction],
    limit: int,
    max_explore_frac: float = 0.2,
) -> List[RankedAction]:
    if limit <= 0:
        return []
    max_explore = max(1, int(round(limit * max_explore_frac)))
    selected: List[RankedAction] = []
    explore_count = 0
    for item in ranked_items:
        evidence_score = 1.0
        if isinstance(item.score_breakdown, dict):
            evidence_score = float(item.score_breakdown.get("evidence_score", 1.0) or 1.0)
        if evidence_score < 0.5:
            if explore_count >= max_explore:
                continue
            explore_count += 1
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def _has_actions_for_target(actions: List[ActionIR], target: str) -> bool:
    return any(target in action.applies_to for action in actions)


def _system_caps(repo_root: Path) -> Dict[str, object]:
    return collect_system_caps(repo_root)


def _best_improvement_pct(
    baseline_exp: Optional[ExperimentIR],
    best_exp: Optional[ExperimentIR],
) -> float:
    if not baseline_exp or not best_exp:
        return 0.0
    base = baseline_exp.results.runtime_seconds
    best = best_exp.results.runtime_seconds
    if base <= 0.0 or best <= 0.0:
        return 0.0
    return (base - best) / base


def _should_refine_on_best(
    baseline_exp: Optional[ExperimentIR],
    best_exp: Optional[ExperimentIR],
    enabled: bool,
    min_improvement_pct: float,
) -> bool:
    if not enabled or not best_exp or not best_exp.action:
        return False
    return _best_improvement_pct(baseline_exp, best_exp) >= min_improvement_pct


def _prepare_actions(
    base_actions: List[ActionIR],
    candidate_policy: Optional[Dict[str, object]],
    system_caps: Dict[str, object],
    experiments: List[ExperimentIR],
    adapter_cfg: Optional[Dict[str, object]],
    job: JobIR,
    fixed_threads: Optional[int] = None,
) -> List[ActionIR]:
    selected_policy = _select_candidate_policy(candidate_policy, job.case_id)
    actions = _expand_dynamic_actions(
        base_actions,
        selected_policy,
        system_caps,
        experiments,
        fixed_threads=fixed_threads,
    )
    actions = _filter_fixed_threads(actions, fixed_threads)
    actions = _apply_adapter(actions, adapter_cfg, job)
    actions = _filter_mpi_actions(actions, job)
    return actions


def _select_candidate_policy(
    candidate_policy: Optional[Dict[str, object]],
    case_id: str,
) -> Dict[str, object]:
    if not isinstance(candidate_policy, dict):
        return {}
    cases = candidate_policy.get("cases", {})
    base: Dict[str, object] = {
        key: value
        for key, value in candidate_policy.items()
        if key not in {"cases"}
    }
    default = base.get("default")
    if isinstance(default, dict):
        base = _deep_merge_dicts(base, default)
    case_cfg = cases.get(case_id) if isinstance(cases, dict) else None
    if isinstance(case_cfg, dict):
        base = _deep_merge_dicts(base, case_cfg)
    return base


def _deep_merge_dicts(base: Dict[str, object], overlay: Dict[str, object]) -> Dict[str, object]:
    merged: Dict[str, object] = dict(base)
    for key, value in overlay.items():
        if key == "default":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


def _expand_dynamic_actions(
    actions: List[ActionIR],
    candidate_policy: Optional[Dict[str, object]],
    system_caps: Dict[str, object],
    experiments: List[ExperimentIR],
    fixed_threads: Optional[int] = None,
) -> List[ActionIR]:
    policy = candidate_policy or {}
    families_cfg = policy.get("families", {}) if isinstance(policy, dict) else {}
    threads_cfg = families_cfg.get("parallel_omp", {}).get("threads", {})
    thread_candidates = _thread_candidates(
        threads_cfg,
        system_caps,
        experiments,
        fixed_threads=fixed_threads,
    )
    allowed_templates = _allowed_templates(threads_cfg)
    if not thread_candidates:
        return [action for action in actions if not action.parameters.get("dynamic_threads")]
    expanded: List[ActionIR] = []
    for action in actions:
        if action.parameters.get("dynamic_threads"):
            template_id = action.parameters.get("template_id", "template")
            if allowed_templates and template_id not in allowed_templates:
                continue
            for threads in thread_candidates:
                expanded.append(_materialize_thread_action(action, template_id, threads))
        else:
            expanded.append(action)
    return expanded


def _thread_candidates(
    cfg: Dict[str, object],
    system_caps: Dict[str, object],
    experiments: List[ExperimentIR],
    fixed_threads: Optional[int] = None,
) -> List[int]:
    if fixed_threads is not None:
        return [int(fixed_threads)]
    if not cfg:
        return []
    stop_on_peak = bool(cfg.get("stop_on_peak", False))
    peak_min_improvement = float(cfg.get("peak_min_improvement_pct", 0.01))
    peak_min_samples = int(cfg.get("peak_min_samples", 3))
    if stop_on_peak:
        peak_threads = _detect_thread_peak(
            experiments,
            min_improvement_pct=peak_min_improvement,
            min_samples=peak_min_samples,
        )
        if peak_threads is not None:
            return [peak_threads]
    mode = str(cfg.get("mode", "powers_of_two"))
    explicit_values = cfg.get("values") if mode == "explicit" else None
    min_threads = int(cfg.get("min", 1))
    max_raw = cfg.get("max", "physical")
    max_threads = _resolve_threads_cap(max_raw, system_caps)
    include_physical = bool(cfg.get("include_physical", True))
    include_logical = bool(cfg.get("include_logical", False))
    include_core_groups = bool(cfg.get("include_core_groups", False))
    include_socket_cores = bool(cfg.get("include_socket_cores", False))
    include_numa_cores = bool(cfg.get("include_numa_cores", False))
    max_pool = int(cfg.get("max_pool", 6))
    keep_bottom = int(cfg.get("keep_bottom_k", 0))
    keep_top = int(cfg.get("keep_top_k", 0))
    fractions = _parse_fractions(cfg.get("include_fractions"))
    candidates = _generate_threads(mode, min_threads, max_threads, explicit_values)
    if include_physical:
        candidates.append(_get_int(system_caps.get("physical_cores"), max_threads))
    if include_logical:
        candidates.append(_get_int(system_caps.get("logical_cores"), max_threads))
    if include_socket_cores:
        socket_cores = _get_int(system_caps.get("cores_per_socket"), 0)
        if socket_cores:
            candidates.append(socket_cores)
    if include_numa_cores:
        numa_nodes = _get_int(system_caps.get("numa_nodes"), 0)
        physical = _get_int(system_caps.get("physical_cores"), 0)
        if numa_nodes and physical:
            per_numa = max(1, int(round(physical / float(numa_nodes))))
            candidates.append(per_numa)
    if include_core_groups:
        core_groups = system_caps.get("core_groups")
        if isinstance(core_groups, list):
            for group in core_groups:
                if isinstance(group, dict):
                    count = _get_int(group.get("count"), 0)
                    if count:
                        candidates.append(count)
    if fractions:
        for frac in fractions:
            guess = int(round(max_threads * frac))
            if min_threads <= guess <= max_threads:
                candidates.append(guess)
    candidates = sorted({c for c in candidates if c >= min_threads})
    candidates = _trim_candidates(candidates, max_pool, keep_bottom, keep_top)
    if cfg.get("expand_if_best_at_max", False):
        best_threads = _best_threads(experiments)
        if best_threads and candidates:
            max_candidate = max(candidates)
            if best_threads >= max_candidate and max_candidate < max_threads:
                step = int(cfg.get("expansion_step", 0)) or 0
                next_threads = max_candidate + step if step else min(max_threads, max_candidate * 2)
                next_threads = min(max_threads, next_threads)
                candidates = sorted(set(candidates + [next_threads]))
                candidates = _trim_candidates(candidates, max_pool, keep_bottom, keep_top)
    return candidates


def _detect_thread_peak(
    experiments: List[ExperimentIR],
    min_improvement_pct: float,
    min_samples: int,
    family: str = "parallel_omp",
) -> Optional[int]:
    perf: Dict[int, float] = {}
    for exp in experiments:
        if not exp.action or exp.action.family != family:
            continue
        if exp.verdict != "PASS":
            continue
        env = exp.job.env or {}
        threads_raw = env.get("OMP_NUM_THREADS")
        if threads_raw is None:
            continue
        try:
            threads = int(threads_raw)
        except (TypeError, ValueError):
            continue
        runtime = exp.results.runtime_seconds
        if runtime <= 0:
            continue
        best = perf.get(threads)
        if best is None or runtime < best:
            perf[threads] = runtime
    if len(perf) < max(3, min_samples):
        return None
    ordered = sorted(perf.items(), key=lambda item: item[0])
    threads_sorted = [t for t, _ in ordered]
    best_threads = min(perf, key=perf.get)
    if best_threads == threads_sorted[0] or best_threads == threads_sorted[-1]:
        return None
    lower = max(t for t in threads_sorted if t < best_threads)
    higher = min(t for t in threads_sorted if t > best_threads)
    best_time = perf[best_threads]
    lower_time = perf.get(lower)
    higher_time = perf.get(higher)
    if lower_time is None or higher_time is None:
        return None
    lower_improve = (lower_time - best_time) / max(lower_time, 1.0e-12)
    higher_improve = (higher_time - best_time) / max(higher_time, 1.0e-12)
    if lower_improve >= min_improvement_pct and higher_improve >= min_improvement_pct:
        return best_threads
    return None


def _resolve_threads_cap(max_raw: object, system_caps: Dict[str, object]) -> int:
    if isinstance(max_raw, str):
        if max_raw == "physical":
            return _get_int(system_caps.get("physical_cores"), 1)
        if max_raw == "logical":
            return _get_int(system_caps.get("logical_cores"), 1)
        try:
            return int(max_raw)
        except ValueError:
            return _get_int(system_caps.get("physical_cores"), 1)
    if isinstance(max_raw, (int, float)):
        return int(max_raw)
    return _get_int(system_caps.get("physical_cores"), 1)


def _generate_threads(
    mode: str,
    min_threads: int,
    max_threads: int,
    explicit_values: Optional[object],
) -> List[int]:
    if max_threads < min_threads:
        return [min_threads]
    if mode == "explicit":
        if isinstance(explicit_values, list):
            values = []
            for item in explicit_values:
                try:
                    values.append(int(item))
                except (TypeError, ValueError):
                    continue
            return values
        return [min_threads, max_threads] if min_threads != max_threads else [min_threads]
    if mode == "linear":
        return list(range(min_threads, max_threads + 1))
    if mode == "powers_of_two":
        value = 1
        candidates: List[int] = []
        while value <= max_threads:
            if value >= min_threads:
                candidates.append(value)
            value *= 2
        if min_threads not in candidates:
            candidates.append(min_threads)
        return candidates
    return [min_threads, max_threads] if min_threads != max_threads else [min_threads]


def _trim_candidates_basic(candidates: List[int], max_pool: int) -> List[int]:
    if max_pool <= 0 or len(candidates) <= max_pool:
        return candidates
    if max_pool == 1:
        return [candidates[-1]]
    if max_pool == 2:
        return [candidates[0], candidates[-1]]
    trimmed = [candidates[0], candidates[-1]]
    step = (len(candidates) - 1) / float(max_pool - 1)
    for idx in range(1, max_pool - 1):
        pick = int(round(idx * step))
        trimmed.append(candidates[pick])
    return sorted(set(trimmed))


def _trim_candidates(
    candidates: List[int],
    max_pool: int,
    keep_bottom: int = 0,
    keep_top: int = 0,
) -> List[int]:
    if max_pool <= 0 or len(candidates) <= max_pool:
        return candidates
    keep_bottom = max(0, keep_bottom)
    keep_top = max(0, keep_top)
    if keep_bottom == 0 and keep_top == 0:
        return _trim_candidates_basic(candidates, max_pool)
    ordered = sorted(set(candidates))
    if keep_bottom >= len(ordered):
        return ordered[:max_pool]
    max_kept = min(max_pool, len(ordered))
    if keep_bottom + keep_top > max_kept:
        keep_top = max(0, max_kept - keep_bottom)
    kept = ordered[:keep_bottom]
    if keep_top:
        kept.extend(ordered[-keep_top:])
    remaining_budget = max_pool - len(kept)
    if remaining_budget <= 0:
        return sorted(set(kept))[:max_pool]
    remaining = [c for c in ordered if c not in kept]
    if remaining:
        kept.extend(_trim_candidates_basic(remaining, remaining_budget))
    return sorted(set(kept))


def _best_threads(experiments: List[ExperimentIR]) -> Optional[int]:
    best = _best_pass_exp(experiments)
    if not best or not best.job:
        return None
    env = best.job.env or {}
    value = env.get("OMP_NUM_THREADS")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _allowed_templates(cfg: Dict[str, object]) -> Optional[set[str]]:
    template_ids = cfg.get("template_ids") if isinstance(cfg, dict) else None
    if not template_ids:
        return None
    if isinstance(template_ids, list):
        return {str(item) for item in template_ids if str(item).strip()}
    return None


def _parse_fractions(value: object) -> List[float]:
    if not isinstance(value, list):
        return []
    fractions: List[float] = []
    for item in value:
        try:
            frac = float(item)
        except (TypeError, ValueError):
            continue
        if 0.0 < frac <= 1.0:
            fractions.append(frac)
    return sorted(set(fractions))


def _get_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _materialize_thread_action(
    template_action: ActionIR, template_id: str, threads: int
) -> ActionIR:
    action = template_action.model_copy(deep=True)
    action.action_id = f"{template_action.family}.t{threads}_{template_id}"
    action.description = _format_thread_description(template_action.description, threads)
    params = dict(action.parameters or {})
    env = dict(params.get("env", {}) or {})
    env["OMP_NUM_THREADS"] = str(threads)
    env = _replace_thread_placeholders(env, threads)
    params["env"] = env
    backend_threads = params.get("backend_threads")
    if isinstance(backend_threads, str) and "{threads}" in backend_threads:
        params["backend_threads"] = backend_threads.replace("{threads}", str(threads))
    run_args_cfg = _replace_thread_placeholders_in_run_args(params.get("run_args"), threads)
    if run_args_cfg:
        params["run_args"] = run_args_cfg
    params["dynamic_threads"] = False
    action.parameters = params
    return action


def _replace_thread_placeholders(env: Dict[str, str], threads: int) -> Dict[str, str]:
    replaced: Dict[str, str] = {}
    for key, value in env.items():
        if isinstance(value, str) and "{threads}" in value:
            replaced[key] = value.replace("{threads}", str(threads))
        else:
            replaced[key] = value
    return replaced


def _replace_thread_placeholders_in_run_args(
    run_args_cfg: object, threads: int
) -> Dict[str, object] | None:
    if not isinstance(run_args_cfg, dict):
        return None
    replaced: Dict[str, object] = dict(run_args_cfg)
    set_flags = []
    for entry in run_args_cfg.get("set_flags", []) or []:
        if not isinstance(entry, dict):
            continue
        values = entry.get("values", [])
        if isinstance(values, list):
            new_values = []
            for val in values:
                if isinstance(val, str) and "{threads}" in val:
                    new_values.append(val.replace("{threads}", str(threads)))
                else:
                    new_values.append(val)
            entry = dict(entry)
            entry["values"] = new_values
        set_flags.append(entry)
    if set_flags:
        replaced["set_flags"] = set_flags
    return replaced


def _format_thread_description(description: str, threads: int) -> str:
    if not description:
        return f"OMP={threads}"
    return description.replace("{threads}", str(threads))


def _apply_adapter(
    actions: List[ActionIR],
    adapter_cfg: Optional[Dict[str, object]],
    job: JobIR,
) -> List[ActionIR]:
    if not adapter_cfg or adapter_cfg.get("app") not in {job.app}:
        return actions
    adapted: List[ActionIR] = []
    for action in actions:
        adapted.append(apply_app_adapter(action, job, adapter_cfg))
    return adapted


_MPI_FAMILIES = {"parallel_mpi", "mpi_omp_hybrid", "comm_tune"}


def _has_launcher(action: ActionIR) -> bool:
    """Return True if the action specifies a non-direct MPI launcher."""
    params = action.parameters or {}
    launcher = params.get("launcher")
    if isinstance(launcher, dict) and launcher.get("type", "direct") != "direct":
        return True
    return False


def _filter_mpi_actions(
    actions: List[ActionIR],
    job: JobIR,
    reporter: Optional["ConsoleUI"] = None,
) -> List[ActionIR]:
    """Remove actions that require MPI when the binary lacks real MPI support."""
    from skills.hardware_probe import check_binary_mpi_support

    # Only check once; cache result
    has_mpi = check_binary_mpi_support(job.app_bin)
    if has_mpi:
        return actions

    filtered: List[ActionIR] = []
    dropped = 0
    for action in actions:
        if _has_launcher(action):
            dropped += 1
            continue
        filtered.append(action)
    if dropped and reporter:
        reporter._print(
            f"  [MPI guard] Dropped {dropped} action(s): binary has no MPI support"
        )
    return filtered


def _filter_fixed_threads(
    actions: List[ActionIR],
    fixed_threads: Optional[int],
) -> List[ActionIR]:
    if fixed_threads is None:
        return actions
    fixed_str = str(int(fixed_threads))
    filtered: List[ActionIR] = []
    for action in actions:
        params = action.parameters or {}
        env = params.get("env", {})
        if isinstance(env, dict) and "OMP_NUM_THREADS" in env:
            if str(env.get("OMP_NUM_THREADS")) != fixed_str:
                continue
        backend_threads = params.get("backend_threads")
        if backend_threads is not None and str(backend_threads) != fixed_str:
            continue
        filtered.append(action)
    return filtered


def _merge_run_args(base: Optional[Dict[str, object]], extra: Optional[Dict[str, object]]) -> Dict[str, object]:
    merged: Dict[str, object] = {}
    if isinstance(base, dict):
        merged.update(base)
    if not isinstance(extra, dict):
        return merged
    set_flags = list(merged.get("set_flags", []))
    seen = {item.get("flag") for item in set_flags if isinstance(item, dict)}
    for item in extra.get("set_flags", []):
        if not isinstance(item, dict):
            continue
        flag = item.get("flag")
        if flag in seen:
            continue
        set_flags.append(item)
        seen.add(flag)
    if set_flags:
        merged["set_flags"] = set_flags
    return merged


def _select_validation_base(
    baseline_exp: ExperimentIR,
    best_exp: ExperimentIR,
    experiments: List[ExperimentIR],
) -> tuple[JobIR, Optional[str], Optional[str]]:
    if best_exp and best_exp.job:
        best_action_id = best_exp.action.action_id if best_exp.action else "baseline"
        return best_exp.job, best_exp.run_id, best_action_id
    if best_exp.base_run_id:
        base_exp = _find_experiment_by_run_id(experiments, best_exp.base_run_id)
        if base_exp:
            base_action_id = base_exp.action.action_id if base_exp.action else "baseline"
            return base_exp.job, base_exp.run_id, base_action_id
    return baseline_exp.job, baseline_exp.run_id, "baseline"


def _find_experiment_by_run_id(
    experiments: List[ExperimentIR], run_id: str
) -> Optional[ExperimentIR]:
    for exp in experiments:
        if exp.run_id == run_id:
            return exp
    return None


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


class TraceBuffer(list):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path

    def append(self, item) -> None:
        super().append(item)
        try:
            self._path.write_text(json.dumps(self, indent=2), encoding="utf-8")
        except OSError:
            pass


@dataclass
class DeepAnalysisInitOutput:
    opportunities: List[ActionIR]
    deep_analysis_result: Optional[DeepCodeAnalysisResult]
    opportunity_graph: Optional[OpportunityGraph]
    selected_opportunities: Optional[SelectedOpportunities]


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
    direction_space: Optional[List[Dict[str, object]]],
    llm_client: Optional[LLMClient],
    profiling_cfg: Optional[Dict[str, object]] = None,
    wrappers_cfg: Optional[List[Dict[str, object]]] = None,
    candidate_policy: Optional[Dict[str, object]] = None,
    build_packs: Optional[Dict[str, object]] = None,
    patch_families: Optional[Dict[str, object]] = None,
    survey_guidance: Optional[Dict[str, object]] = None,
    hierarchical_cfg: Optional[Dict[str, object]] = None,
    adapter_cfg: Optional[Dict[str, object]] = None,
    planner_cfg: Optional[Dict[str, object]] = None,
    reporter: Optional[ConsoleUI] = None,
    build_cfg: Optional[Dict[str, object]] = None,
    baseline_repeats: int = 1,
    baseline_stat: str = "mean",
    validate_top1_repeats: int = 0,
    min_improvement_pct: float = 0.0,
    resume_state: Optional[Dict[str, object]] = None,
    fixed_threads: Optional[int] = None,
    skip_baseline: bool = False,
) -> Dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    profiler = ProfilerAgent()
    planner = PlannerAgent(planner_cfg or {}, llm_client)
    optimizer = OptimizerAgent(llm_client)
    ranker = RouterRankerAgent(llm_client)
    verifier = VerifierAgent(llm_client)
    reviewer = ReviewerAgent(llm_client)
    orchestrator_agent = None
    orchestrator_cfg = (planner_cfg or {}).get("orchestrator_agent", {})
    from orchestrator.agents.patch_planner import PatchPlannerAgent
    patch_planner = PatchPlannerAgent(llm_client)
    # Legacy agents kept for non-patch phases; will be removed in Phase 10.
    idea_agent = IdeaAgent(llm_client)
    code_survey_agent = CodeSurveyAgent(llm_client)
    action_synth = ActionSynthAgent(llm_client)
    # Check if agentic code patching is enabled
    agentic_cfg = (planner_cfg or {}).get("agentic_code_patch", {})
    use_agentic_patcher = isinstance(agentic_cfg, dict) and agentic_cfg.get("enabled", False)
    if use_agentic_patcher:
        # Use the new agentic code optimizer with tool use and multi-turn conversation
        # Keep agent repo_root at orchestration root so all paths remain
        # consistently repo-relative across action params, allowed_files, and hints.
        agentic_repo_root = repo_root
        _agentic_src_raw = str((build_cfg or {}).get("source_dir", "") or "")
        if _agentic_src_raw:
            _agentic_src_p = Path(_agentic_src_raw)
            agentic_build_dir = _agentic_src_p if _agentic_src_p.is_absolute() else (repo_root / _agentic_src_p)
        else:
            agentic_build_dir = agentic_repo_root
        llm_config = {
            "api_key_env": agentic_cfg.get("api_key_env", "DEEPSEEK_API_KEY"),
            "base_url": agentic_cfg.get("base_url", "https://api.deepseek.com"),
            "model": agentic_cfg.get("model", "deepseek-chat"),
            "enabled": True,
            "max_turns": agentic_cfg.get("max_turns", 25),
            "max_tool_calls_per_turn": agentic_cfg.get("max_tool_calls_per_turn", 5),
            "max_invalid_tool_calls_total": agentic_cfg.get("max_invalid_tool_calls_total", 5),
            "max_invalid_tool_calls_per_tool": agentic_cfg.get("max_invalid_tool_calls_per_tool", 2),
        }
        code_patcher = create_agentic_code_patch_agent(
            repo_root=agentic_repo_root,
            build_dir=agentic_build_dir,
            llm_config=llm_config,
        )
    else:
        code_patcher = CodePatchAgent(llm_client)
    patch_debugger = PatchDebugAgent(llm_client)
    patch_reviewer = PatchReviewAgent(llm_client)
    executor = ExecutorAgent(_run_experiment)
    reporter_agent = ReporterAgent()
    triage = TriageAgent()
    memory = OptimizationMemory()
    memory_cfg = (planner_cfg or {}).get("memory", {})
    knowledge_root = (artifacts_dir.parent if artifacts_dir else Path("artifacts")) / "knowledge"
    correctness_contracts_path = knowledge_root / "correctness_contracts.json"
    verifier.configure_contract_store(correctness_contracts_path)
    memory_path = knowledge_root / "experience.jsonl"
    if isinstance(memory_cfg, dict) and isinstance(memory_cfg.get("path"), str):
        memory_path = Path(memory_cfg["path"])
    experience_memory = ExperienceMemory.from_config(
        memory_cfg if isinstance(memory_cfg, dict) else {}, memory_path
    )
    arg_rules_state: List[Dict[str, object]] = []
    if isinstance(orchestrator_cfg, dict) and orchestrator_cfg.get("enabled", False):
        orch_config = AgentConfig(
            enabled=True,
            api_key_env=orchestrator_cfg.get("api_key_env", "DEEPSEEK_API_KEY"),
            base_url=orchestrator_cfg.get("base_url", "https://api.deepseek.com"),
            model=orchestrator_cfg.get("model", "deepseek-chat"),
            temperature=float(orchestrator_cfg.get("temperature", 0.3)),
            max_tokens=int(orchestrator_cfg.get("max_tokens", 4096)),
            max_turns=int(orchestrator_cfg.get("max_turns", 25)),
            max_tool_calls_per_turn=int(orchestrator_cfg.get("max_tool_calls_per_turn", 5)),
        )
        input_script_path = Path(job.input_script) if job.input_script else None
        orchestrator_agent = OrchestratorAgent(
            config=orch_config,
            repo_root=_resolve_app_repo_root(
                Path(__file__).resolve().parents[1],
                job,
                adapter_cfg if isinstance(adapter_cfg, dict) else None,
            ),
            input_script_path=input_script_path,
            experience_db=experience_memory,
        )
    state = StopState()
    trace_events: List[Dict[str, object]] = TraceBuffer(artifacts_dir / "agent_trace.json")
    phase = "RUN_TUNE"
    frozen_run_id: Optional[str] = None
    frozen_build_id: Optional[str] = None
    seed_experiments: List[ExperimentIR] = []
    resume_best_chain_exp: Optional[ExperimentIR] = None
    if isinstance(resume_state, dict):
        start_phase = resume_state.get("start_phase")
        if isinstance(start_phase, str) and start_phase:
            phase = start_phase.upper()
        frozen_run_id = resume_state.get("frozen_run_id") or frozen_run_id
        frozen_build_id = resume_state.get("frozen_build_id") or frozen_build_id
        chain_path = resume_state.get("best_chain_exp_path")
        if chain_path:
            resume_best_chain_exp = _read_experiment(Path(chain_path))
        for entry in resume_state.get("seed_experiments", []) or []:
            if not entry:
                continue
            exp = _read_experiment(Path(entry))
            if exp:
                seed_experiments.append(exp)
    seed_baseline: Optional[ExperimentIR] = None
    seed_nonbaseline: List[ExperimentIR] = []
    if seed_experiments:
        for exp in seed_experiments:
            if exp.action is None or (exp.action and exp.action.action_id == "baseline"):
                if seed_baseline is None:
                    seed_baseline = exp
                else:
                    seed_nonbaseline.append(exp)
            else:
                seed_nonbaseline.append(exp)
    retune_remaining = int((planner_cfg or {}).get("retune_budget", 1))
    post_patch_retune = int((planner_cfg or {}).get("post_patch_retune_budget", 0))
    retune_cfg = (planner_cfg or {}).get("retune_policy", {})
    retune_min_improvement = 0.0
    retune_max_candidates = 0
    retune_families: Optional[List[str]] = None
    if isinstance(retune_cfg, dict):
        retune_min_improvement = float(retune_cfg.get("min_improvement_pct", 0.0) or 0.0)
        retune_max_candidates = int(retune_cfg.get("max_candidates", 0) or 0)
        families = retune_cfg.get("families")
        if isinstance(families, list):
            retune_families = [str(item) for item in families]
    retune_origin: Optional[str] = None
    two_phase_cfg = (planner_cfg or {}).get("two_phase", {})
    use_two_phase = isinstance(two_phase_cfg, dict) and two_phase_cfg.get("enabled", False)
    phase1_cache_path: Optional[Path] = None
    phase1_cache_payload: Optional[Dict[str, object]] = None
    cached_phase1_action: Optional[ActionIR] = None
    cached_phase1_entry: Optional[Dict[str, object]] = None
    prefer_phase1_cache_seed = bool((planner_cfg or {}).get("prefer_phase1_cache_seed", True))
    if use_two_phase and _phase1_cache_enabled(planner_cfg):
        phase1_cache_path = _phase1_cache_path(artifacts_dir, planner_cfg)
        phase1_cache_payload = _load_phase1_cache(phase1_cache_path)
        if phase1_cache_payload:
            cached_phase1_action, cached_phase1_entry = _lookup_phase1_cached_action(
                actions, job, phase1_cache_payload
            )
            if cached_phase1_action is None and phase1_cache_path:
                cached_phase1_action, cached_phase1_entry = _bootstrap_phase1_cache_from_history(
                    artifacts_dir=artifacts_dir,
                    actions=actions,
                    job=job,
                    cache_path=phase1_cache_path,
                    cache_payload=phase1_cache_payload,
                )

    start_time = time.monotonic()
    system_caps = _system_caps(repo_root)

    # Inject detected MPI launcher into adapter_cfg so app adapters can use it
    if system_caps.get("mpi_launcher") and isinstance(adapter_cfg, dict):
        adapter_cfg.setdefault("mpi_launcher", system_caps["mpi_launcher"])

    candidate_policy_summary = _select_candidate_policy(candidate_policy, job.case_id)
    llm_summary_enabled = bool((planner_cfg or {}).get("llm_summary", False))
    direction_map = {
        direction.get("id"): direction
        for direction in (direction_space or [])
        if isinstance(direction, dict) and direction.get("id")
    }
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
        )
    last_review_decision: Optional[Dict[str, object]] = None
    baseline_from_seed = False
    baseline_seeded_by_phase1_cache = False
    if skip_baseline and seed_baseline:
        baseline_exp = seed_baseline
        baseline_from_seed = True
        trace_events.append(
            {
                "event": "baseline_reused",
                "agent": "Orchestrator",
                "run_id": baseline_exp.run_id,
                "path": str(baseline_exp.patch_path) if baseline_exp.patch_path else None,
            }
        )
    elif skip_baseline and seed_nonbaseline:
        baseline_exp = _best_pass_exp(seed_nonbaseline) or seed_nonbaseline[0]
        baseline_from_seed = True
        trace_events.append(
            {
                "event": "baseline_reused",
                "agent": "Orchestrator",
                "run_id": baseline_exp.run_id,
                "path": str(baseline_exp.patch_path) if baseline_exp.patch_path else None,
                "note": "baseline skipped; reusing best seed experiment",
            }
        )
    elif (not skip_baseline) and prefer_phase1_cache_seed and cached_phase1_action:
        seeded_env, seeded_run_args, _ = _apply_run_config_action(
            job, cached_phase1_action, arg_rules=arg_rules_state
        )
        seeded_job = job.model_copy(
            update={
                "env": seeded_env,
                "run_args": seeded_run_args,
            }
        )
        baseline_exp = executor.execute(
            exp_id="baseline",
            job=seeded_job,
            base_job=None,
            base_run_id=None,
            base_action_id=None,
            action=None,
            actions_root=repo_root,
            policy=policy,
            gates=gates,
            profiler=profiler,
            verifier=verifier,
            artifacts_dir=artifacts_dir,
            time_command=time_command,
            profiling_cfg=profiling_cfg,
            wrappers_cfg=wrappers_cfg,
            build_cfg=build_cfg or {},
            build_packs=build_packs,
            adapter_cfg=adapter_cfg,
            repeats=baseline_repeats,
            runtime_agg=baseline_stat,
            baseline_exp=None,
            baseline_exp_for_verify=None,
            baseline_runtime=None,
            prior_samples=None,
            trace_events=trace_events,
            parent_run_id=None,
            iteration=None,
            llm_trace=None,
            reporter=reporter,
            run_purpose="score",
        )
        baseline_seeded_by_phase1_cache = True
        trace_events.append(
            {
                "event": "baseline_seeded_from_phase1_cache",
                "agent": "Orchestrator",
                "action_id": cached_phase1_action.action_id,
                "cached_best_run_id": (
                    cached_phase1_entry.get("best_run_id")
                    if isinstance(cached_phase1_entry, dict)
                    else None
                ),
                "run_id": baseline_exp.run_id,
            }
        )
        if reporter:
            reporter._section("Baseline: Cache Seed")
            reporter._print(
                f"Seed baseline with cached Phase 1 config: {cached_phase1_action.action_id}"
            )
    else:
        baseline_exp = executor.execute(
            exp_id="baseline",
            job=job,
            base_job=None,
            base_run_id=None,
            base_action_id=None,
            action=None,
            actions_root=repo_root,
            policy=policy,
            gates=gates,
            profiler=profiler,
            verifier=verifier,
            artifacts_dir=artifacts_dir,
            time_command=time_command,
            profiling_cfg=profiling_cfg,
            wrappers_cfg=wrappers_cfg,
            build_cfg=build_cfg or {},
            build_packs=build_packs,
            adapter_cfg=adapter_cfg,
            repeats=baseline_repeats,
            runtime_agg=baseline_stat,
            baseline_exp=None,
            baseline_exp_for_verify=None,
            baseline_runtime=None,
            prior_samples=None,
            trace_events=trace_events,
            parent_run_id=None,
            iteration=None,
            llm_trace=None,
            reporter=reporter,
            run_purpose="score",
        )
    if baseline_seeded_by_phase1_cache and baseline_exp.verdict == "FAIL":
        trace_events.append(
            {
                "event": "baseline_seed_failed_fallback",
                "agent": "Orchestrator",
                "seed_action_id": cached_phase1_action.action_id if cached_phase1_action else None,
                "seed_run_id": baseline_exp.run_id,
                "reasons": list(baseline_exp.reasons or []),
            }
        )
        if reporter:
            reporter._print(
                "Seeded baseline failed gates; fallback to canonical single-thread baseline."
            )
        baseline_exp = executor.execute(
            exp_id="baseline",
            job=job,
            base_job=None,
            base_run_id=None,
            base_action_id=None,
            action=None,
            actions_root=repo_root,
            policy=policy,
            gates=gates,
            profiler=profiler,
            verifier=verifier,
            artifacts_dir=artifacts_dir,
            time_command=time_command,
            profiling_cfg=profiling_cfg,
            wrappers_cfg=wrappers_cfg,
            build_cfg=build_cfg or {},
            build_packs=build_packs,
            adapter_cfg=adapter_cfg,
            repeats=baseline_repeats,
            runtime_agg=baseline_stat,
            baseline_exp=None,
            baseline_exp_for_verify=None,
            baseline_runtime=None,
            prior_samples=None,
            trace_events=trace_events,
            parent_run_id=None,
            iteration=None,
            llm_trace=None,
            reporter=reporter,
            run_purpose="score",
        )
        baseline_seeded_by_phase1_cache = False
    memory.record(baseline_exp)
    chain_min_improvement_pct = float(
        (planner_cfg or {}).get("chain_min_improvement_pct", 0.001) or 0.001
    )
    memory.min_best_improvement_pct = chain_min_improvement_pct
    best_chain_exp = baseline_exp
    best_chain_runtime = baseline_exp.results.runtime_seconds
    profile_probe_enabled = bool((planner_cfg or {}).get("profile_each_iteration", True))
    profile_probe_repeats = int((planner_cfg or {}).get("profile_probe_repeats", 1) or 1)
    latest_profile_report: Optional[ProfileReport] = baseline_exp.profile_report
    if profile_probe_enabled and profiling_cfg:
        baseline_profile_exp = executor.execute(
            exp_id="baseline-profile",
            job=job,
            base_job=baseline_exp.job,
            base_run_id=baseline_exp.run_id,
            base_action_id=baseline_exp.action.action_id if baseline_exp.action else "baseline",
            action=None,
            actions_root=repo_root,
            policy=policy,
            gates=gates,
            profiler=profiler,
            verifier=verifier,
            artifacts_dir=artifacts_dir,
            time_command=time_command,
            profiling_cfg=profiling_cfg,
            wrappers_cfg=wrappers_cfg,
            build_cfg=build_cfg or {},
            build_packs=build_packs,
            adapter_cfg=adapter_cfg,
            repeats=max(1, profile_probe_repeats),
            runtime_agg=baseline_stat,
            baseline_exp=baseline_exp,
            baseline_exp_for_verify=baseline_exp,
            baseline_runtime=baseline_exp.results.runtime_seconds,
            prior_samples=None,
            trace_events=trace_events,
            parent_run_id=baseline_exp.run_id,
            iteration=None,
            llm_trace=None,
            reporter=reporter,
            run_purpose="profile",
        )
        latest_profile_report = baseline_profile_exp.profile_report
        if baseline_profile_exp.profile_report:
            baseline_exp.profile_report = baseline_profile_exp.profile_report
    if seed_nonbaseline:
        for exp in seed_nonbaseline:
            memory.record(exp)
        trace_events.append(
            {
                "event": "resume_seed",
                "agent": "Orchestrator",
                "phase": phase,
                "frozen_run_id": frozen_run_id,
                "frozen_build_id": frozen_build_id,
                "seed_runs": [exp.run_id for exp in seed_nonbaseline],
            }
        )
        if memory.best and memory.best.verdict == "PASS":
            best_chain_exp = memory.best
            best_chain_runtime = memory.best.results.runtime_seconds
    if resume_best_chain_exp and resume_best_chain_exp.verdict == "PASS":
        best_chain_exp = resume_best_chain_exp
        best_chain_runtime = resume_best_chain_exp.results.runtime_seconds
        memory.best = resume_best_chain_exp
    if any(
        exp.action and exp.action.family == "neighbor_tune" for exp in memory.experiments
    ):
        state.neighbor_tune_done = True
        state.blocked_families.add("neighbor_tune")
    if not baseline_from_seed:
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
            candidate_policy=candidate_policy_summary,
            review_decision=last_review_decision,
            phase_transitions=_extract_phase_transitions(trace_events),
            composite_exp=None,
            min_improvement_pct=min_improvement_pct,
        )
        report_info["agent_trace"] = agent_trace_path
        return report_info

    # --- Two-phase dispatch ---
    if use_two_phase:
        best_param_exp: Optional[ExperimentIR] = None
        param_experiments: List[ExperimentIR] = []
        if baseline_seeded_by_phase1_cache:
            best_param_exp = baseline_exp
            trace_events.append(
                {
                    "event": "phase1_cache_seed_accepted",
                    "agent": "Orchestrator",
                    "run_id": baseline_exp.run_id,
                    "action_id": cached_phase1_action.action_id if cached_phase1_action else None,
                }
            )
            if reporter and cached_phase1_action:
                reporter._print(
                    f"Phase 1 skipped: seeded baseline already uses cached best "
                    f"{cached_phase1_action.action_id}"
                )
        elif cached_phase1_action:
            trace_events.append({
                "event": "phase1_cache_hit",
                "agent": "Orchestrator",
                "case_id": job.case_id,
                "action_id": cached_phase1_action.action_id,
                "cached_best_run_id": (
                    cached_phase1_entry.get("best_run_id")
                    if isinstance(cached_phase1_entry, dict)
                    else None
                ),
            })
            if reporter:
                reporter._section("Phase 1: Cache Reuse")
                reporter._print(
                    f"Phase 1 cache hit for {job.case_id}: "
                    f"{cached_phase1_action.action_id} (single-run validation)"
                )
            try:
                cached_exp = executor.execute(
                    exp_id=f"phase1-cache-{cached_phase1_action.action_id}",
                    job=job,
                    base_job=baseline_exp.job,
                    base_run_id=baseline_exp.run_id,
                    base_action_id=baseline_exp.action.action_id if baseline_exp.action else None,
                    action=cached_phase1_action,
                    actions_root=repo_root,
                    policy=policy,
                    gates=gates,
                    profiler=profiler,
                    verifier=verifier,
                    artifacts_dir=artifacts_dir,
                    time_command=time_command,
                    profiling_cfg=profiling_cfg,
                    wrappers_cfg=wrappers_cfg,
                    build_cfg=build_cfg or {},
                    build_packs=build_packs,
                    adapter_cfg=adapter_cfg,
                    repeats=1,
                    runtime_agg="mean",
                    baseline_exp=baseline_exp,
                    baseline_exp_for_verify=baseline_exp,
                    baseline_runtime=baseline_exp.results.runtime_seconds,
                    prior_samples=None,
                    trace_events=trace_events,
                    parent_run_id=None,
                    iteration=0,
                    llm_trace=None,
                    reporter=reporter,
                    arg_rules=arg_rules_state,
                )
                param_experiments.append(cached_exp)
                memory.record(cached_exp)
                experience_memory.record_experiment(cached_exp, baseline_exp)
                state.run_count += 1
                improvement_pct = _phase1_improvement_pct(
                    baseline_exp.results.runtime_seconds,
                    cached_exp.results.runtime_seconds,
                )
                min_required_pct = chain_min_improvement_pct * 100.0
                if (
                    cached_exp.verdict == "PASS"
                    and cached_exp.results.runtime_seconds > 0
                    and improvement_pct >= min_required_pct
                ):
                    best_param_exp = cached_exp
                    trace_events.append({
                        "event": "phase1_cache_accepted",
                        "agent": "Orchestrator",
                        "run_id": cached_exp.run_id,
                        "action_id": cached_phase1_action.action_id,
                        "improvement_pct": improvement_pct,
                    })
                    if reporter:
                        reporter._print(
                            f"Phase 1 cache accepted: {cached_exp.results.runtime_seconds:.3f}s "
                            f"({improvement_pct:.2f}% faster vs baseline)"
                        )
                else:
                    trace_events.append({
                        "event": "phase1_cache_rejected",
                        "agent": "Orchestrator",
                        "action_id": cached_phase1_action.action_id,
                        "verdict": cached_exp.verdict,
                        "improvement_pct": improvement_pct,
                        "min_required_pct": min_required_pct,
                    })
                    if reporter:
                        reporter._print(
                            "Phase 1 cache validation failed; falling back to full exploration."
                        )
            except LLMUnavailableError:
                raise
            except Exception as exc:
                trace_events.append({
                    "event": "phase1_cache_error",
                    "agent": "ExecutorAgent",
                    "action_id": cached_phase1_action.action_id,
                    "error": str(exc),
                })
                if reporter:
                    reporter._print(
                        "Phase 1 cache execution failed; falling back to full exploration."
                    )
        if best_param_exp is None:
            best_param_exp, param_experiments = _run_parameter_exploration_phase(
                job=job,
                actions=actions,
                baseline_exp=baseline_exp,
                executor=executor,
                profiler=profiler,
                verifier=verifier,
                policy=policy,
                gates=gates,
                artifacts_dir=artifacts_dir,
                time_command=time_command,
                profiling_cfg=profiling_cfg,
                wrappers_cfg=wrappers_cfg,
                build_cfg=build_cfg,
                build_packs=build_packs,
                adapter_cfg=adapter_cfg,
                planner_cfg=planner_cfg,
                experience_memory=experience_memory,
                memory=memory,
                state=state,
                trace_events=trace_events,
                reporter=reporter,
                repo_root=repo_root,
                system_caps=system_caps,
                chain_min_improvement_pct=chain_min_improvement_pct,
                arg_rules=arg_rules_state,
            )

        if best_param_exp and best_param_exp.verdict == "PASS":
            if phase1_cache_path and phase1_cache_payload is not None:
                _record_phase1_cache_entry(
                    phase1_cache_path, phase1_cache_payload, job, baseline_exp, best_param_exp
                )
            best_chain_exp = best_param_exp
            best_chain_runtime = best_param_exp.results.runtime_seconds
            memory.best = best_param_exp
            phase = "PATCH"  # Phase 2: directly enter source patch iteration
            frozen_run_id = best_param_exp.run_id
            trace_events.append({
                "event": "two_phase_enter_patch",
                "agent": "Orchestrator",
                "best_param_run_id": best_param_exp.run_id,
                "best_param_runtime": best_param_exp.results.runtime_seconds,
            })
        else:
            # Exploration failed or no improvement — fall back to iterative mode
            use_two_phase = False
            trace_events.append({
                "event": "two_phase_fallback",
                "agent": "Orchestrator",
                "reason": "exploration produced no improvement",
            })

    if profile_probe_enabled and profiling_cfg and best_chain_exp:
        pre_da_probe_action = _build_source_patch_replay_action(best_chain_exp, memory.experiments)
        pre_da_probe = executor.execute(
            exp_id="phase2-precheck-profile",
            job=job,
            base_job=best_chain_exp.job,
            base_run_id=best_chain_exp.run_id,
            base_action_id=best_chain_exp.action.action_id if best_chain_exp.action else "baseline",
            action=pre_da_probe_action,
            actions_root=repo_root,
            policy=policy,
            gates=gates,
            profiler=profiler,
            verifier=verifier,
            artifacts_dir=artifacts_dir,
            time_command=time_command,
            profiling_cfg=profiling_cfg,
            wrappers_cfg=wrappers_cfg,
            build_cfg=build_cfg or {},
            build_packs=build_packs,
            adapter_cfg=adapter_cfg,
            repeats=max(1, profile_probe_repeats),
            runtime_agg="mean",
            baseline_exp=baseline_exp,
            baseline_exp_for_verify=baseline_exp,
            baseline_runtime=baseline_exp.results.runtime_seconds,
            prior_samples=None,
            trace_events=trace_events,
            parent_run_id=best_chain_exp.run_id,
            iteration=None,
            llm_trace=None,
            reporter=reporter,
            run_purpose="profile",
        )
        latest_profile_report = pre_da_probe.profile_report
        if pre_da_probe.profile_report:
            best_chain_exp.profile_report = pre_da_probe.profile_report
            if best_chain_exp.run_id == baseline_exp.run_id:
                baseline_exp.profile_report = pre_da_probe.profile_report

    # --- Deep Code Analysis (runs once at start of Phase 2) ---
    deep_init = _run_phase2_deep_analysis_init(
        use_two_phase=use_two_phase,
        phase=phase,
        planner_cfg=planner_cfg,
        reporter=reporter,
        job=job,
        best_chain_exp=best_chain_exp,
        baseline_exp=baseline_exp,
        memory=memory,
        repo_root=repo_root,
        adapter_cfg=adapter_cfg,
        build_cfg=build_cfg,
        experience_memory=experience_memory,
        system_caps=system_caps,
        patch_families=patch_families,
        patch_planner=patch_planner,
        artifacts_dir=artifacts_dir,
        trace_events=trace_events,
    )
    deep_analysis_opportunities = deep_init.opportunities
    _deep_analysis_result = deep_init.deep_analysis_result
    _opportunity_graph = deep_init.opportunity_graph
    _selected_opportunities = deep_init.selected_opportunities

    opportunity_graph_mode = _opportunity_graph is not None

    # --- Batch selection + reflection state ---
    _batch_cfg = (planner_cfg or {}).get("batch_selection", {})
    _batch_for_opportunity_graph = bool(_batch_cfg.get("enable_for_opportunity_graph", True))
    use_batch_selection = bool(_batch_cfg.get("enabled", False)) and (
        _batch_for_opportunity_graph or not opportunity_graph_mode
    )
    _batch_min = int(_batch_cfg.get("min_batch", 3))
    _batch_max = int(_batch_cfg.get("max_batch", 5))
    _refl_cfg = (planner_cfg or {}).get("reflection", {})
    use_reflection = bool(_refl_cfg.get("enabled", False))
    deep_succeeded_ids: set = set()
    deep_failed_ids: set = set()
    deep_skipped_ids: set = set()
    last_reflection: Optional[ReflectionResult] = None
    decision_feedback: Dict[str, object] = {}
    reflection_agent = ReflectionAgent(llm_client) if use_reflection else None
    prev_best_time = best_chain_runtime
    iteration = 0
    # Accumulate patch failure descriptions for feedback to the code patch agent
    patch_failure_feedback: List[str] = []
    refine_cfg = (planner_cfg or {}).get("refine_on_best", {})
    refine_enabled = bool(refine_cfg.get("enabled", False))
    refine_min_improvement = float(refine_cfg.get("min_improvement_pct", min_improvement_pct or 0.0))
    while True:
        elapsed = time.monotonic() - start_time
        if should_stop(job.budgets, iteration, state, elapsed, min_delta_seconds):
            if reporter:
                reporter.stop(
                    _stop_reason(job.budgets, iteration, state, elapsed, min_delta_seconds)
                )
            break
        iteration += 1
        prev_best_time_before = prev_best_time
        # Capture baseline for this iteration (for marginal gain calculation)
        iteration_baseline_exp = best_chain_exp
        phase_started = phase
        patch_only = phase_started == "PATCH" and bool(
            (planner_cfg or {}).get("patch_only", True)
        )
        patch_cooldown_subfamilies: set[str] = set()
        if state.patch_family_blocked_until:
            expired = [fam for fam, until in state.patch_family_blocked_until.items() if iteration > until]
            for fam in expired:
                state.patch_family_blocked_until.pop(fam, None)
            for fam, until in state.patch_family_blocked_until.items():
                if iteration > until:
                    continue
                if fam.startswith("source_patch:"):
                    patch_cooldown_subfamilies.add(fam.split(":", 1)[1])
                else:
                    state.blocked_families.add(fam)
        if state.neighbor_tune_done:
            state.blocked_families.add("neighbor_tune")

        refine_on_best = _should_refine_on_best(
            baseline_exp=baseline_exp,
            best_exp=memory.best,
            enabled=refine_enabled,
            min_improvement_pct=refine_min_improvement,
        )
        ctx_job = best_chain_exp.job
        input_text = _safe_read(Path(ctx_job.input_script))
        # Reconstruct effective run_args/env with adapter-injected flags
        # (e.g. "-sf omp") so that variant matching and hotspot detection
        # see the actual runtime configuration, not the raw JobIR.
        _eff_env, _eff_run_args, _ = _apply_run_config_action(
            ctx_job, best_chain_exp.action, arg_rules=arg_rules_state
        )
        ctx = RuleContext(
            job=ctx_job, input_text=input_text, run_args=_eff_run_args, env=_eff_env
        )
        _backend_variant: Optional[str] = None  # set later by _adapter_variant_files
        if profile_probe_enabled and profiling_cfg:
            probe_action = _build_source_patch_replay_action(best_chain_exp, memory.experiments)
            probe_base_job = best_chain_exp.job
            probe_base_run_id = best_chain_exp.run_id
            probe_base_action_id = best_chain_exp.action.action_id if best_chain_exp.action else "baseline"
            profile_probe_exp = executor.execute(
                exp_id=f"iter{iteration}-profile",
                job=job,
                base_job=probe_base_job,
                base_run_id=probe_base_run_id,
                base_action_id=probe_base_action_id,
                action=probe_action,
                actions_root=repo_root,
                policy=policy,
                gates=gates,
                profiler=profiler,
                verifier=verifier,
                artifacts_dir=artifacts_dir,
                time_command=time_command,
                profiling_cfg=profiling_cfg,
                wrappers_cfg=wrappers_cfg,
                build_cfg=build_cfg or {},
                build_packs=build_packs,
                adapter_cfg=adapter_cfg,
                repeats=max(1, profile_probe_repeats),
                runtime_agg="mean",
                baseline_exp=baseline_exp,
                baseline_exp_for_verify=baseline_exp,
                baseline_runtime=baseline_exp.results.runtime_seconds,
                prior_samples=None,
                trace_events=trace_events,
                parent_run_id=probe_base_run_id,
                iteration=iteration,
                llm_trace=None,
                reporter=reporter,
                run_purpose="profile",
            )
            latest_profile_report = profile_probe_exp.profile_report
            trace_events.append(
                {
                    "event": "iteration_profile_probe",
                    "agent": "ProfilerAgent",
                    "iteration": iteration,
                    "run_id": profile_probe_exp.run_id,
                    "verdict": profile_probe_exp.verdict,
                }
            )
        profile_ref = latest_profile_report or best_chain_exp.profile_report
        # PATCH phase uses merged hotspots from baseline + best + historical PASS runs.
        if phase == "PATCH":
            _patch_root = (adapter_cfg or {}).get("patch_root", "") if isinstance(adapter_cfg, dict) else ""
            merged_hotspots = _merge_function_hotspots(
                repo_root=repo_root,
                patch_root=_patch_root,
                baseline_exp=baseline_exp,
                best_exp=best_chain_exp,
                experiments=memory.experiments,
                max_entries=int(((planner_cfg or {}).get("patch_hotspots_max", 160)) or 160),
            )
            if merged_hotspots:
                if profile_ref:
                    profile_ref = profile_ref.model_copy(deep=True)
                elif baseline_exp.profile_report:
                    profile_ref = baseline_exp.profile_report.model_copy(deep=True)
                else:
                    profile_ref = ProfileReport(timing_breakdown={}, system_metrics={}, notes=[])
                profile_ref.tau_hotspots = merged_hotspots
                trace_events.append(
                    {
                        "event": "patch_hotspots_merged",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "hotspot_count": len(merged_hotspots),
                    }
                )
            elif baseline_exp.profile_report and baseline_exp.profile_report.tau_hotspots:
                profile_ref = baseline_exp.profile_report
        history_summary = _build_history_summary(memory.experiments)
        cost_model = _build_cost_model(memory.experiments)
        tested_actions = [exp.action.action_id for exp in memory.experiments if exp.action]
        profile_features = build_profile_features(profile_ref)
        best_threads = None
        if ctx_job.env:
            try:
                best_threads = int(ctx_job.env.get("OMP_NUM_THREADS"))
            except (TypeError, ValueError):
                best_threads = None
        if best_threads:
            profile_features["best_threads"] = best_threads
        input_summary = _summarize_input_script(input_text)
        _iter_patch_root = ""
        if isinstance(adapter_cfg, dict):
            _iter_patch_root = adapter_cfg.get("patch_root", "") or ""
        hotspot_map = _hotspot_map(
            input_text, repo_root, ctx.run_args,
            patch_root=_iter_patch_root,
            deep_analysis_result=_deep_analysis_result,
            function_hotspots=(profile_ref.tau_hotspots if profile_ref else None),
        )
        memory_hints = _build_memory_hints(
            experience_memory,
            ctx_job.case_id,
            ctx_job.app,
            _backend_from_args(ctx_job.run_args or []),
        )
        planner_context = _build_planner_context(
            job=job,
            input_summary=input_summary,
            profile=profile_ref,
            profile_features=profile_features,
            hotspot_map=hotspot_map,
            system_caps=system_caps,
        )
        generated_actions: List[ActionIR] = []
        generated_ideas_payload: Optional[Dict[str, object]] = None
        code_survey_payload: Optional[Dict[str, object]] = None
        snippet_feature_map: Dict[Tuple[str, Optional[str]], Dict[str, object]] = {}
        strict_patch_generation = bool((planner_cfg or {}).get("strict_patch_generation", True))
        replay_cfg = (planner_cfg or {}).get("memory_replay", {})
        if (
            phase == "PATCH"
            and not opportunity_graph_mode
            and isinstance(replay_cfg, dict)
            and replay_cfg.get("enabled", False)
        ):
            replay_max = int(replay_cfg.get("max_actions", 0) or 0)
            replay_min_gain = float(replay_cfg.get("min_gain_pct", 1.0) or 1.0)
            replay_root = Path("artifacts")
            if artifacts_dir and artifacts_dir.parent:
                replay_root = artifacts_dir.parent.parent
            replay_actions = _load_patch_replay_actions(
                replay_root,
                job.case_id,
                job.app,
                _backend_from_args(ctx.run_args or []),
                replay_min_gain,
                replay_max,
                set(tested_actions) | {action.action_id for action in actions},
                patch_families or {},
            )
            if replay_actions:
                generated_actions.extend(replay_actions)
                trace_events.append(
                    {
                        "event": "memory_replay_actions",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "action_ids": [action.action_id for action in replay_actions],
                    }
                )

        if not generated_actions and llm_client and llm_client.config.enabled:
            available_families = sorted({action.family for action in actions})
            existing_action_ids = sorted({action.action_id for action in actions})
            action_space_summary = _action_space_summary(actions, patch_families)
            survey_rules = _patch_rules(adapter_cfg, policy)
            survey_allowed_files = _expand_allowed_files(repo_root, survey_rules)
            _survey_patch_root = survey_rules.get("patch_root", "") if isinstance(survey_rules, dict) else ""
            if not _survey_patch_root and isinstance(adapter_cfg, dict):
                _survey_patch_root = adapter_cfg.get("patch_root", "")
            _survey_explore_cfg = (planner_cfg or {}).get("patch_exploration", {})
            _survey_allow_repo_wide = bool(
                _survey_explore_cfg.get("allow_repo_wide", True)
            ) if isinstance(_survey_explore_cfg, dict) else True
            if phase == "PATCH" and _survey_allow_repo_wide:
                _survey_repo_files = _collect_repo_source_files(
                    repo_root=repo_root,
                    patch_root=str(_survey_patch_root or ""),
                    max_files=int(_survey_explore_cfg.get("max_repo_files", 1200) or 1200)
                    if isinstance(_survey_explore_cfg, dict)
                    else 1200,
                )
                if _survey_repo_files:
                    survey_allowed_files = sorted(set(survey_allowed_files) | set(_survey_repo_files))
            survey_max_snippets = int(survey_rules.get("max_snippets", 0) or 0)
            survey_max_context = int(survey_rules.get("max_context_chars", 0) or 0)
            survey_preferred_files: List[str] = []
            hotspot_files = hotspot_map.get("hotspot_files", []) if isinstance(hotspot_map, dict) else []
            for rel in hotspot_files:
                if isinstance(rel, str) and rel and rel not in survey_preferred_files:
                    survey_preferred_files.append(rel)
            for rel in _pair_style_files(input_text, repo_root, ctx.run_args):
                if rel not in survey_preferred_files:
                    survey_preferred_files.append(rel)
            survey_snippet_files = _select_snippet_files(
                repo_root,
                survey_rules,
                profile_ref.timing_breakdown if profile_ref else {},
                hotspot_files or survey_allowed_files,
                survey_preferred_files,
                survey_max_snippets,
            )
            survey_snippets = _collect_code_snippets(
                repo_root, survey_snippet_files, survey_max_snippets, survey_max_context
            )
            snippet_feature_map = _snippet_feature_map(survey_snippets)
            # --- Deep Analysis path (takes priority when opportunities remain) ---
            # Filter out skipped opportunities before selection
            if deep_skipped_ids:
                deep_analysis_opportunities = [
                    a for a in deep_analysis_opportunities
                    if (a.parameters or {}).get("deep_analysis_id") not in deep_skipped_ids
                ]
            if phase == "PATCH" and deep_analysis_opportunities:
                generated_actions.extend(
                    _seed_graph_actions_for_iteration(
                        deep_analysis_opportunities=deep_analysis_opportunities,
                        tested_actions=tested_actions,
                        use_batch_selection=use_batch_selection,
                        batch_min=_batch_min,
                        batch_max=_batch_max,
                        succeeded_ids=deep_succeeded_ids,
                        failed_ids=deep_failed_ids,
                        blocked_action_ids=state.blocked_actions,
                        iteration=iteration,
                        trace_events=trace_events,
                    )
                )
            # --- PatchPlanner path (legacy; disabled in graph-driven strict mode) ---
            elif phase == "PATCH" and survey_snippets and not opportunity_graph_mode:
                patch_family_ids = [
                    item.get("id")
                    for item in (patch_families or {}).get("families", [])
                    if isinstance(item, dict) and item.get("id")
                ]
                allowed_patch_families = set(patch_family_ids)
                _exp_hints = experience_memory.format_hints_for_prompt(
                    app=job.app,
                    backend=_backend_from_args(ctx.run_args or []),
                ) if experience_memory.config.enabled else []
                patch_plan = patch_planner.plan(
                    profile=profile_ref,
                    code_snippets=survey_snippets,
                    patch_families=patch_families or {},
                    allowed_files=survey_allowed_files,
                    experience_hints=_exp_hints or memory_hints or [],
                    backend_variant=_backend_variant,
                    max_actions=int((planner_cfg or {}).get("max_candidates", 3)),
                    existing_action_ids=existing_action_ids,
                )
                if patch_plan and patch_plan.status == "OK" and patch_plan.actions:
                    accepted_patch_actions = []
                    rejected_patch_actions = []
                    for pa in patch_plan.actions:
                        if allowed_patch_families and pa.patch_family not in allowed_patch_families:
                            rejected_patch_actions.append(
                                {
                                    "action_id": pa.action_id,
                                    "patch_family": pa.patch_family,
                                    "reason": "unsupported_patch_family_for_app",
                                }
                            )
                            continue
                        params = {
                            "patch_family": pa.patch_family,
                            "target_file": pa.target_file,
                            "target_anchor": pa.target_anchor,
                            "code_context": pa.code_context,
                            "origin": "patch_planner",
                        }
                        if pa.wrapper_id:
                            params["wrapper_id"] = pa.wrapper_id
                        gen_action = ActionIR(
                            action_id=pa.action_id,
                            family="source_patch",
                            description=pa.mechanism or pa.rationale,
                            applies_to=["source_patch"],
                            parameters=params,
                            expected_effect=(
                                [pa.expected_effect]
                                if pa.expected_effect and pa.expected_effect in _EXPECTED_EFFECTS
                                else _patch_family_effects(patch_families, pa.patch_family)
                            ),
                            risk_level=pa.risk_level or "medium",
                            verification_plan=VerificationPlan(
                                gates=_patch_family_gates(patch_families, pa.patch_family)
                            ),
                        )
                        generated_actions.append(gen_action)
                        accepted_patch_actions.append(pa.action_id)
                    trace_events.append(
                        {
                            "event": "patch_planner_actions",
                            "agent": "PatchPlannerAgent",
                            "iteration": iteration,
                            "actions": accepted_patch_actions,
                            "rejected": rejected_patch_actions,
                        }
                    )
                elif strict_patch_generation and llm_client and llm_client.config.enabled:
                    raise LLMUnavailableError(
                        "PatchPlannerAgent did not generate actionable source_patch proposals"
                    )
            # --- Legacy survey→idea→synth path (for non-PATCH phases) ---
            if not generated_actions and survey_snippets and not (phase == "PATCH" and opportunity_graph_mode):
                survey_context = {
                    "action_space": action_space_summary,
                    "code_snippets": survey_snippets,
                    "job": planner_context["job"],
                    "input_summary": input_summary,
                    "profile": planner_context["profile"],
                    "profile_features": profile_features,
                    "hotspot_map": hotspot_map,
                    "system_caps": system_caps,
                    "memory_hints": memory_hints,
                    "survey_guidance": survey_guidance or {},
                }
                survey_result = code_survey_agent.survey(survey_context)
                if reporter:
                    reporter.agent_trace(
                        "CodeSurveyAgent",
                        getattr(code_survey_agent, "last_llm_trace", None),
                    )
                if survey_result:
                    code_survey_payload = survey_result.model_dump()
                    if code_survey_payload and snippet_feature_map:
                        filtered, dropped = _filter_opportunities_by_structure(
                            list(code_survey_payload.get("opportunities", [])),
                            snippet_feature_map,
                        )
                        code_survey_payload["opportunities"] = filtered
                        if dropped:
                            trace_events.append(
                                {
                                    "event": "code_survey_filtered",
                                    "iteration": iteration,
                                    "dropped": dropped,
                                }
                            )
                    trace_events.append(
                        {
                            "event": "code_survey",
                            "agent": "CodeSurveyAgent",
                            "iteration": iteration,
                            "opportunities": [
                                item.get("opportunity_id")
                                for item in code_survey_payload.get("opportunities", [])
                            ],
                        }
                    )
                planner_context["code_survey_summary"] = _summarize_code_survey(
                    code_survey_payload
                )
                planner_context["action_space"] = action_space_summary
            idea_context = {
                "job": planner_context["job"],
                "input_summary": input_summary,
                "profile": planner_context["profile"],
                "profile_features": profile_features,
                "hotspot_map": hotspot_map,
                "system_caps": system_caps,
                "available_families": available_families,
            }
            ideas = idea_agent.propose(idea_context)
            if reporter:
                reporter.agent_trace(
                    "IdeaAgent",
                    getattr(idea_agent, "last_llm_trace", None),
                )
            generated_ideas_payload = ideas.model_dump() if ideas else {
                "ideas": [],
                "status": "OK",
                "missing_fields": [],
            }
            patch_family_ids = [
                item.get("id")
                for item in (patch_families or {}).get("families", [])
                if isinstance(item, dict) and item.get("id")
            ]
            synth_context = {
                "ideas": generated_ideas_payload,
                "code_opportunities": code_survey_payload.get("opportunities", [])
                if code_survey_payload
                else [],
                "job": planner_context["job"],
                "input_summary": input_summary,
                "profile": planner_context["profile"],
                "profile_features": profile_features,
                "hotspot_map": hotspot_map,
                "system_caps": system_caps,
                "available_families": available_families,
                "existing_action_ids": existing_action_ids,
                "adapter_hints": _adapter_hints(job, adapter_cfg),
                "patch_families": patch_family_ids,
                "patch_family_defs": action_space_summary.get("patch_family_defs", []),
                "memory_hints": memory_hints,
            }
            synth_actions = action_synth.synthesize(synth_context)
            if reporter:
                reporter.agent_trace(
                    "ActionSynthAgent",
                    getattr(action_synth, "last_llm_trace", None),
                )
            if synth_actions:
                generated_actions = _dedupe_actions(synth_actions.actions)
                for action in generated_actions:
                    params = dict(action.parameters or {})
                    params.setdefault("origin", "generated")
                    action.parameters = params
            if not generated_actions and code_survey_payload:
                generated_actions = _actions_from_opportunities(
                    code_survey_payload.get("opportunities", []),
                    patch_families,
                    max_actions=int((planner_cfg or {}).get("max_candidates", 3)),
                )
                if generated_actions:
                    for action in generated_actions:
                        params = dict(action.parameters or {})
                        params.setdefault("origin", "code_survey")
                        action.parameters = params
            if generated_actions:
                generated_actions, dropped_actions = _filter_generated_actions_by_structure(
                    generated_actions, snippet_feature_map
                )
                if dropped_actions:
                    trace_events.append(
                        {
                            "event": "generated_actions_filtered",
                            "iteration": iteration,
                            "dropped": dropped_actions,
                        }
                    )
            if generated_actions:
                trace_events.append(
                    {
                        "event": "generated_actions",
                        "agent": "ActionSynthAgent",
                        "iteration": iteration,
                        "actions": [action.action_id for action in generated_actions],
                    }
                )
                for action in generated_actions:
                    if action.family == "source_patch":
                        action.applies_to = ["source_patch"]
            if generated_actions and code_survey_payload:
                opportunities = code_survey_payload.get("opportunities", [])
                for action in generated_actions:
                    if action.family != "source_patch":
                        continue
                    params = dict(action.parameters or {})
                    if params.get("evidence"):
                        continue
                    target_file = params.get("target_file")
                    patch_family = params.get("patch_family")
                    for opp in opportunities:
                        if not isinstance(opp, dict):
                            continue
                        if target_file and target_file != opp.get("file_path"):
                            continue
                        if patch_family and patch_family != opp.get("patch_family"):
                            continue
                        evidence = opp.get("evidence")
                        if isinstance(evidence, list) and evidence:
                            params["evidence"] = evidence
                            action.parameters = params
                            break
        if _maybe_stop_on_opportunity_graph_exhausted(
            phase=phase,
            opportunity_graph_mode=opportunity_graph_mode,
            generated_actions=generated_actions,
            deep_analysis_opportunities=deep_analysis_opportunities,
            tested_actions=tested_actions,
            failed_ids=deep_failed_ids,
            blocked_action_ids=state.blocked_actions,
            iteration=iteration,
            trace_events=trace_events,
            reporter=reporter,
        ):
            break
        base_actions_for_iteration = _build_iteration_base_actions(
            actions=actions,
            generated_actions=generated_actions,
            phase=phase,
            opportunity_graph_mode=opportunity_graph_mode,
        )
        all_actions = _prepare_actions(
            base_actions=_dedupe_actions(base_actions_for_iteration),
            candidate_policy=candidate_policy,
            system_caps=system_caps,
            experiments=memory.experiments,
            adapter_cfg=adapter_cfg,
            job=job,
            fixed_threads=fixed_threads,
        )
        phase_targets = _phase_targets(phase)
        iteration_actions = _filter_actions_by_targets(all_actions, phase_targets)
        if not iteration_actions and all_actions:
            if phase == "PATCH" and opportunity_graph_mode:
                trace_events.append(
                    {
                        "event": "phase_targets_fallback_disabled",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "phase": phase,
                        "reason": "graph-driven patch mode disallows fallback action pool",
                    }
                )
            else:
                iteration_actions = list(all_actions)
                trace_events.append(
                    {
                        "event": "phase_targets_fallback",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "phase": phase,
                        "reason": "no actions in phase targets; fallback to full action pool",
                    }
                )
        selected_directions, direction_scores = _select_directions_by_signal(
            direction_map,
            profile_features,
            candidate_policy_summary,
            direction_top_k,
        )
        trace_events.append(
            {
                "event": "direction_selection",
                "agent": "DirectionSelector",
                "iteration": iteration,
                "direction_scores": direction_scores,
                "selected_directions": selected_directions,
            }
        )
        analysis = planner.analyze(
            profile_ref,
            history_summary,
            policy,
            job.tags,
            profile_features=profile_features,
        )
        direction_actions = list(iteration_actions)
        direction_actions, domain_rejections = enforce_candidate_policy(
            direction_actions, candidate_policy_summary or {}, profile_features
        )
        if domain_rejections:
            trace_events.append(
                {
                    "event": "candidate_domain_reject",
                    "agent": "ActionDomain",
                    "iteration": iteration,
                    "rejections": domain_rejections,
                }
            )
        available_actions = _apply_confidence_policy(direction_actions, analysis, policy)
        if not available_actions:
            available_actions = list(iteration_actions)
        eligible_pool = filter_actions(available_actions, ctx, None, policy)
        if not eligible_pool and available_actions:
            eligible_pool = list(available_actions)
        if state.blocked_families:
            eligible_pool = [
                action for action in eligible_pool if action.family not in state.blocked_families
            ]
        if state.blocked_actions:
            eligible_pool = [
                action for action in eligible_pool if action.action_id not in state.blocked_actions
            ]
        if patch_cooldown_subfamilies:
            eligible_pool = [
                action
                for action in eligible_pool
                if not (
                    action.family == "source_patch"
                    and isinstance((action.parameters or {}).get("patch_family"), str)
                    and (action.parameters or {}).get("patch_family") in patch_cooldown_subfamilies
                )
            ]
            trace_events.append(
                {
                    "event": "source_patch_subfamily_cooldown",
                    "agent": "Orchestrator",
                    "iteration": iteration,
                    "blocked_patch_families": sorted(patch_cooldown_subfamilies),
                }
            )
        fast_start_cfg = (planner_cfg or {}).get("fast_start", {})
        fast_start_active = bool(fast_start_cfg.get("enabled", False))
        fast_start_iters = int(fast_start_cfg.get("iterations", 0) or 0)
        fast_start_targets = set(fast_start_cfg.get("targets", []) or [])
        if phase == "PATCH":
            fast_start_active = False
        if fast_start_active and fast_start_iters > 0 and iteration <= fast_start_iters:
            family_targets = _family_targets(eligible_pool)
            fast_families = [
                fam
                for fam, targets in family_targets.items()
                if fast_start_targets and targets.intersection(fast_start_targets)
            ]
            if fast_families:
                analysis.allowed_families = sorted(set(fast_families))
                analysis.rationale = (
                    (analysis.rationale or "")
                    + " | fast_start: focus on run/build families"
                ).strip(" |")
                trace_events.append(
                    {
                        "event": "fast_start_override",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "families": analysis.allowed_families,
                        "targets": sorted(fast_start_targets),
                    }
                )
        backend_exp = _latest_backend_exp(memory.experiments)
        if (
            phase == "RUN_RETUNE"
            and backend_exp
            and _run_args_has_backend(ctx.run_args)
        ):
            filtered_pool = [
                action
                for action in eligible_pool
                if action.family != "runtime_backend_select"
            ]
            if filtered_pool:
                eligible_pool = filtered_pool
                analysis.allowed_families = [
                    fam for fam in analysis.allowed_families if fam != "runtime_backend_select"
                ]
                trace_events.append(
                    {
                        "event": "retune_backend_locked",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "reason": "backend already selected; skip runtime_backend_select during RUN_RETUNE",
                    }
                )
        if phase in {"RUN_TUNE", "RUN_RETUNE"} and _is_pthread_cli_model(ctx.run_args, eligible_pool):
            blocked_runtime_families = {"parallel_omp", "omp_pkg", "runtime_backend_select"}
            filtered_pool = [
                action for action in eligible_pool if action.family not in blocked_runtime_families
            ]
            if filtered_pool:
                dropped_families = sorted({a.family for a in eligible_pool if a.family in blocked_runtime_families})
                eligible_pool = filtered_pool
                if analysis.allowed_families:
                    analysis.allowed_families = [
                        fam for fam in analysis.allowed_families if fam not in blocked_runtime_families
                    ]
                trace_events.append(
                    {
                        "event": "runtime_thread_model_filter",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "model": "pthread_cli",
                        "run_args": list(ctx.run_args or []),
                        "dropped_families": dropped_families,
                    }
                )
        patch_scope_cfg = _patch_scope_config(policy)
        patch_rules = _patch_rules(adapter_cfg, policy)
        scope_levels = patch_scope_cfg.get("levels") or []
        scope_id = None
        if scope_levels:
            start_level = patch_scope_cfg.get("start_level")
            if (
                iteration == 1
                and state.patch_scope_level == 0
                and isinstance(start_level, str)
                and start_level in scope_levels
            ):
                state.patch_scope_level = scope_levels.index(start_level)
            scope_index = min(state.patch_scope_level, len(scope_levels) - 1)
            scope_id = scope_levels[scope_index]
            patch_rules = _apply_patch_scope(patch_rules, scope_id)
            trace_events.append(
                {
                    "event": "source_patch_scope_selected",
                    "agent": "Orchestrator",
                    "iteration": iteration,
                    "scope": scope_id,
                }
            )
        debug_max_attempts = int(patch_rules.get("debug_max_attempts", 0) or 0)
        worktree_retry_attempts = int(
            patch_rules.get("worktree_retry_attempts", debug_max_attempts or 2) or 2
        )
        _patch_root = patch_rules.get("patch_root", "") if isinstance(patch_rules, dict) else ""
        if not _patch_root and isinstance(adapter_cfg, dict):
            _patch_root = adapter_cfg.get("patch_root", "")
        explore_cfg = (planner_cfg or {}).get("patch_exploration", {})
        allow_repo_wide_explore = bool(
            explore_cfg.get("allow_repo_wide", True)
        ) if isinstance(explore_cfg, dict) else True
        max_repo_files = int(
            explore_cfg.get("max_repo_files", 1200)
        ) if isinstance(explore_cfg, dict) else 1200
        allowed_files = _expand_allowed_files(repo_root, patch_rules)
        if phase == "PATCH" and allow_repo_wide_explore:
            repo_source_files = _collect_repo_source_files(
                repo_root=repo_root,
                patch_root=str(_patch_root or ""),
                max_files=max_repo_files,
            )
            if repo_source_files:
                allowed_files = sorted(set(allowed_files) | set(repo_source_files))
                trace_events.append(
                    {
                        "event": "patch_repo_wide_explore_enabled",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "allowed_file_count": len(allowed_files),
                        "repo_source_file_count": len(repo_source_files),
                    }
                )
        max_snippets = int(patch_rules.get("max_snippets", 0) or 0)
        max_context = int(patch_rules.get("max_context_chars", 0) or 0)

        # --- preferred_files: agent-driven first, rule-based fallback ---
        def _normalise_to_repo_rel(raw: str) -> Optional[str]:
            """Resolve a file path to repo-relative, adding patch_root prefix if needed."""
            return _normalise_path_to_repo_rel(raw, repo_root, _patch_root)

        # Priority 1: deep analysis hotspot_files (agent judgment)
        agent_preferred: List[str] = []
        if _deep_analysis_result and _deep_analysis_result.hotspot_files:
            for hf in _deep_analysis_result.hotspot_files:
                rel = _normalise_to_repo_rel(hf)
                if rel and rel not in agent_preferred:
                    agent_preferred.append(rel)

        # Priority 2: per-iteration action target_files from deep analysis opportunities
        for action in eligible_pool:
            if action.family != "source_patch":
                continue
            tfiles = (action.parameters or {}).get("target_files", [])
            if isinstance(tfiles, list):
                for tf in tfiles:
                    rel = _normalise_to_repo_rel(tf)
                    if rel and rel not in agent_preferred:
                        agent_preferred.insert(0, rel)

        # Priority 3: rule-based pair_style file inference (fallback)
        rule_preferred = _pair_style_files(input_text, repo_root, ctx.run_args)

        # Merge: agent > rule, deduplicated
        preferred_files = list(agent_preferred)
        for rf in rule_preferred:
            if rf not in preferred_files:
                preferred_files.append(rf)

        # Variant file matching (still useful for filtering)
        variant_files, variant_ids = _adapter_variant_files(
            adapter_cfg,
            repo_root,
            ctx.run_args,
            input_text,
            ctx.env or {},
        )
        if variant_files:
            prefer_mode = None
            if isinstance(adapter_cfg, dict):
                prefer_mode = adapter_cfg.get("variants_prefer_mode")
            if prefer_mode == "exclusive":
                if preferred_files:
                    intersection = [p for p in preferred_files if p in variant_files]
                    preferred_files = intersection or list(preferred_files)
                if max_snippets and max_snippets > len(preferred_files):
                    max_snippets = len(preferred_files)
            else:
                preferred_files = preferred_files + [
                    vf for vf in variant_files if vf not in preferred_files
                ]
        _backend_variant = variant_ids[0] if variant_ids else None
        trace_events.append(
            {
                "event": "preferred_files_resolved",
                "agent": "Orchestrator",
                "iteration": iteration,
                "agent_preferred": agent_preferred,
                "rule_preferred": rule_preferred,
                "variant_ids": variant_ids,
                "final_preferred": preferred_files,
            }
        )
        snippet_files = _select_snippet_files(
            repo_root,
            patch_rules,
            profile_ref.timing_breakdown,
            allowed_files,
            preferred_files,
            max_snippets,
        )
        target_files: List[str] = []
        for action in eligible_pool:
            if action.family != "source_patch":
                continue
            params = action.parameters or {}
            tf = params.get("target_file")
            if isinstance(tf, str) and tf:
                norm = _normalise_to_repo_rel(tf)
                if norm and norm not in target_files:
                    target_files.append(norm)
            tfs = params.get("target_files")
            if isinstance(tfs, list):
                for t in tfs:
                    if isinstance(t, str) and t:
                        norm = _normalise_to_repo_rel(t)
                        if norm and norm not in target_files:
                            target_files.append(norm)
        if target_files:
            expanded_targets: List[str] = []
            for item in target_files:
                for rel in _target_with_related_files(
                    repo_root,
                    item,
                    allowed_files,
                    max_related=max(2, min(max_snippets or 8, 8)),
                ):
                    if rel not in expanded_targets:
                        expanded_targets.append(rel)
            for item in reversed(expanded_targets):
                if item not in snippet_files:
                    snippet_files.insert(0, item)
            if max_snippets:
                snippet_files = snippet_files[:max_snippets]
        available_hotspots = _available_hotspots_for_files(snippet_files, patch_rules)
        eligible_pool = _filter_source_patch_by_hotspot(
            eligible_pool, patch_families or {}, available_hotspots
        )
        coverage_required = bool(
            isinstance(patch_scope_cfg.get("coverage"), dict)
            and patch_scope_cfg.get("coverage", {}).get("require_all_families", False)
        )
        if coverage_required:
            untried_families = _untried_patch_families(memory.experiments, patch_families)
            eligible_pool, coverage_applied = _filter_source_patch_for_coverage(
                eligible_pool, untried_families
            )
            if coverage_applied:
                trace_events.append(
                    {
                        "event": "source_patch_coverage_filter",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "untried_families": sorted(untried_families),
                    }
                )
        # Filter out patch families that have repeatedly failed
        exhausted = _exhausted_patch_families(memory.experiments)
        if exhausted:
            before_len = len(eligible_pool)
            eligible_pool = [
                a for a in eligible_pool
                if not (
                    a.family == "source_patch"
                    and isinstance((a.parameters or {}).get("patch_family"), str)
                    and (a.parameters or {}).get("patch_family") in exhausted
                )
            ]
            if len(eligible_pool) < before_len:
                trace_events.append(
                    {
                        "event": "exhausted_families_filtered",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "exhausted_families": sorted(exhausted),
                    }
                )

        remaining_by_family: Dict[str, int] = {}
        for action in eligible_pool:
            if action.action_id in tested_actions:
                continue
            remaining_by_family[action.family] = remaining_by_family.get(action.family, 0) + 1

        # Navigation hints: lightweight file pointers for the agentic patcher.
        # The agent uses read_file/grep to explore code — hints only provide
        # coordinates so the agent knows where to start.
        nav_hint_limit = int(
            explore_cfg.get("max_navigation_hints", 64)
        ) if isinstance(explore_cfg, dict) else 64
        nav_seed_files = list(
            dict.fromkeys(
                list(snippet_files)
                + list(preferred_files)
                + list(allowed_files[: max(64, nav_hint_limit * 2)])
            )
        )
        nav_hints = _build_navigation_hints(
            repo_root,
            nav_seed_files,
            min(max(1, nav_hint_limit), max(1, len(nav_seed_files))),
        )

        # patch_allowed_files: use the adapter-level allowed_globs (already
        # expanded into `allowed_files`), NOT snippet paths.  This lets the
        # agent modify any file within the adapter's scope, not just the ones
        # that happened to be in the snippet selection.
        patch_allowed_files = allowed_files

        if nav_hints:
            trace_events.append(
                {
                    "event": "navigation_hints_selected",
                    "agent": "Orchestrator",
                    "iteration": iteration,
                    "hints": nav_hints,
                }
            )
        patch_no_candidates = False
        if remaining_by_family.get("source_patch", 0) > 0 and not nav_hints:
            # Fallback 1: deep analysis hotspot_files
            if _deep_analysis_result and _deep_analysis_result.hotspot_files:
                _fallback_files = [
                    _normalise_to_repo_rel(f)
                    for f in _deep_analysis_result.hotspot_files
                    if _normalise_to_repo_rel(f)
                ]
                if _fallback_files:
                    nav_hints = _build_navigation_hints(repo_root, _fallback_files, max_snippets or 12)
            # Fallback 2: allowed_files (agent explores freely)
            if not nav_hints and allowed_files:
                nav_hints = _build_navigation_hints(repo_root, allowed_files[:12], 12)
            # Only block if truly nothing available
            if not nav_hints:
                patch_no_candidates = True
                state.blocked_families.add("source_patch")
                remaining_by_family.pop("source_patch", None)
                trace_events.append(
                    {
                        "event": "source_patch_blocked",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "reason": "no navigation hints available for hotspot files",
                    }
                )
                if reporter:
                    reporter.skip("source_patch 暂停: 未采集到热点代码片段")
        families_in_pool = sorted({action.family for action in eligible_pool})
        allowed_from_analysis = list(analysis.allowed_families or [])
        coverage_floor = _coverage_floor_families(phase, eligible_pool, remaining_by_family)
        if coverage_floor:
            merged_allowed = sorted(set(allowed_from_analysis) | set(coverage_floor))
            allowed_from_analysis = merged_allowed
            analysis.allowed_families = merged_allowed
            trace_events.append(
                {
                    "event": "allowed_families_floor",
                    "agent": "Orchestrator",
                    "iteration": iteration,
                    "phase": phase,
                    "coverage_floor": coverage_floor,
                }
            )
        resolved_families: List[str] = []
        if allowed_from_analysis:
            resolved_families = [
                fam for fam in allowed_from_analysis if remaining_by_family.get(fam, 0) > 0
            ]
        if not resolved_families:
            resolved_families = [fam for fam, count in remaining_by_family.items() if count > 0]
        if not resolved_families:
            resolved_families = families_in_pool
        analysis.allowed_families = sorted(resolved_families)
        if phase == "RUN_RETUNE" and retune_families:
            allowed_set = set(retune_families)
            filtered = [action for action in eligible_pool if action.family in allowed_set]
            if filtered:
                eligible_pool = filtered
                if analysis.allowed_families:
                    analysis.allowed_families = [
                        fam for fam in analysis.allowed_families if fam in allowed_set
                    ]
                if not analysis.allowed_families:
                    analysis.allowed_families = sorted(
                        {action.family for action in eligible_pool}
                    )
                trace_events.append(
                    {
                        "event": "retune_family_filter",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "allowed_families": analysis.allowed_families,
                    }
                )
        trace_events.append(
            {
                "event": "allowed_families_resolved",
                "agent": "Orchestrator",
                "iteration": iteration,
                "allowed_families": analysis.allowed_families,
                "remaining_by_family": remaining_by_family,
            }
        )
        trace_events.append(
            {
                "event": "analysis",
                "agent": "PlannerAgent",
                "iteration": iteration,
                "bottleneck": analysis.bottleneck,
                "allowed_families": analysis.allowed_families,
                "confidence": analysis.confidence,
                "rationale": analysis.rationale,
                "refine_on_best": refine_on_best,
            }
        )
        if reporter:
            reporter.analysis(
                bottlenecks=[analysis.bottleneck],
                allowed_families=analysis.allowed_families,
                confidence=f"{analysis.confidence:.2f}",
                rationale=analysis.rationale,
            )
        availability = {fam: remaining_by_family.get(fam, 0) for fam in analysis.allowed_families}
        decision = None
        decision_action_by_cid: Dict[int, ActionIR] = {}
        use_orchestrator_decision = False
        if orchestrator_agent:
            action_space_payload = {
                "families": [],
                "actions": [],
            }
            seen_families: set[str] = set()
            for action in actions:
                if action.family not in seen_families:
                    seen_families.add(action.family)
                    action_space_payload["families"].append({
                        "id": action.family,
                        "description": action.description or "",
                        "expected_effect": list(action.expected_effect) if action.expected_effect else [],
                    })
                action_space_payload["actions"].append(action.model_dump())
            available_action_payload = []
            for idx, a in enumerate(eligible_pool, start=1):
                decision_action_by_cid[idx] = a
                available_action_payload.append(
                    {
                        "cid": idx,
                        "action_id": a.action_id,
                        "family": a.family,
                        "description": a.description,
                        "applies_to": a.applies_to,
                        "parameters": a.parameters,
                        "expected_effect": a.expected_effect,
                        "risk_level": a.risk_level,
                    }
                )
            decision_context = {
                "iteration": iteration,
                "phase": phase,
                "job": {
                    "case_id": job.case_id,
                    "app": job.app,
                    "tags": job.tags,
                    "run_args": ctx.run_args,
                    "env": ctx.env,
                    "workdir": job.workdir,
                    "app_bin": job.app_bin,
                },
                "budgets": job.budgets.model_dump(),
                "profile": {
                    "timing_breakdown": profile_ref.timing_breakdown,
                    "system_metrics": profile_ref.system_metrics,
                    "notes": profile_ref.notes,
                },
                "profile_features": profile_features,
                "system_caps": system_caps,
                "input_summary": input_summary,
                "history_summary": history_summary,
                "cost_model": cost_model,
                "tested_actions": tested_actions,
                "available_families": sorted({a.family for a in eligible_pool}),
                "available_actions": available_action_payload,
                "decision_feedback": decision_feedback,
                "arg_rules": arg_rules_state,
            }
            decision_result = orchestrator_agent.decide(
                profile={
                    "timing_breakdown": profile_ref.timing_breakdown,
                    "system_metrics": profile_ref.system_metrics,
                    "notes": profile_ref.notes,
                },
                action_space=action_space_payload,
                context=decision_context,
            )
            if reporter:
                reporter.agent_conversation("OrchestratorAgent", decision_result.conversation_log)
                if decision_result.decision:
                    reporter.agent_trace(
                        "OrchestratorAgent",
                        {
                            "payload": decision_context,
                            "response": decision_result.decision.model_dump(),
                        },
                    )
            if decision_result.decision and decision_result.decision.status in {"OK", "PARTIAL"}:
                decision = decision_result.decision
                if decision.arg_rules:
                    arg_rules_state = [rule.model_dump() for rule in decision.arg_rules]
                allowed = set(decision.allowed_families or [])
                blocked = set(decision.blocked_families or [])
                if decision.allowed_families:
                    analysis.allowed_families = list(decision.allowed_families)
                if allowed:
                    eligible_pool = [a for a in eligible_pool if a.family in allowed]
                if blocked:
                    eligible_pool = [a for a in eligible_pool if a.family not in blocked]
                use_orchestrator_decision = True
                planner.llm_client = None
                optimizer.llm_client = None
                ranker.llm_client = None
                reviewer.llm_client = None
                trace_events.append(
                    {
                        "event": "orchestrator_decision",
                        "agent": "OrchestratorAgent",
                        "iteration": iteration,
                        "status": decision.status,
                        "allowed_families": decision.allowed_families,
                        "blocked_families": decision.blocked_families,
                        "candidate_cids": decision.candidate_cids,
                        "ranking_cids": decision.ranking_cids,
                        "candidates": decision.candidates,
                        "ranking": decision.ranking,
                        "stop": decision.stop,
                        "reason": decision.reason,
                    }
                )
                if decision.stop:
                    if reporter:
                        reporter.stop(f"orchestrator_stop: {decision.reason}")
                    break
        plan = planner.plan(
            iteration,
            analysis,
            job.budgets,
            history_summary,
            availability,
            cost_model,
            context=planner_context,
        )
        neighbor_batch = False
        neighbor_batch_actions: List[ActionIR] = []
        if (
            "neighbor_tune" in plan.chosen_families
            and "neighbor_tune" not in state.blocked_families
            and not state.neighbor_tune_done
        ):
            neighbor_batch_actions = [
                action for action in eligible_pool if action.family == "neighbor_tune"
            ]
            if neighbor_batch_actions:
                neighbor_batch = True
                plan.chosen_families = ["neighbor_tune"]
                plan.max_candidates = max(plan.max_candidates, len(neighbor_batch_actions))
                state.blocked_families.add("neighbor_tune")
                state.neighbor_tune_done = True
                trace_events.append(
                    {
                        "event": "neighbor_tune_batch_start",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "count": len(neighbor_batch_actions),
                        "locked": True,
                    }
                )
        if fast_start_active and fast_start_iters > 0 and iteration <= fast_start_iters:
            per_family = int(fast_start_cfg.get("max_candidates_per_family", plan.max_candidates) or plan.max_candidates)
            plan.max_candidates = min(plan.max_candidates, per_family)
            trace_events.append(
                {
                    "event": "fast_start_limits",
                    "agent": "Orchestrator",
                    "iteration": iteration,
                    "max_candidates_per_family": plan.max_candidates,
                }
            )
        if phase == "RUN_RETUNE" and retune_max_candidates > 0:
            plan.max_candidates = min(plan.max_candidates, retune_max_candidates)
        patch_min_candidates = int((planner_cfg or {}).get("min_patch_candidates", 0) or 0)
        if patch_min_candidates > 0 and "source_patch" in plan.chosen_families:
            available_patch = remaining_by_family.get("source_patch", 0)
            if available_patch > 0:
                plan.max_candidates = max(
                    plan.max_candidates, min(patch_min_candidates, available_patch)
                )
        if phase in {"RUN_TUNE", "RUN_RETUNE"}:
            families_by_target = _families_by_target(eligible_pool, remaining_by_family)
            before = list(plan.chosen_families)
            runtime_prefer = _runtime_family_preference(ctx.run_args, eligible_pool)
            plan.chosen_families = _inject_family(
                plan.chosen_families,
                families_by_target.get("run_config", []),
                prefer=runtime_prefer,
            )
            probe_cfg = (planner_cfg or {}).get("runtime_probe", {})
            if isinstance(probe_cfg, dict):
                metrics = (profile_features or {}).get("metrics", {})
                neigh_ratio = float(metrics.get("neigh_ratio") or 0.0)
                comm_ratio = float(metrics.get("comm_ratio") or 0.0)
                io_ratio = float(metrics.get("io_ratio") or 0.0)
                min_neigh = float(probe_cfg.get("min_neigh_ratio", 0.0) or 0.0)
                min_comm = float(probe_cfg.get("min_comm_ratio", 0.0) or 0.0)
                min_output = float(probe_cfg.get("min_output_ratio", 0.0) or 0.0)
                input_candidates = families_by_target.get("input_script", [])
                prefer_input: List[str] = []
                if io_ratio >= min_output:
                    prefer_input = ["output_tune", "neighbor_tune"]
                elif neigh_ratio >= min_neigh or comm_ratio >= min_comm:
                    prefer_input = ["neighbor_tune", "output_tune"]
                if prefer_input and input_candidates:
                    plan.chosen_families = _inject_family(
                        plan.chosen_families,
                        input_candidates,
                        prefer=prefer_input,
                    )
            if before != plan.chosen_families:
                trace_events.append(
                    {
                        "event": "plan_runtime_coverage",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "before": before,
                        "after": plan.chosen_families,
                    }
                )
        if phase == "PATCH":
            survey_summary = planner_context.get("code_survey_summary", {})
            opp_count = int(survey_summary.get("count") or 0) if isinstance(survey_summary, dict) else 0
            if opp_count > 0:
                before = list(plan.chosen_families)
                plan.chosen_families = _inject_family(
                    plan.chosen_families,
                    ["source_patch"],
                    prefer=["source_patch"],
                    max_families=2,
                )
                if before != plan.chosen_families:
                    trace_events.append(
                        {
                            "event": "plan_patch_coverage",
                            "agent": "Orchestrator",
                            "iteration": iteration,
                            "before": before,
                            "after": plan.chosen_families,
                            "code_survey_count": opp_count,
                        }
                    )
        trace_events.append(
            {
                "event": "plan",
                "agent": "PlannerAgent",
                "iteration": iteration,
                "plan": plan.model_dump(),
                "cost_model": cost_model,
            }
        )
        if reporter:
            reporter.plan_summary(plan)
            reporter.agent_trace("PlannerAgent", getattr(planner, "last_llm_trace", None))

        if decision and decision.max_candidates is not None:
            if decision.max_candidates > 0:
                plan.max_candidates = int(decision.max_candidates)

        orchestrator_only = bool(
            use_orchestrator_decision
            and decision
            and (
                decision.candidate_cids
                or decision.ranking_cids
                or decision.candidates
                or decision.ranking
            )
        )
        ranked_actions: List[ActionIR] = []
        if orchestrator_only:
            ranked_actions = _select_actions_from_decision(
                eligible_pool, decision, top_k, action_by_cid=decision_action_by_cid
            )
            if not ranked_actions:
                orchestrator_only = False
                trace_events.append(
                    {
                        "event": "orchestrator_empty_candidates",
                        "agent": "OrchestratorAgent",
                        "iteration": iteration,
                        "requested_cids": decision.candidate_cids or decision.ranking_cids,
                        "requested": decision.candidates or decision.ranking,
                    }
                )
            else:
                trace_events.append(
                    {
                        "event": "orchestrator_ranked_actions",
                        "agent": "OrchestratorAgent",
                        "iteration": iteration,
                        "ranked_action_ids": [action.action_id for action in ranked_actions],
                    }
                )
                if reporter:
                    reporter.candidates(
                        actions=ranked_actions,
                        ranking_mode="orchestrator",
                        selection_mode=selection_mode,
                        llm_explanation=decision.reason or None,
                    )

        memory_keep_action_ids: List[str] = []
        if not orchestrator_only and experience_memory.config.enabled and available_actions:
            memory_context = {
                "case_id": job.case_id,
                "app": job.app,
                "backend": _backend_from_args(ctx_job.run_args or []),
            }
            all_scores = experience_memory.score_actions(available_actions, memory_context)
            best_by_family: Dict[str, Tuple[float, str]] = {}
            for action in available_actions:
                score = float(all_scores.get(action.action_id, 0.0))
                if score <= 0.0:
                    continue
                prev = best_by_family.get(action.family)
                if prev is None or score > prev[0]:
                    best_by_family[action.family] = (score, action.action_id)
            memory_keep_action_ids = [item[1] for item in best_by_family.values()]
        if not orchestrator_only:
            candidate_lists = optimizer.propose(
                actions=available_actions,
                ctx=ctx,
                plan=plan,
                policy=policy,
                profile=memory.best.profile_report if memory.best else baseline_exp.profile_report,
                exclude_action_ids=tested_actions,
                memory_keep_action_ids=memory_keep_action_ids,
                system_caps=system_caps,
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
            flat_candidates: List[ActionIR] = []
            for candidate_list in candidate_lists:
                flat_candidates.extend(candidate_list.candidates)
            memory_context = {
                "case_id": job.case_id,
                "app": job.app,
                "backend": _backend_from_args(ctx_job.run_args or []),
            }
            memory_posteriors = experience_memory.bayesian_posteriors(flat_candidates, memory_context)
            memory_scores = {
                action_id: float(item.get("utility", 0.0))
                for action_id, item in memory_posteriors.items()
            }
            rank_cfg = (planner_cfg or {}).get("ranking", {})
            patch_stats: Dict[str, Dict[str, int]] = {}
            for action in flat_candidates:
                action_id = action.action_id
                patch_stats[action_id] = {
                    "context_miss": state.patch_action_context_misses.get(action_id, 0),
                    "preflight_fail": state.patch_action_preflight_fails.get(action_id, 0),
                    "build_fail": 0,
                }
            ranked = ranker.rank(
                candidate_lists,
                ctx,
                policy,
                profile_ref,
                profile_features=profile_features,
                hotspot_map=hotspot_map,
                rank_cfg=rank_cfg if isinstance(rank_cfg, dict) else {},
                tested_actions=tested_actions,
                memory_scores=memory_scores,
                memory_posteriors=memory_posteriors,
                patch_stats=patch_stats,
            )
            rank_limit = min(plan.max_candidates, top_k, max(len(ranked.ranked), 1))
            if fast_start_active and fast_start_iters > 0 and iteration <= fast_start_iters:
                max_total = int(fast_start_cfg.get("max_total_candidates", rank_limit) or rank_limit)
                if ranked.ranked:
                    rank_limit = min(top_k, max_total, len(ranked.ranked))
                else:
                    rank_limit = 0
                trace_events.append(
                    {
                        "event": "fast_start_rank_limit",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "rank_limit": rank_limit,
                    }
                )
            max_explore_frac = 0.2
            if isinstance(rank_cfg, dict):
                max_explore_frac = float(rank_cfg.get("max_explore_frac", max_explore_frac) or max_explore_frac)
            ranked_subset = _select_with_evidence(ranked.ranked, rank_limit, max_explore_frac=max_explore_frac)
            if ranked_subset and any(item.action.family == "source_patch" for item in ranked_subset):
                ranked_subset = _select_distinct_patch_families(ranked_subset, rank_limit)
            patch_budget_cfg = candidate_policy.get("patch_budgets", {}) if isinstance(candidate_policy, dict) else {}
            max_patches_per_round = int(patch_budget_cfg.get("max_patches_per_round", 0) or 0) if isinstance(patch_budget_cfg, dict) else 0
            if max_patches_per_round > 0 and ranked_subset:
                filtered_subset = []
                patch_count = 0
                dropped_patch_actions: List[str] = []
                for item in ranked_subset:
                    if item.action.family == "source_patch":
                        if patch_count >= max_patches_per_round:
                            dropped_patch_actions.append(item.action.action_id)
                            continue
                        patch_count += 1
                    filtered_subset.append(item)
                if dropped_patch_actions:
                    trace_events.append(
                        {
                            "event": "patch_budget_enforced",
                            "agent": "RouterRankerAgent",
                            "iteration": iteration,
                            "max_patches_per_round": max_patches_per_round,
                            "dropped_patch_actions": dropped_patch_actions,
                        }
                    )
                ranked_subset = filtered_subset
            ranked_actions = [item.action for item in ranked_subset]
            if neighbor_batch:
                ranked_actions = sorted(neighbor_batch_actions, key=lambda action: action.action_id)
                trace_events.append(
                    {
                        "event": "neighbor_tune_batch_selected",
                        "agent": "RouterRankerAgent",
                        "iteration": iteration,
                        "ranked_action_ids": [action.action_id for action in ranked_actions],
                    }
                )
        backend_exp = _latest_backend_exp(memory.experiments)
        backend_needed = any(
            _is_parallel_threads_action(action) for action in ranked_actions
        ) and not _run_args_has_backend(job.run_args) and backend_exp is None
        if backend_needed:
            backend_action = next(
                (action for action in ranked_actions if action.family == "runtime_backend_select"),
                None,
            )
            if backend_action is None:
                backend_action = next(
                    (action for action in actions if action.family == "runtime_backend_select"),
                    None,
                )
            if backend_action is not None:
                ranked_actions = [backend_action]
                trace_events.append(
                    {
                        "event": "backend_bootstrap",
                        "agent": "RouterRankerAgent",
                        "iteration": iteration,
                        "action_id": backend_action.action_id,
                        "reason": "parallel_omp requires backend selection first",
                    }
                )
        if not orchestrator_only:
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
                reporter.agent_trace("OptimizerAgent", getattr(optimizer, "last_llm_trace", None))
                reporter.rank_summary(ranked)
                reporter.agent_trace("RouterRankerAgent", getattr(ranker, "last_llm_trace", None))
        else:
            trace_events.append(
                {
                    "event": "ranked_actions",
                    "agent": "OrchestratorAgent",
                    "iteration": iteration,
                    "ranked_action_ids": [action.action_id for action in ranked_actions],
                    "rejected_action_ids": [],
                    "scoring_notes": "orchestrator",
                }
            )
        if use_orchestrator_decision and decision and not orchestrator_only:
            override_actions = _select_actions_from_decision(
                eligible_pool, decision, top_k, action_by_cid=decision_action_by_cid
            )
            if override_actions:
                ranked_actions = override_actions
                trace_events.append(
                    {
                        "event": "orchestrator_override",
                        "agent": "OrchestratorAgent",
                        "iteration": iteration,
                        "ranked_action_ids": [a.action_id for a in ranked_actions],
                    }
                )
                if reporter:
                    reporter.candidates(
                        actions=ranked_actions,
                        ranking_mode="orchestrator",
                        selection_mode=selection_mode,
                        llm_explanation=decision.reason or None,
                    )
        if not ranked_actions:
            if (
                patch_no_candidates
                and scope_levels
                and state.patch_scope_level < len(scope_levels) - 1
            ):
                old_scope = scope_levels[state.patch_scope_level]
                state.patch_scope_level += 1
                state.patch_scope_no_candidates = 0
                state.blocked_families.discard("source_patch")
                trace_events.append(
                    {
                        "event": "source_patch_scope_promoted",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "from_scope": old_scope,
                        "to_scope": scope_levels[state.patch_scope_level],
                        "reason": "no_candidates_in_scope",
                    }
                )
                if reporter:
                    reporter.skip("source_patch scope 提升: 无可用候选")
                continue
            if reporter:
                reporter.stop("无可执行候选")
            break

        iteration_experiments: List[ExperimentIR] = []
        candidate_repeats = max(1, plan.evaluation.candidate_repeats_stage1)
        variance_cfg = gates.get("variance", {}) if isinstance(gates, dict) else {}
        variance_repeats = int(variance_cfg.get("repeats", 2) or 2)
        frozen_build_exp = (
            _find_experiment_by_run_id(memory.experiments, frozen_build_id)
            if frozen_build_id
            else None
        )
        patch_failures: List[str] = []
        patch_blocked = False
        source_patch_attempted = 0
        attempted_source_actions: Dict[str, Optional[str]] = {}
        source_patch_context_miss_actions: set[str] = set()
        for action in ranked_actions:
            if state.run_count >= job.budgets.max_runs:
                break
            effective_repeats = candidate_repeats
            if _is_volatile_action(action):
                effective_repeats = max(effective_repeats, variance_repeats)
            exp_id = _iter_exp_id(iteration, action.action_id)
            base_build_cfg = build_cfg or {}
            if frozen_build_exp and frozen_build_exp.action:
                base_build_cfg = _apply_build_config(
                    base_build_cfg, frozen_build_exp.action, build_packs
                )
            refine_on_best = _should_refine_on_best(
                baseline_exp=baseline_exp,
                best_exp=memory.best,
                enabled=refine_enabled,
                min_improvement_pct=refine_min_improvement,
            )
            if action and "source_patch" in (action.applies_to or []):
                source_patch_attempted += 1
                attempted_source_actions[action.action_id] = (
                    (action.parameters or {}).get("deep_analysis_id")
                    if isinstance(action.parameters, dict)
                    else None
                )
                if action.parameters is None:
                    action.parameters = {}
                if not action.parameters.get("patch_root"):
                    adapter_root = (adapter_cfg or {}).get("patch_root")
                    if isinstance(adapter_root, str) and adapter_root:
                        action.parameters["patch_root"] = adapter_root
                action_allowed_files = list(patch_allowed_files)
                target_file = (action.parameters or {}).get("target_file")
                target_anchor = (action.parameters or {}).get("target_anchor")
                patch_family = (action.parameters or {}).get("patch_family")

                # Normalize target_file to repo-relative path
                if isinstance(target_file, str) and target_file:
                    norm_tf = _normalise_to_repo_rel(target_file)
                    if norm_tf:
                        target_file = norm_tf
                if isinstance(target_anchor, str):
                    target_anchor = target_anchor.strip()

                # Build per-action navigation hints (filter to target_file if specified)
                action_nav_hints = list(nav_hints)
                if isinstance(target_file, str) and target_file:
                    target_related_files = _target_with_related_files(
                        repo_root,
                        target_file,
                        patch_allowed_files,
                        max_related=max(2, min(max_snippets or 8, 8)),
                    )
                    action_allowed_files = target_related_files or [target_file]
                    target_hints = [
                        h for h in action_nav_hints if h.get("path") in set(action_allowed_files)
                    ]
                    if target_hints:
                        action_nav_hints = target_hints
                    else:
                        # Target file not in global hints — build a hint for it
                        rel = target_file
                        _tf_path = repo_root / rel
                        if _tf_path.is_file():
                            try:
                                _tf_lines = _tf_path.read_text(encoding="utf-8", errors="replace").splitlines()
                                _tf_hint: Dict[str, object] = {"path": rel, "total_lines": len(_tf_lines)}
                                _tf_marker, _ = _find_marker_line(_tf_lines)
                                if _tf_marker is not None:
                                    _tf_hint["hotspot_line"] = _tf_marker + 1
                                    _tf_func = _find_function_start(_tf_lines, _tf_marker)
                                    if _tf_func is not None:
                                        _tf_hint["function_start"] = _tf_func + 1
                                        _tf_hint["function_signature"] = _tf_lines[_tf_func].strip()[:120]
                                action_nav_hints = [_tf_hint]
                            except Exception:
                                pass

                # Lazy code snippet generation for non-agentic fallback paths
                # (debug agent, NEED_MORE_CONTEXT expansion, worktree retries).
                # The agentic patcher uses navigation hints + read_file instead.
                _debug_snippet_files = (
                    action_allowed_files
                    if isinstance(target_file, str) and target_file
                    else snippet_files
                )
                base_context = int(patch_rules.get("max_context_chars", 0) or 0)
                if base_context <= 0:
                    base_context = 60000
                code_snippets: Optional[List[Dict[str, object]]] = None
                action_code_snippets_by_root: Dict[str, List[Dict[str, object]]] = {}

                def _get_debug_snippets(source_root: Optional[Path] = None) -> List[Dict[str, object]]:
                    nonlocal code_snippets
                    snippet_root = source_root or repo_root
                    cache_key = str(snippet_root.resolve())
                    if cache_key not in action_code_snippets_by_root:
                        action_code_snippets_by_root[cache_key] = _collect_code_snippets(
                            snippet_root, _debug_snippet_files, max_snippets, base_context,
                        )
                    snippets_for_root = action_code_snippets_by_root[cache_key]
                    if source_root is None:
                        code_snippets = snippets_for_root
                    return snippets_for_root
                _initial_patch_snippets = [] if use_agentic_patcher else _get_debug_snippets()
                if isinstance(action.parameters, dict) and not target_anchor:
                    inferred_anchor = _infer_target_anchor_for_action(
                        action=action,
                        repo_root=repo_root,
                        code_snippets=_initial_patch_snippets or _get_debug_snippets(),
                    )
                    if inferred_anchor:
                        action.parameters["target_anchor"] = inferred_anchor
                        target_anchor = inferred_anchor
                        trace_events.append(
                            {
                                "event": "target_anchor_inferred",
                                "agent": "Orchestrator",
                                "iteration": iteration,
                                "action_id": action.action_id,
                                "target_anchor": inferred_anchor,
                            }
                        )

                # Fetch reference template for algorithm-level families.
                _ref_template = None
                if isinstance(patch_family, str):
                    _ref_template = get_template_context(
                        patch_family,
                        backend=_backend_variant,
                        repo_root=repo_root,
                    )
                if _ref_template is None:
                    _ref_template = _fallback_reference_template_for_action(
                        action=action,
                        target_file=target_file if isinstance(target_file, str) else None,
                        target_anchor=target_anchor if isinstance(target_anchor, str) else None,
                    )
                # Guard against stale patch_path from a previous iteration.
                # When iter1 generates a patch, it mutates action.parameters["patch_path"]
                # (line 6209). If the same ActionIR object is reused in iter2 (via
                # generated_actions), iter2 would skip proposal and try to reapply
                # iter1's diff — which fails because the worktree is clean.
                # Only replay actions (origin=memory_replay) should carry patch_path.
                if not isinstance(action.parameters, dict):
                    action.parameters = {}
                _action_params = action.parameters
                _origin = _action_params.get("origin", "")
                if _origin != "memory_replay" and _action_params.get("patch_path"):
                    _stale = _action_params.pop("patch_path", None)
                    _action_params.pop("patch_files", None)
                    trace_events.append({
                        "event": "patch_path_stale_cleared",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "action_id": action.action_id,
                        "stale_path": str(_stale),
                    })
                if not _action_params.get("patch_path"):
                    # Build failure feedback string from previous iterations
                    _patch_feedback_str = None
                    if patch_failure_feedback:
                        _patch_feedback_str = (
                            "PREVIOUS FAILED PATCHES (do NOT repeat similar optimizations):\n"
                            + "\n".join(patch_failure_feedback[-10:])
                        )
                    patch_proposal = code_patcher.propose(
                        action=action,
                        profile=profile_ref,
                        patch_rules=patch_rules,
                        allowed_files=action_allowed_files,
                        code_snippets=_initial_patch_snippets,
                        repo_root=repo_root,
                        feedback=_patch_feedback_str,
                        backend_variant=_backend_variant,
                        reference_template=_ref_template,
                        navigation_hints=action_nav_hints,
                    )
                    if reporter:
                        note = None
                        if patch_proposal and patch_proposal.missing_fields:
                            note = "; ".join(patch_proposal.missing_fields)
                        _proposal_stats = _quick_diff_stats(
                            patch_proposal.patch_diff if patch_proposal else ""
                        )
                        reporter.patch_proposal(
                            action.action_id,
                            patch_proposal.status if patch_proposal else "NO_RESPONSE",
                            note,
                            diff_stats=_proposal_stats,
                        )
                    trace_events.append(
                        {
                            "event": "patch_llm_proposal",
                            "agent": "CodePatchAgent",
                            "iteration": iteration,
                            "action_id": action.action_id,
                            "proposal": patch_proposal.model_dump()
                            if patch_proposal
                            else None,
                            "raw_response": (
                                code_patcher.last_trace.get("response")
                                if (not patch_proposal and code_patcher.last_trace)
                                else None
                            ),
                        }
                    )
                    if not patch_proposal or patch_proposal.status != "OK":
                        reason = "patch_proposal_missing"
                        if patch_proposal:
                            reason = patch_proposal.status
                            if patch_proposal.missing_fields:
                                reason = f"{reason}: {', '.join(patch_proposal.missing_fields)}"
                        if not patch_proposal or patch_proposal.status == "NO_RESPONSE":
                            patch_proposal = code_patcher.propose(
                                action=action,
                                profile=profile_ref,
                                patch_rules=patch_rules,
                                allowed_files=action_allowed_files,
                                code_snippets=_initial_patch_snippets,
                                repo_root=repo_root,
                                feedback="no_response_retry: respond with a valid structured patch",
                                backend_variant=_backend_variant,
                                reference_template=_ref_template,
                                navigation_hints=action_nav_hints,
                            )
                            trace_events.append(
                                {
                                    "event": "patch_llm_retry",
                                    "agent": "CodePatchAgent",
                                    "iteration": iteration,
                                    "action_id": action.action_id,
                                    "proposal": patch_proposal.model_dump()
                                    if patch_proposal
                                    else None,
                                }
                            )
                            if patch_proposal and patch_proposal.status == "OK":
                                reason = "OK"
                        patch_failures.append(reason)
                        if patch_proposal and patch_proposal.status == "NEED_MORE_CONTEXT":
                            state.patch_action_context_misses[action.action_id] = (
                                state.patch_action_context_misses.get(action.action_id, 0) + 1
                            )
                            expanded_snippets = list(_get_debug_snippets())
                            last_fingerprint = _snippet_fingerprint(expanded_snippets)
                            expand_budget = base_context
                            expand_cap = max(base_context * 4, 240000)
                            max_expand_rounds = int(patch_rules.get("max_context_expand_rounds", 0) or 3)
                            expand_round = 0
                            while patch_proposal and patch_proposal.status == "NEED_MORE_CONTEXT":
                                expand_round += 1
                                if expand_round > max_expand_rounds:
                                    break
                                _missing_fields = patch_proposal.missing_fields or []
                                if (
                                    isinstance(action.parameters, dict)
                                    and any("target_anchor" in field for field in _missing_fields)
                                ):
                                    inferred_anchor = _infer_target_anchor_for_action(
                                        action=action,
                                        repo_root=repo_root,
                                        code_snippets=expanded_snippets or _get_debug_snippets(),
                                    )
                                    if inferred_anchor:
                                        action.parameters["target_anchor"] = inferred_anchor
                                        target_anchor = inferred_anchor
                                expand_budget = min(expand_budget + base_context, expand_cap)
                                targets = _parse_need_more_context_targets(
                                    _missing_fields, repo_root
                                )
                                snippet_files_for_expand = action_allowed_files or snippet_files
                                if not snippet_files_for_expand:
                                    snippet_files_for_expand = [
                                        item.get("path")
                                        for item in _get_debug_snippets()
                                        if item.get("path")
                                    ]
                                if isinstance(target_file, str) and target_file:
                                    snippet_files_for_expand = [target_file]
                                expanded = _expand_snippets_for_missing_context(
                                    patch_proposal.missing_fields or [],
                                    repo_root,
                                    snippet_files_for_expand,
                                    patch_rules,
                                    max_chars_override=expand_budget,
                                )
                                if not expanded:
                                    expanded = _expand_snippets_for_file(
                                        repo_root=repo_root,
                                        file_path=snippet_files_for_expand[0]
                                        if snippet_files_for_expand
                                        else (snippet_files[0] if snippet_files else ""),
                                        max_snippets=max_snippets,
                                        max_chars=expand_cap,
                                    )
                                if not expanded:
                                    break
                                new_fingerprint = _snippet_fingerprint(expanded)
                                if new_fingerprint == last_fingerprint:
                                    expanded = _expand_snippets_for_file(
                                        repo_root=repo_root,
                                        file_path=snippet_files_for_expand[0]
                                        if snippet_files_for_expand
                                        else (snippet_files[0] if snippet_files else ""),
                                        max_snippets=max_snippets,
                                        max_chars=expand_cap,
                                    )
                                    new_fingerprint = _snippet_fingerprint(expanded)
                                    if new_fingerprint == last_fingerprint:
                                        break
                                last_fingerprint = new_fingerprint
                                expanded_snippets = expanded
                                trace_events.append(
                                    {
                                        "event": "code_snippets_expanded",
                                        "agent": "Orchestrator",
                                        "iteration": iteration,
                                        "round": expand_round,
                                        "file": targets[0][0] if targets else None,
                                        "snippets": [
                                            {
                                                "path": item.get("path"),
                                                "tag": item.get("tag"),
                                                "start_line": item.get("start_line"),
                                                "end_line": item.get("end_line"),
                                            }
                                            for item in expanded
                                        ],
                                    }
                                )
                                reason = patch_proposal.status
                                if patch_proposal.missing_fields:
                                    reason = f"{reason}: {', '.join(patch_proposal.missing_fields)}"
                                patch_proposal = code_patcher.propose(
                                    action=action,
                                    profile=profile_ref,
                                    patch_rules=patch_rules,
                                    allowed_files=action_allowed_files,
                                    code_snippets=expanded_snippets,
                                    repo_root=repo_root,
                                    feedback=reason,
                                    backend_variant=_backend_variant,
                                    reference_template=_ref_template,
                                )
                                trace_events.append(
                                    {
                                        "event": "patch_llm_retry",
                                        "agent": "CodePatchAgent",
                                        "iteration": iteration,
                                        "action_id": action.action_id,
                                        "proposal": patch_proposal.model_dump()
                                        if patch_proposal
                                        else None,
                                    }
                                )
                                if not patch_proposal or patch_proposal.status != "NEED_MORE_CONTEXT":
                                    break
                            if not patch_proposal or patch_proposal.status != "OK":
                                if patch_proposal and patch_proposal.status == "NEED_MORE_CONTEXT":
                                    source_patch_context_miss_actions.add(action.action_id)
                                state.blocked_actions.add(action.action_id)
                                continue
                        elif patch_proposal and "edit_apply_failed" in reason:
                            expanded = None
                            anchor_ctx = _parse_edit_failure_anchor(patch_proposal.missing_fields)
                            if anchor_ctx:
                                max_snippets = int(patch_rules.get("max_snippets", 0) or 0)
                                max_context = int(patch_rules.get("max_context_chars", 0) or 0)
                                expanded = _expand_snippets_for_anchor(
                                    repo_root=repo_root,
                                    file_path=anchor_ctx[0],
                                    anchor=anchor_ctx[1],
                                    max_snippets=max_snippets,
                                    max_chars=max_context,
                                )
                                if expanded:
                                    trace_events.append(
                                        {
                                            "event": "code_snippets_expanded",
                                            "agent": "Orchestrator",
                                            "iteration": iteration,
                                            "file": anchor_ctx[0],
                                            "snippets": [
                                                {
                                                    "path": item.get("path"),
                                                    "tag": item.get("tag"),
                                                    "start_line": item.get("start_line"),
                                                    "end_line": item.get("end_line"),
                                                }
                                                for item in expanded
                                            ],
                                        }
                                    )
                            patch_retry = code_patcher.propose(
                                action=action,
                                profile=profile_ref,
                                patch_rules=patch_rules,
                                allowed_files=action_allowed_files,
                                code_snippets=expanded or _get_debug_snippets(),
                                repo_root=repo_root,
                                feedback=reason,
                                backend_variant=_backend_variant,
                                reference_template=_ref_template,
                            )
                            trace_events.append(
                                {
                                    "event": "patch_llm_retry",
                                    "agent": "CodePatchAgent",
                                    "iteration": iteration,
                                    "action_id": action.action_id,
                                    "proposal": patch_retry.model_dump()
                                    if patch_retry
                                    else None,
                                }
                            )
                            if not patch_retry or patch_retry.status != "OK":
                                state.blocked_actions.add(action.action_id)
                                continue
                            patch_proposal = patch_retry
                        else:
                            state.blocked_actions.add(action.action_id)
                            continue
                    format_ok, format_reason = _check_patch_format(
                        repo_root, patch_proposal.patch_diff
                    )
                    if not format_ok:
                        patch_failures.append(f"patch_format_invalid: {format_reason}")
                        patch_retry = code_patcher.propose(
                            action=action,
                            profile=profile_ref,
                            patch_rules=patch_rules,
                            allowed_files=allowed_files,
                            code_snippets=_initial_patch_snippets,
                            repo_root=repo_root,
                            feedback=f"patch_format_invalid: {format_reason}",
                            backend_variant=_backend_variant,
                            reference_template=_ref_template,
                            navigation_hints=action_nav_hints,
                        )
                        if reporter:
                            note = None
                            if patch_retry and patch_retry.missing_fields:
                                note = "; ".join(patch_retry.missing_fields)
                            reporter.patch_proposal(
                                action.action_id,
                                patch_retry.status if patch_retry else "NO_RESPONSE",
                                note,
                            )
                        trace_events.append(
                            {
                                "event": "patch_llm_retry",
                                "agent": "CodePatchAgent",
                                "iteration": iteration,
                                "action_id": action.action_id,
                                "proposal": patch_retry.model_dump()
                                if patch_retry
                                else None,
                            }
                        )
                        if not patch_retry or patch_retry.status != "OK":
                            state.blocked_actions.add(action.action_id)
                            continue
                        patch_proposal = patch_retry
                        format_ok, format_reason = _check_patch_format(
                            repo_root, patch_proposal.patch_diff
                        )
                        if not format_ok:
                            patch_failures.append(f"patch_format_invalid: {format_reason}")
                            state.blocked_actions.add(action.action_id)
                            continue
                    det_ok, det_reasons, det_info = review_patch_diff(
                        patch_proposal.patch_diff, repo_root, patch_rules
                    )
                    trace_events.append(
                        {
                            "event": "patch_review_deterministic",
                            "agent": "PatchReview",
                            "iteration": iteration,
                            "action_id": action.action_id,
                            "ok": det_ok,
                            "reasons": det_reasons,
                            "info": det_info,
                        }
                    )
                    if not det_ok:
                        patch_failures.append(f"deterministic_review_failed: {', '.join(det_reasons)}")
                        state.blocked_actions.add(action.action_id)
                        if reporter:
                            reporter.patch_review(
                                action.action_id,
                                "deterministic",
                                "FAIL",
                                "; ".join(det_reasons) if det_reasons else None,
                                diff_stats=det_info,
                            )
                        continue
                    # Target file enforcement: if the action specifies target_files
                    # (e.g. from deep_analysis), verify the patch touches at least one.
                    _action_params = action.parameters or {}
                    _action_targets = _action_params.get("target_files", [])
                    if not _action_targets:
                        _single_target = _action_params.get("target_file")
                        if isinstance(_single_target, str) and _single_target:
                            _action_targets = [_single_target]
                    if isinstance(_action_targets, list) and _action_targets:
                        _patch_files = det_info.get("files", [])
                        _target_basenames = {
                            Path(t).name for t in _action_targets if isinstance(t, str)
                        }
                        _patch_basenames = {
                            Path(f).name for f in _patch_files if isinstance(f, str)
                        }
                        if _target_basenames and not (_target_basenames & _patch_basenames):
                            _miss_msg = (
                                f"patch targets {_patch_basenames or 'no files'} "
                                f"but action expects {_target_basenames}"
                            )
                            patch_failures.append(f"target_file_mismatch: {_miss_msg}")
                            state.blocked_actions.add(action.action_id)
                            trace_events.append({
                                "event": "patch_target_mismatch",
                                "agent": "PatchReview",
                                "iteration": iteration,
                                "action_id": action.action_id,
                                "reason": _miss_msg,
                            })
                            if reporter:
                                reporter.patch_review(
                                    action.action_id, "target_check", "FAIL", _miss_msg,
                                )
                            continue

                    if reporter:
                        reporter.patch_review(
                            action.action_id, "deterministic", "PASS",
                            diff_stats=det_info,
                        )
                    # Skip LLM review when deterministic review passes --
                    # LLM review added no value (only checked file paths and
                    # line counts) and frequently produced false negatives
                    # that triggered expensive retry loops.
                    llm_review = None
                    _skip_llm_review = patch_rules.get("skip_llm_review", True)
                    if not _skip_llm_review:
                        llm_review = patch_reviewer.review(
                            patch_diff=patch_proposal.patch_diff,
                            patch_rules=patch_rules,
                            context={
                                "action": action.action_id,
                                "family": action.family,
                                "risk_level": action.risk_level,
                                "expected_effect": action.expected_effect,
                                "profile": profile_ref.model_dump(),
                            },
                        )
                    if llm_review:
                        trace_events.append(
                            {
                                "event": "patch_review_llm",
                                "agent": "PatchReviewAgent",
                                "iteration": iteration,
                                "action_id": action.action_id,
                                "review": llm_review.model_dump(),
                            }
                        )
                        if reporter:
                            reason = "; ".join(llm_review.reasons) if llm_review.reasons else None
                            reporter.patch_review(
                                action.action_id,
                                "llm",
                                f"{llm_review.status} {llm_review.verdict}",
                                reason,
                            )
                        if llm_review.status != "OK" or llm_review.verdict != "PASS":
                            patch_failures.append(
                                f"llm_review_failed: {llm_review.status} {llm_review.verdict}"
                            )
                            patched = _retry_patch_after_review(
                                patch_debugger=patch_debugger,
                                patch_reviewer=patch_reviewer,
                                action=action,
                                profile=profile_ref,
                                patch_rules=patch_rules,
                                allowed_files=action_allowed_files,
                                code_snippets=_collect_code_snippets(repo_root, snippet_files, max_snippets, max_context),
                                repo_root=repo_root,
                                patch_proposal=patch_proposal,
                                review_reasons=llm_review.reasons or [],
                                debug_max_attempts=debug_max_attempts,
                                snippet_files=snippet_files,
                                iteration=iteration,
                                trace_events=trace_events,
                                reporter=reporter,
                            )
                            if not patched:
                                state.blocked_actions.add(action.action_id)
                                continue
                            patch_proposal = patched
                    patch_root_value = action.parameters.get("patch_root")
                    patch_root_path = Path(patch_root_value) if patch_root_value else None
                    apply_ok, apply_reason = _check_patch_apply(
                        repo_root, patch_proposal.patch_diff, patch_root_path
                    )
                    if reporter:
                        reporter.patch_apply_check(action.action_id, apply_ok, apply_reason)
                    trace_events.append(
                        {
                            "event": "patch_apply_check",
                            "agent": "PatchApplyCheck",
                            "iteration": iteration,
                            "action_id": action.action_id,
                            "ok": apply_ok,
                            "reason": apply_reason,
                        }
                    )
                    if not apply_ok:
                        patch_failures.append(f"patch_apply_failed: {apply_reason}")
                        state.blocked_actions.add(action.action_id)
                        patch_retry = code_patcher.propose(
                            action=action,
                            profile=profile_ref,
                            patch_rules=patch_rules,
                            allowed_files=allowed_files,
                            code_snippets=_initial_patch_snippets,
                            repo_root=repo_root,
                            feedback=f"patch_apply_failed: {apply_reason}",
                            backend_variant=_backend_variant,
                            reference_template=_ref_template,
                            navigation_hints=action_nav_hints,
                        )
                        if reporter:
                            note = None
                            if patch_retry and patch_retry.missing_fields:
                                note = "; ".join(patch_retry.missing_fields)
                            reporter.patch_proposal(
                                action.action_id,
                                patch_retry.status if patch_retry else "NO_RESPONSE",
                                note,
                            )
                        trace_events.append(
                            {
                                "event": "patch_llm_retry_apply",
                                "agent": "CodePatchAgent",
                                "iteration": iteration,
                                "action_id": action.action_id,
                                "proposal": patch_retry.model_dump()
                                if patch_retry
                                else None,
                            }
                        )
                        if not patch_retry or patch_retry.status != "OK":
                            state.blocked_actions.add(action.action_id)
                            continue
                        patch_proposal = patch_retry
                        format_ok, format_reason = _check_patch_format(
                            repo_root, patch_proposal.patch_diff
                        )
                        if not format_ok:
                            patch_failures.append(f"patch_format_invalid: {format_reason}")
                            state.blocked_actions.add(action.action_id)
                            continue
                        det_ok, det_reasons, det_info = review_patch_diff(
                            patch_proposal.patch_diff, repo_root, patch_rules
                        )
                        trace_events.append(
                            {
                                "event": "patch_review_deterministic",
                                "agent": "PatchReview",
                                "iteration": iteration,
                                "action_id": action.action_id,
                                "ok": det_ok,
                                "reasons": det_reasons,
                                "info": det_info,
                            }
                        )
                        if not det_ok:
                            patch_failures.append(
                                f"deterministic_review_failed: {', '.join(det_reasons)}"
                            )
                            state.blocked_actions.add(action.action_id)
                            if reporter:
                                reporter.patch_review(
                                    action.action_id,
                                    "deterministic",
                                    "FAIL",
                                    "; ".join(det_reasons) if det_reasons else None,
                                )
                            continue
                        if reporter:
                            reporter.patch_review(action.action_id, "deterministic", "PASS")
                        llm_review = patch_reviewer.review(
                            patch_diff=patch_proposal.patch_diff,
                            patch_rules=patch_rules,
                            context={
                                "action": action.action_id,
                                "family": action.family,
                                "risk_level": action.risk_level,
                                "expected_effect": action.expected_effect,
                                "profile": profile_ref.model_dump(),
                            },
                        )
                        if llm_review:
                            trace_events.append(
                                {
                                    "event": "patch_review_llm",
                                    "agent": "PatchReviewAgent",
                                    "iteration": iteration,
                                    "action_id": action.action_id,
                                    "review": llm_review.model_dump(),
                                }
                            )
                            if reporter:
                                reason = (
                                    "; ".join(llm_review.reasons)
                                    if llm_review.reasons
                                    else None
                                )
                                reporter.patch_review(
                                    action.action_id,
                                    "llm",
                                    f"{llm_review.status} {llm_review.verdict}",
                                    reason,
                                )
                            if llm_review.status != "OK" or llm_review.verdict != "PASS":
                                patch_failures.append(
                                    f"llm_review_failed: {llm_review.status} {llm_review.verdict}"
                                )
                                patched = _retry_patch_after_review(
                                    patch_debugger=patch_debugger,
                                    patch_reviewer=patch_reviewer,
                                    action=action,
                                    profile=profile_ref,
                                    patch_rules=patch_rules,
                                    allowed_files=action_allowed_files,
                                    code_snippets=_collect_code_snippets(repo_root, snippet_files, max_snippets, max_context),
                                    repo_root=repo_root,
                                    patch_proposal=patch_proposal,
                                    review_reasons=llm_review.reasons or [],
                                    debug_max_attempts=debug_max_attempts,
                                    snippet_files=snippet_files,
                                    iteration=iteration,
                                    trace_events=trace_events,
                                    reporter=reporter,
                                )
                                if not patched:
                                    state.blocked_actions.add(action.action_id)
                                    continue
                                patch_proposal = patched
                        apply_ok, apply_reason = _check_patch_apply(
                            repo_root, patch_proposal.patch_diff, patch_root_path
                        )
                        if reporter:
                            reporter.patch_apply_check(action.action_id, apply_ok, apply_reason)
                        trace_events.append(
                            {
                                "event": "patch_apply_check",
                                "agent": "PatchApplyCheck",
                                "iteration": iteration,
                                "action_id": action.action_id,
                                "ok": apply_ok,
                                "reason": apply_reason,
                            }
                        )
                        if not apply_ok:
                            patch_failures.append(f"patch_apply_failed: {apply_reason}")
                            state.blocked_actions.add(action.action_id)
                            continue
                    run_dir = artifacts_dir / "runs" / f"iter{iteration}-{action.action_id}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    patch_path = run_dir / "patch_llm.diff"
                    patch_path.write_text(patch_proposal.patch_diff, encoding="utf-8")
                    action.parameters["patch_path"] = str(patch_path)
                    if patch_proposal.touched_files:
                        action.parameters["patch_files"] = patch_proposal.touched_files
                    allowlist = policy.get("input_edit_allowlist", [])
                    if not allowlist:
                        allowlist = app_input_allowlist(job.app)
                    patch_root_value = action.parameters.get("patch_root")
                    patch_root_path = Path(patch_root_value) if patch_root_value else None
                    preflight_ok, preflight_reason, preflight_log = _preflight_patch_compile(
                        repo_root=repo_root,
                        exp_id=exp_id,
                        patch_path=patch_path,
                        patch_root=patch_root_path,
                        input_script=Path(job.input_script),
                        allowlist=allowlist,
                        build_cfg=base_build_cfg,
                        build_packs=build_packs,
                        action=action,
                        run_dir=run_dir,
                        worktree_retries=worktree_retry_attempts,
                    )
                    if reporter:
                        reporter.patch_preflight(action.action_id, preflight_ok, preflight_reason)
                    trace_events.append(
                        {
                            "event": "patch_preflight_build",
                            "agent": "PatchPreflight",
                            "iteration": iteration,
                            "action_id": action.action_id,
                            "ok": preflight_ok,
                            "reason": preflight_reason,
                            "build_log": preflight_log,
                        }
                    )
                    if not preflight_ok:
                        patch_failures.append(f"patch_preflight_build_failed: {preflight_reason}")
                        retry_on_preflight = bool(patch_rules.get("retry_on_preflight", True))
                        if not retry_on_preflight:
                            state.blocked_actions.add(action.action_id)
                            continue
                        patch_retry = code_patcher.propose(
                            action=action,
                            profile=profile_ref,
                            patch_rules=patch_rules,
                            allowed_files=allowed_files,
                            code_snippets=_initial_patch_snippets,
                            repo_root=repo_root,
                            feedback=f"compile_error:\n{preflight_reason}",
                            backend_variant=_backend_variant,
                            reference_template=_ref_template,
                            navigation_hints=action_nav_hints,
                        )
                        if reporter:
                            note = None
                            if patch_retry and patch_retry.missing_fields:
                                note = "; ".join(patch_retry.missing_fields)
                            reporter.patch_proposal(
                                action.action_id,
                                patch_retry.status if patch_retry else "NO_RESPONSE",
                                note,
                            )
                        trace_events.append(
                            {
                                "event": "patch_llm_retry_compile",
                                "agent": "CodePatchAgent",
                                "iteration": iteration,
                                "action_id": action.action_id,
                                "proposal": patch_retry.model_dump()
                                if patch_retry
                                else None,
                            }
                        )
                        if not patch_retry or patch_retry.status != "OK":
                            state.blocked_actions.add(action.action_id)
                            continue
                        patch_proposal = patch_retry
                        format_ok, format_reason = _check_patch_format(
                            repo_root, patch_proposal.patch_diff
                        )
                        if not format_ok:
                            patch_failures.append(f"patch_format_invalid: {format_reason}")
                            state.blocked_actions.add(action.action_id)
                            continue
                        det_ok, det_reasons, det_info = review_patch_diff(
                            patch_proposal.patch_diff, repo_root, patch_rules
                        )
                        trace_events.append(
                            {
                                "event": "patch_review_deterministic",
                                "agent": "PatchReview",
                                "iteration": iteration,
                                "action_id": action.action_id,
                                "ok": det_ok,
                                "reasons": det_reasons,
                                "info": det_info,
                            }
                        )
                        if not det_ok:
                            patch_failures.append(
                                f"deterministic_review_failed: {', '.join(det_reasons)}"
                            )
                            state.blocked_actions.add(action.action_id)
                            if reporter:
                                reporter.patch_review(
                                    action.action_id,
                                    "deterministic",
                                    "FAIL",
                                    "; ".join(det_reasons) if det_reasons else None,
                                )
                            continue
                        if reporter:
                            reporter.patch_review(action.action_id, "deterministic", "PASS")
                        llm_review = patch_reviewer.review(
                            patch_diff=patch_proposal.patch_diff,
                            patch_rules=patch_rules,
                            context={
                                "action": action.action_id,
                                "family": action.family,
                                "risk_level": action.risk_level,
                                "expected_effect": action.expected_effect,
                                "profile": profile_ref.model_dump(),
                            },
                        )
                        if llm_review:
                            trace_events.append(
                                {
                                    "event": "patch_review_llm",
                                    "agent": "PatchReviewAgent",
                                    "iteration": iteration,
                                    "action_id": action.action_id,
                                    "review": llm_review.model_dump(),
                                }
                            )
                            if reporter:
                                reason = (
                                    "; ".join(llm_review.reasons)
                                    if llm_review.reasons
                                    else None
                                )
                                reporter.patch_review(
                                    action.action_id,
                                    "llm",
                                    f"{llm_review.status} {llm_review.verdict}",
                                    reason,
                                )
                            if llm_review.status != "OK" or llm_review.verdict != "PASS":
                                patch_failures.append(
                                    f"llm_review_failed: {llm_review.status} {llm_review.verdict}"
                                )
                                state.blocked_actions.add(action.action_id)
                                continue
                        patch_root_value = action.parameters.get("patch_root")
                        patch_root_path = Path(patch_root_value) if patch_root_value else None
                        apply_ok, apply_reason = _check_patch_apply(
                            repo_root, patch_proposal.patch_diff, patch_root_path
                        )
                        if reporter:
                            reporter.patch_apply_check(action.action_id, apply_ok, apply_reason)
                        trace_events.append(
                            {
                                "event": "patch_apply_check",
                                "agent": "PatchApplyCheck",
                                "iteration": iteration,
                                "action_id": action.action_id,
                                "ok": apply_ok,
                                "reason": apply_reason,
                            }
                        )
                        if not apply_ok:
                            patch_failures.append(f"patch_apply_failed: {apply_reason}")
                            state.blocked_actions.add(action.action_id)
                            continue
                        patch_path.write_text(patch_proposal.patch_diff, encoding="utf-8")
                        action.parameters["patch_path"] = str(patch_path)
                        if patch_proposal.touched_files:
                            action.parameters["patch_files"] = patch_proposal.touched_files
                        preflight_ok, preflight_reason, preflight_log = _preflight_patch_compile(
                            repo_root=repo_root,
                            exp_id=exp_id,
                            patch_path=patch_path,
                            patch_root=patch_root_path,
                            input_script=Path(job.input_script),
                            allowlist=allowlist,
                            build_cfg=base_build_cfg,
                            build_packs=build_packs,
                            action=action,
                            run_dir=run_dir,
                            worktree_retries=worktree_retry_attempts,
                        )
                        trace_events.append(
                            {
                                "event": "patch_preflight_build",
                                "agent": "PatchPreflight",
                                "iteration": iteration,
                                "action_id": action.action_id,
                                "ok": preflight_ok,
                                "reason": preflight_reason,
                                "build_log": preflight_log,
                            }
                        )
                        if not preflight_ok:
                            patch_failures.append(
                                f"patch_preflight_build_failed: {preflight_reason}"
                            )
                            debug_ok = False
                            if debug_max_attempts > 0 and preflight_log:
                                build_log_text = _summarize_build_log(Path(preflight_log))
                                debug_allowed_files = action_allowed_files
                                for attempt in range(debug_max_attempts):
                                    debug_ctx_exp_id = f"{exp_id}-preflight-debug{attempt + 1}"
                                    debug_ctx_dir = run_dir / "debug_context"
                                    debug_ctx_dir.mkdir(parents=True, exist_ok=True)
                                    try:
                                        with GitPatchContext(
                                            repo_root=repo_root,
                                            exp_id=debug_ctx_exp_id,
                                            artifacts_dir=debug_ctx_dir,
                                            input_script=Path(job.input_script),
                                            input_edit=None,
                                            allowlist=allowlist,
                                            patch_path=patch_path,
                                            patch_root=patch_root_path,
                                            worktree_retries=worktree_retry_attempts,
                                        ) as debug_ctx:
                                            debug_repo_root = debug_ctx.worktree_dir
                                            debug_snippets = _get_debug_snippets(debug_repo_root)
                                            debug_proposal = patch_debugger.repair(
                                                action=action,
                                                profile=profile_ref,
                                                patch_rules=patch_rules,
                                                allowed_files=debug_allowed_files,
                                                code_snippets=debug_snippets,
                                                repo_root=debug_repo_root,
                                                patch_diff=patch_proposal.patch_diff,
                                                build_log=build_log_text,
                                                feedback=preflight_reason,
                                            )
                                            if reporter:
                                                note = preflight_reason or ""
                                                if debug_proposal and debug_proposal.missing_fields:
                                                    note = "; ".join(debug_proposal.missing_fields)
                                                reporter.patch_debug(
                                                    action.action_id,
                                                    attempt + 1,
                                                    debug_proposal.status if debug_proposal else "NO_RESPONSE",
                                                    note,
                                                )
                                            trace_events.append(
                                                {
                                                    "event": "patch_debug_attempt",
                                                    "agent": "PatchDebugAgent",
                                                    "iteration": iteration,
                                                    "action_id": action.action_id,
                                                    "attempt": attempt + 1,
                                                    "proposal": debug_proposal.model_dump()
                                                    if debug_proposal
                                                    else None,
                                                }
                                            )
                                            if not debug_proposal:
                                                continue
                                            if debug_proposal.status == "NEED_MORE_CONTEXT":
                                                base_context = int(
                                                    patch_rules.get("max_context_chars", 0) or 0
                                                )
                                                if base_context <= 0:
                                                    base_context = 60000
                                                expand_budget = base_context
                                                expand_cap = max(base_context * 4, 240000)
                                                max_expand_rounds = int(
                                                    patch_rules.get("max_context_expand_rounds", 0) or 3
                                                )
                                                expand_round = 0
                                                last_fingerprint = _snippet_fingerprint(debug_snippets)
                                                while (
                                                    debug_proposal
                                                    and debug_proposal.status == "NEED_MORE_CONTEXT"
                                                ):
                                                    expand_round += 1
                                                    if expand_round > max_expand_rounds:
                                                        break
                                                    expand_budget = min(
                                                        expand_budget + base_context,
                                                        expand_cap,
                                                    )
                                                    expanded = _expand_snippets_for_missing_context(
                                                        debug_proposal.missing_fields,
                                                        debug_repo_root,
                                                        snippet_files,
                                                        patch_rules,
                                                        max_chars_override=expand_budget,
                                                    )
                                                    if not expanded:
                                                        break
                                                    new_fingerprint = _snippet_fingerprint(expanded)
                                                    if new_fingerprint == last_fingerprint:
                                                        break
                                                    last_fingerprint = new_fingerprint
                                                    snippet_paths = [
                                                        item.get("path")
                                                        for item in expanded
                                                        if item.get("path")
                                                    ]
                                                    debug_allowed_files = (
                                                        snippet_paths or debug_allowed_files
                                                    )
                                                    debug_snippets = expanded
                                                    trace_events.append(
                                                        {
                                                            "event": "code_snippets_expanded",
                                                            "agent": "Orchestrator",
                                                            "iteration": iteration,
                                                            "action_id": action.action_id,
                                                            "round": expand_round,
                                                            "snippets": [
                                                                {
                                                                    "path": item.get("path"),
                                                                    "tag": item.get("tag"),
                                                                    "start_line": item.get("start_line"),
                                                                    "end_line": item.get("end_line"),
                                                                }
                                                                for item in expanded
                                                            ],
                                                        }
                                                    )
                                                    debug_proposal = patch_debugger.repair(
                                                        action=action,
                                                        profile=profile_ref,
                                                        patch_rules=patch_rules,
                                                        allowed_files=debug_allowed_files,
                                                        code_snippets=debug_snippets,
                                                        repo_root=debug_repo_root,
                                                        patch_diff=patch_proposal.patch_diff,
                                                        build_log=build_log_text,
                                                        feedback=preflight_reason,
                                                    )
                                                    if reporter:
                                                        note = preflight_reason or ""
                                                        if (
                                                            debug_proposal
                                                            and debug_proposal.missing_fields
                                                        ):
                                                            note = "; ".join(
                                                                debug_proposal.missing_fields
                                                            )
                                                        reporter.patch_debug(
                                                            action.action_id,
                                                            attempt + 1,
                                                            debug_proposal.status
                                                            if debug_proposal
                                                            else "NO_RESPONSE",
                                                            note,
                                                        )
                                                    trace_events.append(
                                                        {
                                                            "event": "patch_debug_retry",
                                                            "agent": "PatchDebugAgent",
                                                            "iteration": iteration,
                                                            "action_id": action.action_id,
                                                            "attempt": attempt + 1,
                                                            "proposal": debug_proposal.model_dump()
                                                            if debug_proposal
                                                            else None,
                                                        }
                                                    )
                                                    if (
                                                        not debug_proposal
                                                        or debug_proposal.status != "NEED_MORE_CONTEXT"
                                                    ):
                                                        break
                                                if not debug_proposal or debug_proposal.status != "OK":
                                                    continue
                                            if debug_proposal.status != "OK":
                                                continue
                                            compose_ok, composed_patch = _compose_patch_from_debug_delta(
                                                worktree_root=debug_repo_root,
                                                debug_patch_diff=debug_proposal.patch_diff,
                                                patch_root=patch_root_path,
                                            )
                                            if not compose_ok:
                                                patch_failures.append(
                                                    f"patch_debug_compose_failed: {composed_patch}"
                                                )
                                                continue
                                            debug_proposal.patch_diff = composed_patch
                                    except Exception as exc:
                                        patch_failures.append(
                                            f"patch_debug_worktree_failed: {exc}"
                                        )
                                        trace_events.append(
                                            {
                                                "event": "patch_debug_worktree_failed",
                                                "agent": "GitPatchContext",
                                                "iteration": iteration,
                                                "action_id": action.action_id,
                                                "attempt": attempt + 1,
                                                "reason": str(exc),
                                            }
                                        )
                                        continue
                                    patch_proposal = debug_proposal
                                    format_ok, format_reason = _check_patch_format(
                                        repo_root, patch_proposal.patch_diff
                                    )
                                    if not format_ok:
                                        patch_failures.append(
                                            f"patch_format_invalid: {format_reason}"
                                        )
                                        continue
                                    det_ok, det_reasons, det_info = review_patch_diff(
                                        patch_proposal.patch_diff, repo_root, patch_rules
                                    )
                                    trace_events.append(
                                        {
                                            "event": "patch_review_deterministic",
                                            "agent": "PatchReview",
                                            "iteration": iteration,
                                            "action_id": action.action_id,
                                            "ok": det_ok,
                                            "reasons": det_reasons,
                                            "info": det_info,
                                        }
                                    )
                                    if not det_ok:
                                        patch_failures.append(
                                            f"deterministic_review_failed: {', '.join(det_reasons)}"
                                        )
                                        continue
                                    llm_review = patch_reviewer.review(
                                        patch_diff=patch_proposal.patch_diff,
                                        patch_rules=patch_rules,
                                        context={
                                            "action": action.action_id,
                                            "family": action.family,
                                            "risk_level": action.risk_level,
                                            "expected_effect": action.expected_effect,
                                            "profile": profile_ref.model_dump(),
                                        },
                                    )
                                    if llm_review:
                                        trace_events.append(
                                            {
                                                "event": "patch_review_llm",
                                                "agent": "PatchReviewAgent",
                                                "iteration": iteration,
                                                "action_id": action.action_id,
                                                "review": llm_review.model_dump(),
                                            }
                                        )
                                        if (
                                            llm_review.status != "OK"
                                            or llm_review.verdict != "PASS"
                                        ):
                                            patch_failures.append(
                                                f"llm_review_failed: {llm_review.status} {llm_review.verdict}"
                                            )
                                            continue
                                    apply_ok, apply_reason = _check_patch_apply(
                                        repo_root, patch_proposal.patch_diff, patch_root_path
                                    )
                                    trace_events.append(
                                        {
                                            "event": "patch_apply_check",
                                            "agent": "PatchApplyCheck",
                                            "iteration": iteration,
                                            "action_id": action.action_id,
                                            "ok": apply_ok,
                                            "reason": apply_reason,
                                        }
                                    )
                                    if not apply_ok:
                                        patch_failures.append(
                                            f"patch_apply_failed: {apply_reason}"
                                        )
                                        continue
                                    patch_path.write_text(
                                        patch_proposal.patch_diff, encoding="utf-8"
                                    )
                                    action.parameters["patch_path"] = str(patch_path)
                                    if patch_proposal.touched_files:
                                        action.parameters["patch_files"] = (
                                            patch_proposal.touched_files
                                        )
                                    preflight_ok, preflight_reason, preflight_log = (
                                        _preflight_patch_compile(
                                            repo_root=repo_root,
                                            exp_id=exp_id,
                                            patch_path=patch_path,
                                            patch_root=patch_root_path,
                                            input_script=Path(job.input_script),
                                            allowlist=allowlist,
                                            build_cfg=base_build_cfg,
                                            build_packs=build_packs,
                                            action=action,
                                            run_dir=run_dir,
                                            worktree_retries=worktree_retry_attempts,
                                        )
                                    )
                                    if reporter:
                                        reporter.patch_preflight(
                                            action.action_id,
                                            preflight_ok,
                                            preflight_reason,
                                        )
                                    trace_events.append(
                                        {
                                            "event": "patch_preflight_build",
                                            "agent": "PatchPreflight",
                                            "iteration": iteration,
                                            "action_id": action.action_id,
                                            "ok": preflight_ok,
                                            "reason": preflight_reason,
                                            "build_log": preflight_log,
                                        }
                                    )
                                    if preflight_ok:
                                        debug_ok = True
                                        break
                                    state.patch_action_preflight_fails[action.action_id] = (
                                        state.patch_action_preflight_fails.get(action.action_id, 0) + 1
                                    )
                                    _pfam = (action.parameters or {}).get("patch_family")
                                    if not isinstance(_pfam, str) or not _pfam:
                                        _pfam = action.family
                                    state.patch_family_preflight_fails[_pfam] = (
                                        state.patch_family_preflight_fails.get(_pfam, 0) + 1
                                    )
                                    _pf_block_threshold = int(
                                        patch_rules.get("family_preflight_block_threshold", 4) or 4
                                    )
                                    if state.patch_family_preflight_fails[_pfam] >= _pf_block_threshold:
                                        state.patch_family_blocked_until[f"source_patch:{_pfam}"] = iteration + 1
                                if not debug_ok:
                                    state.blocked_actions.add(action.action_id)
                                    continue
                            else:
                                state.blocked_actions.add(action.action_id)
                                continue
            patch_ok, patch_reason = validate_patch_action(
                action, patch_families or {}, gates, False
            )
            if not patch_ok:
                trace_events.append(
                    {
                        "event": "patch_triage_reject",
                        "agent": "PatchTriage",
                        "iteration": iteration,
                        "action_id": action.action_id,
                        "reason": patch_reason,
                    }
                )
                continue
            base_exp = best_chain_exp
            base_job = base_exp.job
            base_run_id = base_exp.run_id
            base_action_id = base_exp.action.action_id if base_exp.action else "baseline"
            verify_baseline = base_exp
            base_patch_paths = _collect_source_patch_chain_paths(base_exp, memory.experiments)
            _attach_patch_stack_to_action(action, base_patch_paths)
            _write_generated_actions(
                artifacts_dir / "runs" / exp_id,
                generated_actions,
                generated_ideas_payload,
                code_survey_payload,
            )
            exp = executor.execute(
                exp_id=exp_id,
                job=job,
                base_job=base_job,
                base_run_id=base_run_id,
                base_action_id=base_action_id,
                action=action,
                actions_root=repo_root,
                policy=policy,
                gates=gates,
                profiler=profiler,
                verifier=verifier,
                artifacts_dir=artifacts_dir,
                time_command=time_command,
                profiling_cfg=profiling_cfg,
                wrappers_cfg=wrappers_cfg,
                build_cfg=base_build_cfg,
                build_packs=build_packs,
                adapter_cfg=adapter_cfg,
                repeats=effective_repeats,
                runtime_agg="mean",
                baseline_exp=baseline_exp,
                baseline_exp_for_verify=verify_baseline,
                baseline_runtime=baseline_exp.results.runtime_seconds,
                prior_samples=None,
                trace_events=trace_events,
                parent_run_id=base_run_id,
                iteration=iteration,
                llm_trace=None,
                reporter=reporter,
                arg_rules=arg_rules_state,
            )
            debug_exp = None
            if (
                exp.verdict == "FAIL"
                and action
                and "source_patch" in (action.applies_to or [])
                and debug_max_attempts > 0
            ):
                reason_text = " ".join(exp.reasons or [])
                build_failed = (
                    "Build command failed" in reason_text
                    or "build did not produce lammps binary" in reason_text
                )
                runtime_failed = (
                    "nonzero exit code" in reason_text
                    or "runtime error" in reason_text.lower()
                    or "signal" in reason_text.lower()
                )
                run_dir = artifacts_dir / "runs" / exp.run_id
                build_log_path = run_dir / "build.log"
                stderr_log_path = run_dir / "stderr.log"
                time_log_path = run_dir / "time.log"
                patch_path_value = action.parameters.get("patch_path")
                patch_root_value = action.parameters.get("patch_root")
                patch_root_path = Path(patch_root_value) if patch_root_value else None
                if (build_failed or runtime_failed) and patch_path_value:
                    patch_path = Path(patch_path_value)
                    if patch_path.exists():
                        debug_log_parts: List[str] = []
                        if build_failed and build_log_path.exists():
                            build_summary = _summarize_build_log(build_log_path)
                            if build_summary:
                                debug_log_parts.append(f"build.log:\n{build_summary}")
                        if runtime_failed:
                            for label, path in (
                                ("stderr.log", stderr_log_path),
                                ("time.log", time_log_path),
                            ):
                                if not path.exists():
                                    continue
                                runtime_summary = _summarize_build_log(
                                    path, max_lines=160, max_chars=5000
                                )
                                if runtime_summary:
                                    debug_log_parts.append(f"{label}:\n{runtime_summary}")
                            if not debug_log_parts and build_log_path.exists():
                                fallback_summary = _summarize_build_log(build_log_path)
                                if fallback_summary:
                                    debug_log_parts.append(f"build.log:\n{fallback_summary}")
                        current_reason_text = reason_text
                        current_debug_log_text = "\n\n".join(debug_log_parts).strip() or reason_text
                        debug_allowed_files = action_allowed_files
                        for attempt in range(debug_max_attempts):
                            current_patch_value = action.parameters.get("patch_path")
                            current_patch_path = (
                                Path(current_patch_value)
                                if isinstance(current_patch_value, str) and current_patch_value
                                else patch_path
                            )
                            if not current_patch_path.exists():
                                patch_failures.append(
                                    f"patch_debug_missing_patch_file: {current_patch_path}"
                                )
                                continue
                            patch_diff = current_patch_path.read_text(
                                encoding="utf-8", errors="replace"
                            )
                            debug_ctx_exp_id = f"{exp_id}-runtime-debug{attempt + 1}"
                            debug_ctx_dir = run_dir / "debug_context"
                            debug_ctx_dir.mkdir(parents=True, exist_ok=True)
                            try:
                                with GitPatchContext(
                                    repo_root=repo_root,
                                    exp_id=debug_ctx_exp_id,
                                    artifacts_dir=debug_ctx_dir,
                                    input_script=Path(job.input_script),
                                    input_edit=None,
                                    allowlist=allowlist,
                                    patch_path=current_patch_path,
                                    patch_root=patch_root_path,
                                    worktree_retries=worktree_retry_attempts,
                                ) as debug_ctx:
                                    debug_repo_root = debug_ctx.worktree_dir
                                    debug_snippets = _get_debug_snippets(debug_repo_root)
                                    debug_proposal = patch_debugger.repair(
                                        action=action,
                                        profile=profile_ref,
                                        patch_rules=patch_rules,
                                        allowed_files=debug_allowed_files,
                                        code_snippets=debug_snippets,
                                        repo_root=debug_repo_root,
                                        patch_diff=patch_diff,
                                        build_log=current_debug_log_text,
                                        feedback=current_reason_text,
                                    )
                                    if reporter:
                                        note = current_reason_text
                                        if debug_proposal and debug_proposal.missing_fields:
                                            note = "; ".join(debug_proposal.missing_fields)
                                        reporter.patch_debug(
                                            action.action_id,
                                            attempt + 1,
                                            debug_proposal.status if debug_proposal else "NO_RESPONSE",
                                            note,
                                        )
                                    trace_events.append(
                                        {
                                            "event": "patch_debug_attempt",
                                            "agent": "PatchDebugAgent",
                                            "iteration": iteration,
                                            "action_id": action.action_id,
                                            "attempt": attempt + 1,
                                            "proposal": debug_proposal.model_dump()
                                            if debug_proposal
                                            else None,
                                        }
                                    )
                                    if not debug_proposal:
                                        continue
                                    if debug_proposal.status == "NEED_MORE_CONTEXT":
                                        base_context = int(
                                            patch_rules.get("max_context_chars", 0) or 0
                                        )
                                        if base_context <= 0:
                                            base_context = 60000
                                        expand_budget = base_context
                                        expand_cap = max(base_context * 4, 240000)
                                        max_expand_rounds = int(
                                            patch_rules.get("max_context_expand_rounds", 0) or 3
                                        )
                                        expand_round = 0
                                        last_fingerprint = _snippet_fingerprint(debug_snippets)
                                        while (
                                            debug_proposal
                                            and debug_proposal.status == "NEED_MORE_CONTEXT"
                                        ):
                                            expand_round += 1
                                            if expand_round > max_expand_rounds:
                                                break
                                            expand_budget = min(
                                                expand_budget + base_context,
                                                expand_cap,
                                            )
                                            expanded = _expand_snippets_for_missing_context(
                                                debug_proposal.missing_fields,
                                                debug_repo_root,
                                                snippet_files,
                                                patch_rules,
                                                max_chars_override=expand_budget,
                                            )
                                            if not expanded:
                                                break
                                            new_fingerprint = _snippet_fingerprint(expanded)
                                            if new_fingerprint == last_fingerprint:
                                                break
                                            last_fingerprint = new_fingerprint
                                            snippet_paths = [
                                                item.get("path")
                                                for item in expanded
                                                if item.get("path")
                                            ]
                                            debug_allowed_files = snippet_paths or debug_allowed_files
                                            debug_snippets = expanded
                                            trace_events.append(
                                                {
                                                    "event": "code_snippets_expanded",
                                                    "agent": "Orchestrator",
                                                    "iteration": iteration,
                                                    "action_id": action.action_id,
                                                    "round": expand_round,
                                                    "snippets": [
                                                        {
                                                            "path": item.get("path"),
                                                            "tag": item.get("tag"),
                                                            "start_line": item.get("start_line"),
                                                            "end_line": item.get("end_line"),
                                                        }
                                                        for item in expanded
                                                    ],
                                                }
                                            )
                                            debug_proposal = patch_debugger.repair(
                                                action=action,
                                                profile=profile_ref,
                                                patch_rules=patch_rules,
                                                allowed_files=debug_allowed_files,
                                                code_snippets=debug_snippets,
                                                repo_root=debug_repo_root,
                                                patch_diff=patch_diff,
                                                build_log=current_debug_log_text,
                                                feedback=current_reason_text,
                                            )
                                            if reporter:
                                                note = current_reason_text
                                                if (
                                                    debug_proposal
                                                    and debug_proposal.missing_fields
                                                ):
                                                    note = "; ".join(
                                                        debug_proposal.missing_fields
                                                    )
                                                reporter.patch_debug(
                                                    action.action_id,
                                                    attempt + 1,
                                                    debug_proposal.status
                                                    if debug_proposal
                                                    else "NO_RESPONSE",
                                                    note,
                                                )
                                            trace_events.append(
                                                {
                                                    "event": "patch_debug_retry",
                                                    "agent": "PatchDebugAgent",
                                                    "iteration": iteration,
                                                    "action_id": action.action_id,
                                                    "attempt": attempt + 1,
                                                    "proposal": debug_proposal.model_dump()
                                                    if debug_proposal
                                                    else None,
                                                }
                                            )
                                            if (
                                                not debug_proposal
                                                or debug_proposal.status != "NEED_MORE_CONTEXT"
                                            ):
                                                break
                                        if not debug_proposal or debug_proposal.status != "OK":
                                            continue
                                    if debug_proposal.status != "OK":
                                        continue
                                    compose_ok, composed_patch = _compose_patch_from_debug_delta(
                                        worktree_root=debug_repo_root,
                                        debug_patch_diff=debug_proposal.patch_diff,
                                        patch_root=patch_root_path,
                                    )
                                    if not compose_ok:
                                        patch_failures.append(
                                            f"patch_debug_compose_failed: {composed_patch}"
                                        )
                                        continue
                                    debug_proposal.patch_diff = composed_patch
                            except Exception as exc:
                                patch_failures.append(f"patch_debug_worktree_failed: {exc}")
                                trace_events.append(
                                    {
                                        "event": "patch_debug_worktree_failed",
                                        "agent": "GitPatchContext",
                                        "iteration": iteration,
                                        "action_id": action.action_id,
                                        "attempt": attempt + 1,
                                        "reason": str(exc),
                                    }
                                )
                                continue
                            debug_exp_id = f"{exp_id}-debug{attempt + 1}"
                            debug_run_dir = artifacts_dir / "runs" / debug_exp_id
                            debug_run_dir.mkdir(parents=True, exist_ok=True)
                            debug_patch_path = debug_run_dir / f"patch.debug{attempt + 1}.diff"
                            debug_patch_path.write_text(
                                debug_proposal.patch_diff, encoding="utf-8"
                            )
                            action.parameters["patch_path"] = str(debug_patch_path)
                            action.parameters.pop("patch_paths", None)
                            _attach_patch_stack_to_action(action, base_patch_paths)
                            if debug_proposal.touched_files:
                                action.parameters["patch_files"] = debug_proposal.touched_files
                            debug_exp = executor.execute(
                                exp_id=debug_exp_id,
                                job=job,
                                base_job=base_job,
                                base_run_id=base_run_id,
                                base_action_id=base_action_id,
                                action=action,
                                actions_root=repo_root,
                                policy=policy,
                                gates=gates,
                                profiler=profiler,
                                verifier=verifier,
                                artifacts_dir=artifacts_dir,
                                time_command=time_command,
                                profiling_cfg=profiling_cfg,
                                wrappers_cfg=wrappers_cfg,
                                build_cfg=base_build_cfg,
                                build_packs=build_packs,
                                adapter_cfg=adapter_cfg,
                                repeats=effective_repeats,
                                runtime_agg="mean",
                                baseline_exp=baseline_exp,
                                baseline_exp_for_verify=verify_baseline,
                                baseline_runtime=baseline_exp.results.runtime_seconds,
                                prior_samples=None,
                                trace_events=trace_events,
                                parent_run_id=base_run_id,
                                iteration=iteration,
                                llm_trace=None,
                                reporter=reporter,
                                arg_rules=arg_rules_state,
                            )
                            if debug_exp.verdict == "PASS":
                                break
                            debug_reason_text = " ".join(debug_exp.reasons or []).strip()
                            if debug_reason_text:
                                current_reason_text = debug_reason_text
                            latest_run_dir = artifacts_dir / "runs" / debug_exp_id
                            latest_build_log = latest_run_dir / "build.log"
                            latest_stderr_log = latest_run_dir / "stderr.log"
                            latest_time_log = latest_run_dir / "time.log"
                            latest_log_parts: List[str] = []
                            if latest_build_log.exists():
                                latest_build_summary = _summarize_build_log(latest_build_log)
                                if latest_build_summary:
                                    latest_log_parts.append(
                                        f"build.log:\n{latest_build_summary}"
                                    )
                            for label, path in (
                                ("stderr.log", latest_stderr_log),
                                ("time.log", latest_time_log),
                            ):
                                if not path.exists():
                                    continue
                                latest_runtime_summary = _summarize_build_log(
                                    path, max_lines=160, max_chars=5000
                                )
                                if latest_runtime_summary:
                                    latest_log_parts.append(
                                        f"{label}:\n{latest_runtime_summary}"
                                    )
                            if latest_log_parts:
                                current_debug_log_text = "\n\n".join(latest_log_parts).strip()
                            else:
                                current_debug_log_text = current_reason_text
                        if debug_exp and debug_exp.verdict != "PASS":
                            debug_exp = None
            memory.record(exp)
            # Use iteration_baseline for marginal gain (improvement vs state before this action)
            experience_memory.record_experiment(exp, iteration_baseline_exp)
            iteration_experiments.append(exp)
            state.run_count += 1
            if debug_exp:
                memory.record(debug_exp)
                experience_memory.record_experiment(debug_exp, iteration_baseline_exp)
                iteration_experiments.append(debug_exp)
                state.run_count += 1
            final_exp = debug_exp or exp
            if final_exp.verdict == "PASS" and best_chain_runtime > 0.0:
                if _is_volatile_action(final_exp.action):
                    variance_cv = _extract_variance_cv(final_exp)
                    samples = final_exp.results.samples or []
                    cv_max = float(variance_cfg.get("cv_max", 1.0) or 1.0)
                    if variance_cv is None or len(samples) < variance_repeats or variance_cv > cv_max:
                        trace_events.append(
                            {
                                "event": "best_chain_skip_unstable",
                                "agent": "Orchestrator",
                                "run_id": final_exp.run_id,
                                "action_id": final_exp.action.action_id if final_exp.action else "baseline",
                                "variance_cv": variance_cv,
                                "samples": len(samples),
                                "cv_max": cv_max,
                            }
                        )
                        continue
            if final_exp.verdict == "FAIL":
                state.fail_count += 1
                failure = triage.classify(final_exp, artifacts_dir / "runs" / final_exp.run_id)
                trace_events.append(
                    {
                        "event": "triage",
                        "agent": "TriageAgent",
                        "run_id": final_exp.run_id,
                        "summary": failure.model_dump(),
                    }
                )
            _append_run_index(
                artifacts_dir, exp, parent_run_id=base_run_id, iteration=iteration
            )
            if debug_exp:
                _append_run_index(
                    artifacts_dir,
                    debug_exp,
                    parent_run_id=base_run_id,
                    iteration=iteration,
                )

        # --- Track deep analysis opportunity outcomes ---
        if attempted_source_actions:
            executed_action_ids = {
                exp.action.action_id for exp in iteration_experiments if exp.action
            }
            context_miss_fail_threshold = int(
                patch_rules.get("context_miss_fail_threshold", 2) or 2
            )
            for action_id, da_id in attempted_source_actions.items():
                if action_id in executed_action_ids:
                    continue
                if not isinstance(da_id, str) or not da_id:
                    continue
                if action_id in source_patch_context_miss_actions:
                    miss_count = int(state.patch_action_context_misses.get(action_id, 0) or 0)
                    if miss_count < context_miss_fail_threshold:
                        trace_events.append(
                            {
                                "event": "deep_analysis_action_deferred",
                                "agent": "Orchestrator",
                                "iteration": iteration,
                                "action_id": action_id,
                                "deep_analysis_id": da_id,
                                "reason": "need_more_context",
                                "miss_count": miss_count,
                                "threshold": context_miss_fail_threshold,
                            }
                        )
                        continue
                deep_failed_ids.add(da_id)
                trace_events.append(
                    {
                        "event": "deep_analysis_action_failed",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "action_id": action_id,
                        "deep_analysis_id": da_id,
                        "reason": "no_experiment_emitted",
                    }
                )

        failures: List[Dict[str, object]] = []
        for exp in iteration_experiments:
            feedback = _collect_failure_feedback(exp, artifacts_dir)
            if feedback:
                failures.append(feedback)
        decision_feedback = {"failures": failures}

        for exp in iteration_experiments:
            da_id = (exp.action.parameters or {}).get("deep_analysis_id") if exp.action else None
            if not da_id:
                continue
            if exp.verdict == "PASS":
                deep_succeeded_ids.add(da_id)
            else:
                deep_failed_ids.add(da_id)

        # --- Post-loop: select non-conflicting improvements and compose ---
        winners = _select_non_conflicting(
            iteration_experiments,
            best_chain_runtime,
            chain_min_improvement_pct,
            variance_cfg=variance_cfg,
            variance_repeats=variance_repeats,
            baseline_run_id=best_chain_exp.run_id if best_chain_exp else None,
        )

        if len(winners) == 1:
            # Single best improvement — update best_chain directly
            best_chain_exp = winners[0]
            best_chain_runtime = winners[0].results.runtime_seconds
            memory.best = winners[0]
            trace_events.append(
                {
                    "event": "best_chain_update",
                    "agent": "Orchestrator",
                    "run_id": winners[0].run_id,
                    "action_id": winners[0].action.action_id if winners[0].action else "baseline",
                    "improvement_pct": (
                        (prev_best_time - winners[0].results.runtime_seconds) / prev_best_time
                        if prev_best_time > 0
                        else 0.0
                    ),
                    "composition": "single",
                }
            )
        elif len(winners) > 1:
            # Multiple non-conflicting improvements — try composing them
            composite_action = _build_final_composite_action(winners, gates)
            composed = False
            if composite_action and state.run_count < job.budgets.max_runs:
                composite_action.action_id = f"iter{iteration}_composite"
                composite_action.family = "iter_composite"
                composite_action.description = (
                    f"Composite of {len(winners)} non-conflicting improvements: "
                    + ", ".join(w.action.action_id for w in winners if w.action)
                )
                trace_events.append(
                    {
                        "event": "iter_composite_start",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "components": [w.action.action_id for w in winners if w.action],
                    }
                )
                composite_exp_id = f"iter{iteration}-composite"
                composite_base = winners[0]  # Build from the fastest single winner
                composite_exp = executor.execute(
                    exp_id=composite_exp_id,
                    job=job,
                    base_job=composite_base.job,
                    base_run_id=composite_base.run_id,
                    base_action_id=composite_base.action.action_id if composite_base.action else "baseline",
                    action=composite_action,
                    actions_root=repo_root,
                    policy=policy,
                    gates=gates,
                    profiler=profiler,
                    verifier=verifier,
                    artifacts_dir=artifacts_dir,
                    time_command=time_command,
                    profiling_cfg=profiling_cfg,
                    wrappers_cfg=wrappers_cfg,
                    build_cfg=build_cfg or {},
                    build_packs=build_packs,
                    adapter_cfg=adapter_cfg,
                    repeats=max(1, plan.evaluation.candidate_repeats_stage1) if plan else 1,
                    runtime_agg="mean",
                    baseline_exp=baseline_exp,
                    baseline_exp_for_verify=iteration_baseline_exp,
                    baseline_runtime=baseline_exp.results.runtime_seconds,
                    prior_samples=None,
                    trace_events=trace_events,
                    parent_run_id=composite_base.run_id,
                    iteration=iteration,
                    llm_trace=None,
                    reporter=reporter,
                    arg_rules=arg_rules_state,
                )
                memory.record(composite_exp)
                experience_memory.record_experiment(composite_exp, iteration_baseline_exp)
                state.run_count += 1
                _append_run_index(
                    artifacts_dir,
                    composite_exp,
                    parent_run_id=composite_base.run_id,
                    iteration=iteration,
                )
                if (
                    composite_exp.verdict == "PASS"
                    and composite_exp.results.runtime_seconds < winners[0].results.runtime_seconds
                ):
                    # Composite is better than single best — use it
                    best_chain_exp = composite_exp
                    best_chain_runtime = composite_exp.results.runtime_seconds
                    memory.best = composite_exp
                    composed = True
                    trace_events.append(
                        {
                            "event": "best_chain_update",
                            "agent": "Orchestrator",
                            "run_id": composite_exp.run_id,
                            "action_id": composite_exp.action.action_id if composite_exp.action else "composite",
                            "improvement_pct": (
                                (prev_best_time - composite_exp.results.runtime_seconds) / prev_best_time
                                if prev_best_time > 0
                                else 0.0
                            ),
                            "composition": "composite",
                            "components": [w.action.action_id for w in winners if w.action],
                        }
                    )
                else:
                    trace_events.append(
                        {
                            "event": "iter_composite_fallback",
                            "agent": "Orchestrator",
                            "iteration": iteration,
                            "composite_verdict": composite_exp.verdict,
                            "composite_runtime": composite_exp.results.runtime_seconds,
                            "single_best_runtime": winners[0].results.runtime_seconds,
                        }
                    )
                    if composite_exp.verdict == "FAIL":
                        state.fail_count += 1

            if not composed:
                # Composite failed or wasn't attempted — fall back to single best
                best_chain_exp = winners[0]
                best_chain_runtime = winners[0].results.runtime_seconds
                memory.best = winners[0]
                trace_events.append(
                    {
                        "event": "best_chain_update",
                        "agent": "Orchestrator",
                        "run_id": winners[0].run_id,
                        "action_id": winners[0].action.action_id if winners[0].action else "baseline",
                        "improvement_pct": (
                            (prev_best_time - winners[0].results.runtime_seconds) / prev_best_time
                            if prev_best_time > 0
                            else 0.0
                        ),
                        "composition": "single_fallback",
                        "attempted_components": [w.action.action_id for w in winners if w.action],
                    }
                )

        if neighbor_batch:
            state.blocked_families.add("neighbor_tune")
            state.neighbor_tune_done = True
            trace_events.append(
                {
                    "event": "neighbor_tune_batch_complete",
                    "agent": "Orchestrator",
                    "iteration": iteration,
                    "blocked": True,
                }
            )
        elif (
            not state.neighbor_tune_done
            and any(
                exp.action and exp.action.family == "neighbor_tune"
                for exp in (iteration_experiments or [])
            )
        ):
            state.neighbor_tune_done = True
            state.blocked_families.add("neighbor_tune")
            trace_events.append(
                {
                    "event": "neighbor_tune_locked",
                    "agent": "Orchestrator",
                    "iteration": iteration,
                    "blocked": True,
                }
            )

        if not iteration_experiments and source_patch_attempted and patch_failures:
            if opportunity_graph_mode:
                trace_events.append(
                    {
                        "event": "source_patch_generation_failed",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "reason": "graph action(s) produced no runnable patch",
                        "failed_action_ids": sorted(attempted_source_actions.keys()),
                        "details": patch_failures[:3],
                    }
                )
                if reporter:
                    reporter.skip("source_patch 当前候选未生成可用补丁，切换下一个机会节点")
            else:
                patch_blocked = True
                state.blocked_families.add("source_patch")
                trace_events.append(
                    {
                        "event": "source_patch_blocked",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "reason": "no valid patch generated",
                        "details": patch_failures[:3],
                    }
                )
                if reporter:
                    reporter.skip("source_patch 暂停: 未生成可用补丁")

        # --- Accumulate patch failure feedback for next iteration ---
        if phase_started == "PATCH" and iteration_experiments:
            for exp in iteration_experiments:
                if (
                    exp.action
                    and exp.action.family in ("source_patch", "iter_composite")
                    and exp.results.runtime_seconds > best_chain_runtime
                ):
                    slowdown = (
                        (exp.results.runtime_seconds - best_chain_runtime)
                        / best_chain_runtime * 100
                    )
                    pfam = (exp.action.parameters or {}).get("patch_family", "unknown")
                    patch_failure_feedback.append(
                        f"iter{iteration} {exp.action.action_id} "
                        f"(family={pfam}): +{slowdown:.1f}% SLOWER. "
                        f"Description: {exp.action.description[:120]}"
                    )

        llm_iteration_summary = None
        if (
            llm_summary_enabled
            and iteration_experiments
            and llm_client
            and llm_client.config.enabled
        ):
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
                best_combo = _effective_action_combo(memory.experiments, best_chain_exp)
                reporter.iteration_summary(
                    iteration,
                    _best_pass_exp(iteration_experiments),
                    best_chain_exp,
                    best_combo,
                )

        # --- Reflection step (batch results → reprioritise remaining opportunities) ---
        if (
            phase == "PATCH"
            and reflection_agent
            and deep_analysis_opportunities
            and iteration_experiments
        ):
            _baseline_rt = baseline_exp.results.runtime_seconds if baseline_exp else 0.0
            _cumulative_gain = (
                (_baseline_rt - best_chain_runtime) / _baseline_rt * 100.0
                if _baseline_rt > 0 else 0.0
            )
            _batch_results = []
            for exp in iteration_experiments:
                _da_id = (
                    (exp.action.parameters or {}).get("deep_analysis_id")
                    if exp.action else None
                )
                if not _da_id:
                    continue
                _gain = (
                    (_baseline_rt - exp.results.runtime_seconds) / _baseline_rt * 100.0
                    if _baseline_rt > 0 else 0.0
                )
                _batch_results.append({
                    "opportunity_id": _da_id,
                    "verdict": exp.verdict,
                    "gain_pct": round(_gain, 3),
                    "failure_reason": (
                        "; ".join(exp.reasons) if exp.reasons else ""
                    ),
                })
            if _batch_results:
                _remaining_opps = []
                for _opp_action in deep_analysis_opportunities:
                    _p = _opp_action.parameters or {}
                    _remaining_opps.append({
                        "opportunity_id": _p.get("deep_analysis_id", ""),
                        "title": _opp_action.description,
                        "category": _p.get("patch_family", ""),
                        "family_hint": _p.get("patch_family"),
                        "estimated_impact": _p.get("implementation_complexity", "medium"),
                    })
                _reflection_payload = {
                    "iteration": iteration,
                    "cumulative_gain_pct": round(_cumulative_gain, 3),
                    "batch_results": _batch_results,
                    "remaining_opportunities": _remaining_opps,
                    "history_summary": history_summary,
                    "strategy_rationale": (
                        _deep_analysis_result.strategy_rationale
                        if _deep_analysis_result else ""
                    ),
                }
                _refl = reflection_agent.reflect(_reflection_payload)
                if _refl:
                    last_reflection = _refl
                    if reporter:
                        reporter.agent_trace(
                            "ReflectionAgent",
                            getattr(reflection_agent, "last_llm_trace", None),
                        )
                    # 1. Add skip_ids
                    deep_skipped_ids.update(_refl.skip_ids)
                    # 2. Reprioritise remaining opportunities
                    if _refl.reprioritized_ids:
                        _opp_map = {
                            (a.parameters or {}).get("deep_analysis_id", ""): a
                            for a in deep_analysis_opportunities
                        }
                        _reordered: List[ActionIR] = []
                        for _rid in _refl.reprioritized_ids:
                            if _rid in _opp_map and _rid not in deep_skipped_ids:
                                _reordered.append(_opp_map.pop(_rid))
                        # Append any remaining that weren't mentioned
                        for _a in deep_analysis_opportunities:
                            _aid = (_a.parameters or {}).get("deep_analysis_id", "")
                            if _aid in _opp_map and _aid not in deep_skipped_ids:
                                _reordered.append(_a)
                        deep_analysis_opportunities = _reordered
                    trace_events.append({
                        "event": "reflection",
                        "agent": "ReflectionAgent",
                        "iteration": iteration,
                        "reprioritized_ids": _refl.reprioritized_ids,
                        "skip_ids": _refl.skip_ids,
                        "strategy_note": _refl.strategy_note,
                        "direction_hint": _refl.direction_hint,
                    })
                    if reporter:
                        reporter._print(
                            f"Reflection: reprioritized {len(_refl.reprioritized_ids)} "
                            f"opportunities, skipped {len(_refl.skip_ids)}"
                        )

        best_improved = bool(
            best_chain_exp
            and best_chain_exp.results.runtime_seconds + min_delta_seconds < prev_best_time
        )
        if best_improved:
            prev_best_time = best_chain_exp.results.runtime_seconds

        if scope_levels:
            patch_gain = False
            if source_patch_attempted and iteration_experiments:
                for exp in iteration_experiments:
                    if not exp.action or exp.action.family != "source_patch":
                        continue
                    if exp.verdict != "PASS":
                        continue
                    if exp.results.runtime_seconds + min_delta_seconds < prev_best_time_before:
                        patch_gain = True
                        break
            if source_patch_attempted:
                if patch_gain:
                    state.patch_scope_no_gain_iters = 0
                else:
                    state.patch_scope_no_gain_iters += 1
                if patch_no_candidates:
                    state.patch_scope_no_candidates += 1
                else:
                    state.patch_scope_no_candidates = 0
                if patch_failures and not iteration_experiments:
                    state.patch_scope_failures += 1
                else:
                    state.patch_scope_failures = 0
            else:
                state.patch_scope_no_gain_iters = 0
                state.patch_scope_no_candidates = 0
                state.patch_scope_failures = 0

            promote_cfg = patch_scope_cfg.get("promote", {}) if patch_scope_cfg else {}
            no_gain_iters = int(promote_cfg.get("no_gain_iters", 0) or 0)
            no_candidates_iters = int(promote_cfg.get("no_candidates_iters", 0) or 0)
            patch_failures_limit = int(promote_cfg.get("patch_failures", 0) or 0)
            should_promote = False
            if no_gain_iters and state.patch_scope_no_gain_iters >= no_gain_iters:
                should_promote = True
            if no_candidates_iters and state.patch_scope_no_candidates >= no_candidates_iters:
                should_promote = True
            if patch_failures_limit and state.patch_scope_failures >= patch_failures_limit:
                should_promote = True
            if should_promote and state.patch_scope_level < len(scope_levels) - 1:
                old_scope = scope_levels[min(state.patch_scope_level, len(scope_levels) - 1)]
                state.patch_scope_level += 1
                state.patch_scope_no_gain_iters = 0
                state.patch_scope_no_candidates = 0
                state.patch_scope_failures = 0
                state.blocked_families.discard("source_patch")
                trace_events.append(
                    {
                        "event": "source_patch_scope_promoted",
                        "agent": "Orchestrator",
                        "iteration": iteration,
                        "from_scope": old_scope,
                        "to_scope": scope_levels[state.patch_scope_level],
                        "reason": "scope_promotion",
                    }
                )

        iteration_best = _best_pass_exp(iteration_experiments) if iteration_experiments else None
        iteration_family = (
            iteration_best.action.family
            if iteration_best and iteration_best.action and iteration_best.action.family
            else None
        )
        if iteration_best is None:
            pass
        elif state.last_family is None or iteration_family != state.last_family:
            state.no_improve_iters = 0
            state.last_family = iteration_family
        else:
            if best_improved and memory.best and memory.best.action and memory.best.action.family == iteration_family:
                state.no_improve_iters = 0
            else:
                state.no_improve_iters += 1

        tested_actions = {exp.action.action_id for exp in memory.experiments if exp.action}
        refreshed_actions = _prepare_actions(
            base_actions=actions,
            candidate_policy=candidate_policy,
            system_caps=system_caps,
            experiments=memory.experiments,
            adapter_cfg=adapter_cfg,
            job=job,
            fixed_threads=fixed_threads,
        )
        remaining_candidates = _remaining_candidates_by_family(refreshed_actions, tested_actions)
        if state.blocked_families:
            for family in list(state.blocked_families):
                remaining_candidates.pop(family, None)
        history_summary = _build_history_summary(memory.experiments)
        cost_model = _build_cost_model(memory.experiments)
        iteration_summaries = _collect_iteration_summaries(
            memory.experiments, baseline_exp.results.runtime_seconds
        )
        phase_order = _phase_order(hierarchical_cfg)
        phase_thresholds = _freeze_thresholds(hierarchical_cfg, phase)
        freeze_target = "runtime_tier" if phase in {"RUN_TUNE", "RUN_RETUNE"} else "build_config"
        best_phase_exp = _best_for_target(memory.experiments, freeze_target)
        if (
            phase == "RUN_TUNE"
            and best_phase_exp
            and best_phase_exp.action
            and best_phase_exp.action.family == "runtime_backend_select"
        ):
            best_phase_exp = None
        freeze_hit, freeze_reason = (False, "")
        if phase in {"RUN_TUNE", "BUILD_TUNE"} and phase_thresholds:
            freeze_hit, freeze_reason = _phase_freeze_decision(
                phase, baseline_exp, best_phase_exp, phase_thresholds, state
            )
        review_payload = {
            "iteration": iteration,
            "budgets": {
                "max_iters": job.budgets.max_iters,
                "max_runs": job.budgets.max_runs,
                "max_wall_seconds": job.budgets.max_wall_seconds,
                "remaining_iters": max(0, job.budgets.max_iters - iteration),
                "remaining_runs": max(0, job.budgets.max_runs - state.run_count),
            },
            "stop_state": {
                "run_count": state.run_count,
                "fail_count": state.fail_count,
                "no_improve_iters": state.no_improve_iters,
                "last_family": state.last_family,
            },
            "history_summary": history_summary,
            "cost_model": cost_model,
            "remaining_candidates": remaining_candidates,
            "iteration_summaries": iteration_summaries[-3:],
            "best_summary": _best_summary(baseline_exp, memory.best),
            "context": {
                "selection_mode": selection_mode,
                "phase": phase,
                "retune_remaining": retune_remaining,
                "tags": job.tags,
            },
            "phase_freeze": {
                "freeze_hit": freeze_hit,
                "freeze_reason": freeze_reason,
                "thresholds": phase_thresholds,
                "phase_order": phase_order,
            },
            "reflection": {
                "strategy_note": last_reflection.strategy_note if last_reflection else "",
                "direction_hint": last_reflection.direction_hint if last_reflection else "",
            },
        }
        review_decision = reviewer.review(review_payload)
        final_review = review_decision
        if patch_blocked and not review_decision.should_stop:
            final_review = review_decision.model_copy(
                update={
                    "suggested_next_step": "switch_family",
                    "reason": f"{review_decision.reason} | source_patch blocked".strip(" |"),
                }
            )
        if freeze_hit and not review_decision.should_stop:
            if review_decision.suggested_next_step != "switch_family":
                final_review = review_decision.model_copy(
                    update={
                        "suggested_next_step": "switch_family",
                        "reason": f"{review_decision.reason} | {freeze_reason}".strip(" |"),
                    }
                )
                trace_events.append(
                    {
                        "event": "phase_freeze_override",
                        "agent": "ReviewerAgent",
                        "iteration": iteration,
                        "reason": freeze_reason,
                    }
                )
        last_review_decision = final_review.model_dump()
        trace_events.append(
            {
                "event": "review_decision",
                "agent": "ReviewerAgent",
                "iteration": iteration,
                "decision": final_review.model_dump(),
                "payload": review_payload,
            }
        )
        if reporter:
            reporter.review_summary(final_review)
            reporter.agent_trace("ReviewerAgent", getattr(reviewer, "last_llm_trace", None))
        if final_review.should_stop:
            if reporter:
                reporter.stop(f"reviewer_stop: {final_review.reason}")
            break
        if final_review.suggested_next_step == "switch_family":
            if patch_only and phase == "PATCH":
                next_phase = "PATCH"
            else:
                next_phase = phase
                if phase == "RUN_TUNE":
                    best_run = _best_for_target(memory.experiments, "runtime_tier") or baseline_exp
                    frozen_run_id = best_run.run_id
                    _has_patch_families = bool(
                        patch_families and isinstance(patch_families, dict)
                        and patch_families.get("families")
                    )
                    next_phase = _next_phase_in_order(
                        phase,
                        phase_order,
                        has_build=_has_actions_for_target(actions, "build_config"),
                        has_patch=_has_actions_for_target(actions, "source_patch") or _has_patch_families,
                    )
                elif phase == "BUILD_TUNE":
                    best_build = _best_for_target(memory.experiments, "build_config")
                    if best_build:
                        frozen_build_id = best_build.run_id
                    next_phase = "RUN_RETUNE" if retune_remaining > 0 else "PATCH"
                    if next_phase == "RUN_RETUNE" and retune_min_improvement > 0.0:
                        improvement = _best_improvement_pct(baseline_exp, memory.best)
                        if improvement < retune_min_improvement:
                            retune_remaining = 0
                            next_phase = "PATCH"
                            trace_events.append(
                                {
                                    "event": "retune_skipped",
                                    "agent": "ReviewerAgent",
                                    "iteration": iteration,
                                    "reason": f"improvement {improvement:.3f} < {retune_min_improvement:.3f}",
                                }
                            )
                elif phase == "RUN_RETUNE":
                    best_run = _best_for_target(memory.experiments, "runtime_tier")
                    if best_run:
                        frozen_run_id = best_run.run_id
                    has_build = _has_actions_for_target(actions, "build_config")
                    _has_patch_families_rt = bool(
                        patch_families and isinstance(patch_families, dict)
                        and patch_families.get("families")
                    )
                    has_patch = _has_actions_for_target(actions, "source_patch") or _has_patch_families_rt
                    if retune_origin == "patch" and has_build:
                        next_phase = "BUILD_TUNE"
                    else:
                        next_phase = _next_phase_in_order(
                            "BUILD_TUNE",
                            phase_order,
                            has_build=has_build,
                            has_patch=has_patch,
                        )
                elif phase == "PATCH":
                    if use_two_phase:
                        # In two-phase mode, parameter tuning is complete;
                        # stay in PATCH for source-code optimization.
                        next_phase = "PATCH"
                    elif patch_blocked:
                        retune_remaining = max(retune_remaining, 1)
                        next_phase = "RUN_RETUNE"
                    elif post_patch_retune > 0:
                        retune_remaining = post_patch_retune
                        next_phase = "RUN_RETUNE"
                        if retune_min_improvement > 0.0:
                            improvement = _best_improvement_pct(baseline_exp, memory.best)
                            if improvement < retune_min_improvement:
                                retune_remaining = 0
                                next_phase = "PATCH"
                                trace_events.append(
                                    {
                                        "event": "retune_skipped",
                                        "agent": "ReviewerAgent",
                                        "iteration": iteration,
                                        "reason": f"improvement {improvement:.3f} < {retune_min_improvement:.3f}",
                                    }
                                )
            if next_phase != phase:
                trace_events.append(
                    {
                        "event": "phase_transition",
                        "agent": "ReviewerAgent",
                        "iteration": iteration,
                        "from_phase": phase,
                        "to_phase": next_phase,
                        "frozen_run_id": frozen_run_id,
                        "frozen_build_id": frozen_build_id,
                        "reason": review_decision.reason,
                    }
                )
                if phase == "RUN_RETUNE":
                    retune_origin = None
                phase = next_phase
                if phase == "RUN_RETUNE":
                    retune_remaining = max(retune_remaining, 1)
                    if retune_origin is None:
                        retune_origin = "patch" if phase_started == "PATCH" else "build"
        if phase_started == "RUN_RETUNE" and retune_remaining > 0:
            retune_remaining -= 1
            if retune_remaining <= 0 and phase == "RUN_RETUNE":
                phase = "PATCH"

    validation_exp = None
    if (
        memory.best
        and memory.best.action
        and validate_top1_repeats > 0
        and state.run_count < job.budgets.max_runs
    ):
        repeats = validate_top1_repeats
        exp_id = f"{memory.best.exp_id}-validate"
        base_job, base_run_id, base_action_id = _select_validation_base(
            baseline_exp=baseline_exp,
            best_exp=memory.best,
            experiments=memory.experiments,
        )
        backend_exp = _latest_backend_exp(memory.experiments)
        verify_baseline = baseline_exp
        if memory.best and memory.best.action:
            if any(t in memory.best.action.applies_to for t in ["build_config", "source_patch"]):
                base_exp_for_verify = (
                    _find_experiment_by_run_id(memory.experiments, base_run_id)
                    if base_run_id
                    else None
                )
                if base_exp_for_verify:
                    verify_baseline = base_exp_for_verify
            elif backend_exp and _run_args_has_backend(base_job.run_args):
                verify_baseline = backend_exp
        validation_exp = executor.execute(
            exp_id=exp_id,
            job=job,
            base_job=base_job,
            base_run_id=base_run_id,
            base_action_id=base_action_id,
            action=memory.best.action,
            actions_root=repo_root,
            policy=policy,
            gates=gates,
            profiler=profiler,
            verifier=verifier,
            artifacts_dir=artifacts_dir,
            time_command=time_command,
            profiling_cfg=profiling_cfg,
            wrappers_cfg=wrappers_cfg,
            build_cfg=build_cfg or {},
            build_packs=build_packs,
            adapter_cfg=adapter_cfg,
            repeats=repeats,
            runtime_agg=baseline_stat,
            baseline_exp=baseline_exp,
            baseline_exp_for_verify=verify_baseline,
            baseline_runtime=baseline_exp.results.runtime_seconds,
            prior_samples=None,
            trace_events=trace_events,
            parent_run_id=base_run_id or memory.best.run_id,
            iteration=iteration,
            llm_trace=None,
            reporter=reporter,
            arg_rules=arg_rules_state,
        )
        memory.record(validation_exp)
        state.run_count += 1
        if validation_exp.verdict == "FAIL":
            state.fail_count += 1
        _append_run_index(
            artifacts_dir,
            validation_exp,
            parent_run_id=base_run_id or memory.best.run_id,
            iteration=iteration,
        )

    composite_exp = None
    baseline_runtime = baseline_exp.results.runtime_seconds
    best_exp = best_chain_exp
    if best_exp and best_exp.verdict == "PASS" and state.run_count < job.budgets.max_runs:
        exp_id = "final-composite"
        final_action = ActionIR(
            action_id="final_best",
            family="final_best",
            description="Final validation run for current best state.",
            applies_to=[],
            parameters={},
        )
        trace_events.append(
            {
                "event": "final_composite_start",
                "agent": "Orchestrator",
                "action_id": final_action.action_id,
                "components": [best_exp.action.action_id if best_exp.action else "baseline"],
            }
        )
        composite_exp = executor.execute(
            exp_id=exp_id,
            job=job,
            base_job=best_exp.job,
            base_run_id=best_exp.run_id,
            base_action_id=best_exp.action.action_id if best_exp.action else "baseline",
            action=final_action,
            actions_root=repo_root,
            policy=policy,
            gates=gates,
            profiler=profiler,
            verifier=verifier,
            artifacts_dir=artifacts_dir,
            time_command=time_command,
            profiling_cfg=profiling_cfg,
            wrappers_cfg=wrappers_cfg,
            build_cfg=build_cfg or {},
            build_packs=build_packs,
            adapter_cfg=adapter_cfg,
            repeats=1,
            runtime_agg=baseline_stat,
            baseline_exp=baseline_exp,
            baseline_exp_for_verify=baseline_exp,
            baseline_runtime=baseline_runtime,
            prior_samples=None,
            trace_events=trace_events,
            parent_run_id=best_exp.run_id,
            iteration=None,
            llm_trace=None,
            reporter=reporter,
            arg_rules=arg_rules_state,
        )
        memory.record(composite_exp)
        state.run_count += 1
        if composite_exp.verdict == "FAIL":
            state.fail_count += 1
        _append_run_index(
            artifacts_dir,
            composite_exp,
            parent_run_id=best_exp.run_id,
            iteration=None,
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
    llm_summary_zh = None
    if llm_summary_enabled:
        llm_summary_zh = _build_llm_summary_zh(
            memory.experiments, baseline_exp, memory.best, success_info, llm_client
        )
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
        candidate_policy=candidate_policy_summary,
        review_decision=last_review_decision,
        phase_transitions=_extract_phase_transitions(trace_events),
        composite_exp=composite_exp,
        min_improvement_pct=min_improvement_pct,
    )
    report_info["agent_trace"] = agent_trace_path
    best_run_exp = _best_for_target(memory.experiments, "runtime_tier")
    best_build_exp = _best_for_target(memory.experiments, "build_config")
    opportunity_graph_payload: Dict[str, object] = {}
    if _opportunity_graph:
        opportunity_graph_payload["summary"] = _opportunity_graph_summary(_opportunity_graph)
        if _selected_opportunities:
            opportunity_graph_payload["selection"] = _selected_opportunities.model_dump()
    best_state_path = _write_best_state(
        artifacts_dir=artifacts_dir,
        baseline_exp=baseline_exp,
        best_exp=memory.best,
        best_chain_exp=best_chain_exp,
        best_run_exp=best_run_exp,
        best_build_exp=best_build_exp,
        phase=phase,
        frozen_run_id=frozen_run_id,
        frozen_build_id=frozen_build_id,
        opportunity_graph=opportunity_graph_payload,
    )
    report_info["best_state"] = best_state_path
    if reporter:
        reporter.final(
            best=memory.best,
            report_md=report_info.get("report_md", ""),
            report_zh=report_info.get("report_zh"),
        )
    return report_info


def _resolve_worktree_path(path_str: str, repo_root: Path) -> str:
    """If *path_str* points inside a (possibly stale) git worktree, map it
    back to the equivalent path under *repo_root*."""
    wt_marker = "/worktrees/"
    idx = path_str.find(wt_marker)
    if idx == -1:
        return path_str
    # <prefix>/worktrees/<run_id>/<repo_relative_path>
    after = path_str[idx + len(wt_marker):]
    slash = after.find("/")
    if slash == -1:
        return path_str
    repo_rel = after[slash + 1:]
    candidate = repo_root / repo_rel
    if candidate.exists():
        return str(candidate)
    # Path does not exist under repo_root either; return original
    return path_str


def _normalize_worktree_paths(job_snapshot: "JobIR", repo_root: Path) -> "JobIR":
    """Return a copy of *job_snapshot* with workdir/input_script resolved out
    of stale worktree paths."""
    workdir = str(job_snapshot.workdir)
    input_script = str(job_snapshot.input_script)
    new_workdir = _resolve_worktree_path(workdir, repo_root)
    new_input = _resolve_worktree_path(input_script, repo_root)
    if new_workdir != workdir or new_input != input_script:
        job_snapshot = job_snapshot.model_copy(deep=True)
        job_snapshot.workdir = new_workdir
        job_snapshot.input_script = new_input
    return job_snapshot


def _run_experiment(
    exp_id: str,
    job: JobIR,
    base_job: Optional[JobIR],
    base_run_id: Optional[str],
    base_action_id: Optional[str],
    action: Optional[ActionIR],
    actions_root: Path,
    policy: Dict[str, object],
    gates: Dict[str, object],
    profiler: ProfilerAgent,
    verifier: VerifierAgent,
    artifacts_dir: Path,
    time_command: Optional[str],
    profiling_cfg: Optional[Dict[str, object]],
    wrappers_cfg: Optional[List[Dict[str, object]]],
    build_cfg: Dict[str, object],
    build_packs: Optional[Dict[str, object]],
    adapter_cfg: Optional[Dict[str, object]],
    repeats: int,
    runtime_agg: str,
    baseline_exp: Optional[ExperimentIR],
    baseline_exp_for_verify: Optional[ExperimentIR],
    baseline_runtime: Optional[float],
    prior_samples: Optional[List[float]],
    trace_events: Optional[List[Dict[str, object]]],
    parent_run_id: Optional[str],
    iteration: Optional[int],
    llm_trace: Optional[Dict[str, object]],
    reporter: Optional[ConsoleUI],
    arg_rules: Optional[List[Dict[str, object]]] = None,
    run_purpose: str = "score",
) -> ExperimentIR:
    run_id = exp_id
    run_dir = artifacts_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    base_job_snapshot = base_job or job
    # Normalize workdir/input_script if they point inside a stale worktree
    base_job_snapshot = _normalize_worktree_paths(base_job_snapshot, actions_root)
    env_overrides, run_args, launcher_cfg = _apply_run_config_action(
        base_job_snapshot, action, arg_rules=arg_rules
    )
    run_args = _ensure_log_path(run_args, run_dir, app=base_job_snapshot.app)
    run_kind = "baseline" if action is None else "experiment"
    run_purpose_norm = (run_purpose or "score").strip().lower()
    if run_purpose_norm == "profile":
        # Profile probes intentionally allow wrappers (TAU/xctrace/perf), and we
        # treat them like baseline-style runs for wrapper selection.
        run_kind = "baseline"
    wrapper_id = None
    if action and action.parameters:
        wrapper_id = action.parameters.get("wrapper_id") or action.parameters.get("wrapper")
    wrapper_command: Optional[List[str]] = None
    wrapper_env: Dict[str, str] = {}
    wrapper_allowed_exit_codes: List[int] = []
    if run_purpose_norm == "profile":
        wrapper_command, wrapper_env, wrapper_allowed_exit_codes = _resolve_wrappers(
            wrappers_cfg, run_kind, run_dir, wrapper_id
        )
    effective_profiling_cfg = profiling_cfg if run_purpose_norm == "profile" else None
    run_env_overrides = dict(env_overrides)
    run_env_overrides.update(wrapper_env)
    run_trace: List[Dict[str, object]] = []
    run_cmd = None
    build_cmds: Optional[List[str]] = None
    is_profile_baseline = run_purpose_norm == "profile"
    build_cfg_final = _apply_build_config(build_cfg, action, build_packs)
    if _requires_build(action):
        build_cmds = _format_build_commands(build_cfg_final, run_dir, actions_root)
        lammps_bin = Path(build_cfg_final.get("lammps_bin") or "lmp")
        run_bin = run_dir / "build" / lammps_bin
    else:
        run_bin = Path(base_job_snapshot.app_bin)
    run_cmd = _format_run_command(
        run_bin,
        run_args,
        run_env_overrides,
        launcher_cfg,
        wrapper_command=wrapper_command,
    )
    if reporter:
        reporter.run_start(
            exp_id=exp_id,
            action=action,
            env_overrides=run_env_overrides,
            run_args=run_args,
            base_run_id=base_run_id,
            base_action_id=base_action_id,
            run_cmd=run_cmd,
            build_cmds=build_cmds,
        )
    _append_trace(
        trace_events,
        run_trace,
        {
            "event": "experiment_start",
            "agent": "OptimizerAgent",
            "run_id": run_id,
            "action_id": action.action_id if action else "baseline",
            "base_run_id": base_run_id,
            "base_action_id": base_action_id,
            "run_purpose": run_purpose_norm,
            "env_overrides": run_env_overrides,
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
    workdir = Path(base_job_snapshot.workdir)
    input_script = Path(base_job_snapshot.input_script)
    source_root = actions_root

    build_output: Optional[BuildOutput] = None
    build_config_diff_path: Optional[str] = None
    binary_provenance: Optional[Dict[str, object]] = None
    repro_script_path: Optional[str] = None
    run_output = None
    baseline_check_output = None

    job_snapshot = base_job_snapshot.model_copy(deep=True)
    job_snapshot.env.update(env_overrides)
    job_snapshot.run_args = run_args

    input_edit = action.parameters.get("input_edit") if action and "input_script" in action.applies_to else None
    patch_params = _extract_patch_params(action, actions_root)
    requires_patch = bool(action and any(t in action.applies_to for t in ["input_script", "source_patch"]))
    allowlist = policy.get("input_edit_allowlist", [])
    if not allowlist:
        allowlist = app_input_allowlist(job.app)

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
            base_run_id=base_run_id,
            base_action_id=base_action_id,
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
            build_seconds=None,
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
            base_run_id=base_run_id,
            base_action_id=base_action_id,
            base_job=base_job_snapshot,
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
            repro_script_path=None,
        )
        _write_run_trace(run_dir, run_trace)
        return exp

    worktree_retries = 2
    if isinstance(adapter_cfg, dict):
        patch_rules = adapter_cfg.get("patch_rules")
        if isinstance(patch_rules, dict):
            worktree_retries = int(
                patch_rules.get("worktree_retry_attempts", patch_rules.get("debug_max_attempts", 2))
                or 2
            )

    ctx_mgr = (
        GitPatchContext(
            repo_root=actions_root,
            exp_id=exp_id,
            artifacts_dir=run_dir,
            input_script=input_script,
            input_edit=input_edit,
            allowlist=allowlist,
            patch_path=patch_params.get("patch_path"),
            patch_paths=patch_params.get("patch_paths"),
            patch_root=patch_params.get("patch_root"),
            worktree_retries=worktree_retries,
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
                workdir = ctx.map_to_worktree(Path(base_job_snapshot.workdir))
                input_script = ctx.map_to_worktree(input_script)
                job_snapshot.workdir = str(workdir)
                job_snapshot.input_script = str(input_script)
                source_root = ctx.worktree_dir
                # Ensure untracked workdir exists inside worktree (git worktrees omit untracked files).
                if not workdir.exists():
                    orig_workdir = Path(base_job_snapshot.workdir)
                    try:
                        if orig_workdir.is_dir():
                            workdir.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copytree(orig_workdir, workdir, dirs_exist_ok=True)
                        elif orig_workdir.is_file():
                            workdir.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(orig_workdir, workdir)
                    except Exception:
                        pass
                synced_inputs = _sync_runtime_inputs_to_worktree(
                    app=base_job_snapshot.app,
                    orig_workdir=Path(base_job_snapshot.workdir),
                    workdir=workdir,
                    run_args=run_args,
                )
                if synced_inputs:
                    _append_trace(
                        trace_events,
                        run_trace,
                        {
                            "event": "worktree_runtime_input_sync",
                            "agent": "ProfilerAgent",
                            "run_id": run_id,
                            "count": len(synced_inputs),
                            "paths": synced_inputs[:20],
                        },
                    )
                if action and "input_script" in (action.applies_to or []):
                    snapshot_path = run_dir / "input_script.snapshot"
                    snapshot_path.write_text(
                        Path(input_script).read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )
                    job_snapshot.input_script = str(snapshot_path)
                    job_snapshot.run_args = _replace_in_arg(job_snapshot.run_args, str(snapshot_path))
                    run_args = _replace_in_arg(run_args, str(snapshot_path))
                    job_snapshot.workdir = base_job_snapshot.workdir
            else:
                git_before = get_git_head(actions_root)
                git_after = git_before

            if _requires_build(action):
                if not build_cfg:
                    raise RuntimeError("build config missing for build/source_patch action")
                final_build_cfg = _apply_build_config(build_cfg, action, build_packs)
                build_config_diff_path = _write_build_config_diff(run_dir, build_cfg, final_build_cfg)
                build_output = build_job(final_build_cfg, source_root, run_dir)
                if not build_output.lammps_bin_path:
                    raise RuntimeError("build did not produce application binary")
                job_snapshot.app_bin = build_output.lammps_bin_path
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

            binary_provenance = collect_binary_provenance(job_snapshot.app_bin, run_dir)
            repro_script_path = _write_repro_script(
                run_dir=run_dir,
                workdir=workdir,
                job=job_snapshot,
                run_args=run_args,
            )

            run_output, profile = profiler.run(
                job_snapshot,
                run_args,
                run_env_overrides,
                workdir,
                run_dir,
                time_command,
                wrapper_command=wrapper_command,
                repeats=repeats,
                launcher_cfg=launcher_cfg,
                profiling_cfg=effective_profiling_cfg,
                is_baseline=(action is None and run_id == "baseline") or is_profile_baseline,
            )
            if (
                wrapper_command
                and "tau_exec" in str(wrapper_command[0])
                and run_output.exit_code in {133, 134}
                and _has_tau_sigtrap(run_dir)
            ):
                trace_events.append(
                    {
                        "event": "tau_sampling_retry",
                        "agent": "ProfilerAgent",
                        "run_id": run_id,
                        "exit_code": run_output.exit_code,
                        "reason": "Trace/BPT trap detected",
                    }
                )
                _backup_run_logs(run_dir, "pre_tau_retry")
                run_env_overrides = _tau_retry_env(run_env_overrides)
                run_output, profile = profiler.run(
                    job_snapshot,
                    run_args,
                    run_env_overrides,
                    workdir,
                    run_dir,
                    time_command,
                    wrapper_command=wrapper_command,
                    repeats=repeats,
                    launcher_cfg=launcher_cfg,
                    profiling_cfg=effective_profiling_cfg,
                    is_baseline=(action is None and run_id == "baseline") or is_profile_baseline,
                )
            # BWA + TAU sampling can yield truncated runs even with exit code 0.
            # Detect incomplete output and fall back to non-sampling TAU run.
            if (
                wrapper_command
                and "tau_exec" in str(wrapper_command[0])
                and job_snapshot.app == "bwa"
                and not _bwa_run_complete(run_dir)
            ):
                trace_events.append(
                    {
                        "event": "tau_sampling_incomplete_bwa",
                        "agent": "ProfilerAgent",
                        "run_id": run_id,
                        "reason": "missing [main] Real time in stderr; rerun with TAU_SAMPLING=0",
                    }
                )
                _backup_run_logs(run_dir, "pre_tau_fallback")
                run_env_overrides = dict(run_env_overrides)
                run_env_overrides["TAU_SAMPLING"] = "0"
                run_env_overrides.pop("TAU_EBS_SOURCE", None)
                run_env_overrides.pop("TAU_EBS_PERIOD", None)
                run_env_overrides.pop("TAU_EBS_UNWIND", None)
                run_output, profile = profiler.run(
                    job_snapshot,
                    run_args,
                    run_env_overrides,
                    workdir,
                    run_dir,
                    time_command,
                    wrapper_command=wrapper_command,
                    repeats=repeats,
                    launcher_cfg=launcher_cfg,
                    profiling_cfg=effective_profiling_cfg,
                    is_baseline=(action is None and run_id == "baseline") or is_profile_baseline,
                )
            # ── Per-repeat sample filtering ──────────────────────────
            # Use per_repeat_exit_codes to discard timing from killed repeats.
            # A repeat is "valid" if exit_code == 0.  Wrapper-allowed codes
            # (e.g. TAU SIGTRAP 133) mean the *wrapper* exited non-zero but
            # the process was killed mid-work, so those samples are INVALID.
            _per_rc = getattr(run_output, "per_repeat_exit_codes", None) or []
            if _per_rc and len(_per_rc) == len(run_output.samples):
                valid_samples = [
                    s for s, rc in zip(run_output.samples, _per_rc) if rc == 0
                ]
                _n_dropped = len(run_output.samples) - len(valid_samples)
                if _n_dropped > 0:
                    trace_events.append(
                        {
                            "event": "repeat_samples_filtered",
                            "agent": "ProfilerAgent",
                            "run_id": run_id,
                            "total_repeats": len(_per_rc),
                            "dropped": _n_dropped,
                            "exit_codes": list(_per_rc),
                        }
                    )
                if valid_samples:
                    _mean = sum(valid_samples) / len(valid_samples)
                    run_output = RunOutput(
                        runtime_seconds=_mean,
                        exit_code=0,
                        stdout_path=run_output.stdout_path,
                        stderr_path=run_output.stderr_path,
                        log_path=run_output.log_path,
                        time_output_path=run_output.time_output_path,
                        system_metrics=run_output.system_metrics,
                        samples=valid_samples,
                        per_repeat_exit_codes=[0] * len(valid_samples),
                    )
                else:
                    # All repeats failed — keep original (let verify reject it)
                    pass
            elif (
                wrapper_allowed_exit_codes
                and run_output.exit_code in wrapper_allowed_exit_codes
            ):
                # Legacy fallback: no per_repeat data, tolerate wrapper exit code
                run_output = RunOutput(
                    runtime_seconds=run_output.runtime_seconds,
                    exit_code=0,
                    stdout_path=run_output.stdout_path,
                    stderr_path=run_output.stderr_path,
                    log_path=run_output.log_path,
                    time_output_path=run_output.time_output_path,
                    system_metrics=run_output.system_metrics,
                    samples=run_output.samples,
                )

            # ── Per-repeat output completeness filter (BWA) ──────────
            # tau_exec can swallow the child exit code (all repeats report 0)
            # even when BWA was killed mid-execution.  Detect incomplete
            # repeats by checking for "[main] Real time" in stderr.
            if (
                job_snapshot.app == "bwa"
                and len(run_output.samples) >= 2
            ):
                _completeness = _bwa_per_repeat_complete(run_dir, len(run_output.samples))
                _valid_idx = [i for i, ok in enumerate(_completeness) if ok]
                if len(_valid_idx) < len(run_output.samples):
                    _dropped_bwa = len(run_output.samples) - len(_valid_idx)
                    trace_events.append(
                        {
                            "event": "bwa_incomplete_repeats_filtered",
                            "agent": "ProfilerAgent",
                            "run_id": run_id,
                            "total_repeats": len(run_output.samples),
                            "dropped": _dropped_bwa,
                            "completeness": _completeness,
                        }
                    )
                    if _valid_idx:
                        _bwa_valid = [run_output.samples[i] for i in _valid_idx]
                        _bwa_mean = sum(_bwa_valid) / len(_bwa_valid)
                        run_output = RunOutput(
                            runtime_seconds=_bwa_mean,
                            exit_code=0,
                            stdout_path=run_output.stdout_path,
                            stderr_path=run_output.stderr_path,
                            log_path=run_output.log_path,
                            time_output_path=run_output.time_output_path,
                            system_metrics=run_output.system_metrics,
                            samples=_bwa_valid,
                            per_repeat_exit_codes=[0] * len(_bwa_valid),
                        )
                    # else: all incomplete → keep as-is, verify will reject

            if action and "source_patch" in (action.applies_to or []):
                baseline_check_dir = run_dir / "baseline_check"
                baseline_check_dir.mkdir(parents=True, exist_ok=True)
                baseline_job = base_job_snapshot.model_copy(deep=True)
                baseline_job.env.update(run_env_overrides)
                baseline_run_args = _ensure_log_path(run_args, baseline_check_dir, app=base_job_snapshot.app)
                baseline_workdir = Path(base_job_snapshot.workdir)
                baseline_wrapper = None
                baseline_wrapper_env: Dict[str, str] = {}
                if run_purpose_norm == "profile":
                    baseline_wrapper, baseline_wrapper_env, _ = _resolve_wrappers(
                        wrappers_cfg, "baseline_check", baseline_check_dir, wrapper_id
                    )
                baseline_env_overrides = dict(run_env_overrides)
                baseline_env_overrides.update(baseline_wrapper_env)
                baseline_check_output, baseline_check_profile = profiler.run(
                    baseline_job,
                    baseline_run_args,
                    baseline_env_overrides,
                    baseline_workdir,
                    baseline_check_dir,
                    time_command,
                    wrapper_command=baseline_wrapper,
                    repeats=1,
                )
                _append_trace(
                    trace_events,
                    run_trace,
                    {
                        "event": "baseline_check",
                        "agent": "ProfilerAgent",
                        "run_id": run_id,
                        "runtime_seconds": baseline_check_output.runtime_seconds,
                        "log_path": baseline_check_output.log_path,
                    },
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
            if baseline_check_output:
                result.derived_metrics["baseline_check_runtime"] = baseline_check_output.runtime_seconds
                if result.runtime_seconds:
                    result.derived_metrics["speedup_vs_baseline_check"] = (
                        baseline_check_output.runtime_seconds / result.runtime_seconds
                    )
                result.logs.extend(
                    [
                        baseline_check_output.stdout_path,
                        baseline_check_output.stderr_path,
                        baseline_check_output.log_path,
                    ]
                )
    except LLMUnavailableError:
        raise
    except Exception as exc:
        if isinstance(exc, WorktreeAddError):
            _append_trace(
                trace_events,
                run_trace,
                {
                    "event": "worktree_add_failed",
                    "agent": "GitPatchContext",
                    "run_id": run_id,
                    "attempts": exc.attempts,
                    "last_error": exc.last_error,
                },
            )
            if reporter:
                reporter.worktree_error(run_id, exc.attempts, exc.last_error)
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
            base_run_id=base_run_id,
            base_action_id=base_action_id,
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
            build_seconds=build_output.build_seconds if build_output else None,
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
            base_run_id=base_run_id,
            base_action_id=base_action_id,
            base_job=base_job_snapshot,
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
            repro_script_path=repro_script_path,
        )
        _write_run_trace(run_dir, run_trace)
        return exp

    if action and action.applies_to == ["run_config"]:
        patch_path = _write_run_config_diff(
            run_dir,
            base_job_snapshot,
            run_args,
            env_overrides,
            base_run_id,
            base_action_id,
        )

    is_final_validation = exp_id.endswith("-validate")
    verify = verifier.verify(
        job_snapshot,
        action,
        result,
        profile,
        gates,
        baseline_exp_for_verify or baseline_exp,
        is_final_validation=is_final_validation,
    )
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
        base_run_id=base_run_id,
        base_action_id=base_action_id,
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
        build_seconds=build_output.build_seconds if build_output else None,
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
        repro_script_path=repro_script_path,
        base_run_id=base_run_id,
        base_action_id=base_action_id,
        base_job=base_job_snapshot,
    )
    _write_run_trace(run_dir, run_trace)
    return exp


def _apply_run_config_action(
    job: JobIR,
    action: Optional[ActionIR],
    arg_rules: Optional[List[Dict[str, object]]] = None,
) -> Tuple[Dict[str, str], List[str], Optional[Dict[str, object]]]:
    env_overrides: Dict[str, str] = dict(job.env or {})
    run_args = list(job.run_args)
    launcher_cfg: Optional[Dict[str, object]] = None
    if not action:
        return env_overrides, run_args, launcher_cfg

    env_overrides.update(action.parameters.get("env", {}))
    run_args_cfg = action.parameters.get("run_args", {})
    rules_by_flag: Dict[str, Dict[str, object]] = {}
    for rule in arg_rules or []:
        if isinstance(rule, dict) and rule.get("flag"):
            rules_by_flag[str(rule.get("flag"))] = rule
    for entry in run_args_cfg.get("set_flags", []):
        flag = entry.get("flag")
        values = entry.get("values", [])
        arg_count = entry.get("arg_count", len(values))
        rule = rules_by_flag.get(flag, {}) if flag else {}
        position = entry.get("position") or rule.get("position")
        replace_if_exists = entry.get("replace_if_exists")
        if replace_if_exists is None:
            replace_if_exists = rule.get("replace_if_exists", True)
        flag_args = [flag] + list(values)
        inserted = False
        if flag and replace_if_exists:
            try:
                idx = run_args.index(flag)
            except ValueError:
                idx = None
            if idx is not None:
                end = idx + 1 + arg_count
                del run_args[idx:end]
                run_args[idx:idx] = flag_args
                inserted = True
        elif flag and not replace_if_exists:
            try:
                idx = run_args.index(flag)
            except ValueError:
                idx = None
            if idx is not None:
                inserted = True
        if not inserted:
            if position == "prepend":
                run_args = flag_args + run_args
            elif position == "after_subcommand":
                if run_args:
                    run_args = run_args[:1] + flag_args + run_args[1:]
                else:
                    run_args = flag_args
            else:
                run_args.extend(flag_args)

    launcher = action.parameters.get("launcher")
    if isinstance(launcher, dict) and launcher.get("type", "direct") != "direct":
        launcher_cfg = launcher

    return env_overrides, run_args, launcher_cfg


def _extract_patch_params(action: Optional[ActionIR], repo_root: Path) -> Dict[str, object]:
    if not action or "source_patch" not in action.applies_to:
        return {"patch_path": None, "patch_paths": [], "patch_root": None}
    patch_paths_raw = action.parameters.get("patch_paths")
    patch_paths: List[Path] = []
    if isinstance(patch_paths_raw, list):
        for item in patch_paths_raw:
            if not isinstance(item, str) or not item.strip():
                continue
            candidate = Path(item)
            if not candidate.is_absolute():
                candidate = (repo_root / candidate).resolve()
            if candidate not in patch_paths:
                patch_paths.append(candidate)
    patch_path = action.parameters.get("patch_path")
    patch_root = action.parameters.get("patch_root")
    if patch_path:
        patch_path = Path(patch_path)
        if not patch_path.is_absolute():
            patch_path = (repo_root / patch_path).resolve()
        if patch_path not in patch_paths:
            patch_paths.append(patch_path)
    return {
        "patch_path": patch_path if patch_path else None,
        "patch_paths": patch_paths,
        "patch_root": Path(patch_root) if patch_root else None,
    }


def _requires_build(action: Optional[ActionIR]) -> bool:
    if not action:
        return False
    return any(target in action.applies_to for target in ["build_config", "source_patch"])


def _apply_build_config(
    base_cfg: Dict[str, object],
    action: Optional[ActionIR],
    build_packs: Optional[Dict[str, object]],
) -> Dict[str, object]:
    merged: Dict[str, object] = {**(base_cfg or {})}
    if not action:
        return merged
    build_pack_id = None
    if action.parameters:
        build_pack_id = action.parameters.get("build_pack_id")
    if build_pack_id and build_packs and isinstance(build_packs, dict):
        pack = _find_build_pack(build_packs, str(build_pack_id))
        if pack:
            merged.setdefault("cmake_args", [])
            merged["cmake_args"] = list(merged["cmake_args"]) + list(pack.get("cmake_args", []))
            merged["build_pack_id"] = build_pack_id
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


def _format_build_commands(
    build_cfg: Dict[str, object],
    run_dir: Path,
    actions_root: Path,
) -> List[str]:
    source_dir_raw = str(build_cfg.get("source_dir") or ".")
    source_dir = Path(source_dir_raw)
    if not source_dir.is_absolute():
        source_dir = actions_root / source_dir
    build_dir = run_dir / "build"
    cmake_args = list(build_cfg.get("cmake_args") or [])
    generator = str(build_cfg.get("generator") or "").strip()
    cmake_cmd = ["cmake", "-S", str(source_dir), "-B", str(build_dir)]
    if generator:
        cmake_cmd.extend(["-G", generator])
    cmake_cmd.extend([str(arg) for arg in cmake_args])
    build_cmd = ["cmake", "--build", str(build_dir)]
    target = str(build_cfg.get("target") or "").strip()
    if target:
        build_cmd.extend(["--target", target])
    build_args = list(build_cfg.get("build_args") or [])
    if build_args:
        build_cmd.append("--")
        build_cmd.extend([str(arg) for arg in build_args])
    return [" ".join(cmake_cmd), " ".join(build_cmd)]


def _format_run_command(
    lammps_bin: Path,
    run_args: List[str],
    env_overrides: Dict[str, str],
    launcher_cfg: Optional[Dict[str, object]] = None,
    wrapper_command: Optional[List[str]] = None,
) -> str:
    from skills.run_local import build_launch_cmd

    prefix = ""
    if env_overrides:
        prefix = " ".join(f"{k}={v}" for k, v in env_overrides.items())
    cmd_parts = build_launch_cmd(
        str(lammps_bin),
        [str(a) for a in run_args],
        launcher_cfg,
        wrapper_command=wrapper_command,
    )
    cmd = " ".join(cmd_parts)
    if prefix:
        return f"{prefix} {cmd}"
    return cmd


def _resolve_wrappers(
    wrappers_cfg: Optional[List[Dict[str, object]]],
    run_kind: str,
    run_dir: Path,
    wrapper_id: Optional[str],
) -> Tuple[Optional[List[str]], Dict[str, str], List[int]]:
    if not wrappers_cfg:
        return None, {}, []
    selected: Optional[Dict[str, object]] = None
    explicit = False
    if wrapper_id:
        for wrapper in wrappers_cfg:
            if str(wrapper.get("id")) == str(wrapper_id):
                selected = wrapper
                break
        explicit = selected is not None
    if not selected:
        for wrapper in wrappers_cfg:
            if not wrapper.get("enabled", False):
                continue
            apply_to = set(wrapper.get("apply_to", []) or [])
            if apply_to and run_kind not in apply_to:
                continue
            if not apply_to and wrapper.get("baseline_only", False) and run_kind != "baseline":
                continue
            selected = wrapper
            break
    if not selected or not selected.get("enabled", False):
        return None, {}, []
    if explicit:
        # Explicit wrapper choice bypasses baseline_only/apply_to, but still respects limits.
        pass

    max_uses = selected.get("max_uses")
    if max_uses is not None:
        try:
            max_uses_int = int(max_uses)
        except (TypeError, ValueError):
            max_uses_int = None
        if max_uses_int is not None:
            used = int(selected.get("_uses", 0) or 0)
            if used >= max_uses_int:
                return None, {}, []
            selected["_uses"] = used + 1

    exec_name = str(selected.get("exec") or "")
    if not exec_name:
        return None, {}, []
    if not (shutil.which(exec_name) or Path(exec_name).exists()):
        return None, {}, []
    args = [str(a) for a in (selected.get("args") or [])]
    wrapper = [exec_name] + args

    env: Dict[str, str] = {}
    for key, value in (selected.get("env") or {}).items():
        expanded = os.path.expandvars(str(value))
        env[str(key)] = expanded

    profile_subdir = str(selected.get("profile_subdir") or selected.get("id") or "wrapper")
    wrapper_dir = run_dir / profile_subdir
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    if selected.get("id") == "tau":
        env.setdefault("TAU_PROFILE", "1")
        env.setdefault("TAU_TRACE", "0")
        env.setdefault("TAU_PROFILE_FORMAT", "profile")
        env.setdefault("TAU_PROFILEDIR", str(wrapper_dir))
        env.setdefault("TAU_SAMPLING", "1")
        env.setdefault("TAU_EBS_SOURCE", "itimer")
        env.setdefault("TAU_EBS_PERIOD", "100000")
        env.setdefault("TAU_EBS_UNWIND", "0")

    allowed_exit_codes: List[int] = []
    raw_codes = selected.get("allowed_exit_codes")
    if isinstance(raw_codes, list):
        allowed_exit_codes = [int(c) for c in raw_codes if isinstance(c, (int, float))]
    elif selected.get("id") == "tau":
        # TAU sampling on macOS ARM exits with SIGTRAP (128+5=133)
        allowed_exit_codes = [133, 134]
    return wrapper, env, allowed_exit_codes


def _replace_in_arg(run_args: List[str], input_path: str) -> List[str]:
    updated = list(run_args)
    for idx in range(len(updated) - 1):
        if updated[idx] == "-in":
            updated[idx + 1] = input_path
            return updated
    return updated


def _sync_path_into_worktree(src: Path, dst: Path) -> bool:
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
        return True
    except Exception:
        try:
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
            return True
        except Exception:
            return False


def _sync_runtime_inputs_to_worktree(
    *,
    app: str,
    orig_workdir: Path,
    workdir: Path,
    run_args: List[str],
) -> List[str]:
    """Ensure runtime input files are present in the patch worktree.

    Git worktrees omit untracked runtime artifacts. We sync only missing
    path-like arguments found in run_args and app-specific sidecars.
    """
    synced: List[str] = []
    if not orig_workdir.exists():
        return synced
    workdir.mkdir(parents=True, exist_ok=True)

    seen_rel: set[str] = set()
    file_inputs: List[Tuple[Path, Path]] = []
    for token in run_args or []:
        arg = str(token or "").strip()
        if not arg or arg.startswith("-"):
            continue
        rel = Path(arg)
        if rel.is_absolute():
            continue
        src = orig_workdir / rel
        if not src.exists():
            continue
        rel_key = str(rel)
        if rel_key in seen_rel:
            continue
        seen_rel.add(rel_key)
        file_inputs.append((src, rel))

    for src, rel in file_inputs:
        dst = workdir / rel
        if _sync_path_into_worktree(src, dst):
            synced.append(str(rel))

    if app.strip().lower() == "bwa":
        sidecar_exts = [".amb", ".ann", ".bwt", ".pac", ".sa", ".fai"]
        for src, rel in file_inputs:
            lower = src.name.lower()
            if not (lower.endswith(".fa") or lower.endswith(".fasta")):
                continue
            for ext in sidecar_exts:
                side_src = Path(f"{src}{ext}")
                if not side_src.exists():
                    continue
                side_rel = Path(f"{rel}{ext}")
                side_dst = workdir / side_rel
                if _sync_path_into_worktree(side_src, side_dst):
                    synced.append(str(side_rel))
    return synced


def _find_build_pack(build_packs: Dict[str, object], pack_id: str) -> Optional[Dict[str, object]]:
    packs = build_packs.get("packs", []) if isinstance(build_packs, dict) else []
    for pack in packs:
        if pack.get("id") == pack_id:
            return pack
    return None


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


def _ensure_log_path(run_args: List[str], run_dir: Path, app: str) -> List[str]:
    return app_ensure_log_path(app, run_args, run_dir)


def _build_result_ir(
    run_output,
    profile: ProfileReport,
    baseline_runtime: Optional[float],
    runtime_agg: str,
    prior_samples: Optional[List[float]],
) -> ResultIR:
    timing = profile.timing_breakdown
    samples = (prior_samples or []) + (run_output.samples or [])
    aggregate_wall = _aggregate_runtime(samples, runtime_agg) if samples else run_output.runtime_seconds
    loop_total = timing.get("total")
    total = loop_total or aggregate_wall
    comm_ratio = (timing.get("comm", 0.0) / total) if total else 0.0
    derived = {"comm_ratio": comm_ratio}
    if aggregate_wall:
        derived["runtime_wall_seconds"] = aggregate_wall
        if loop_total:
            derived["runtime_overhead_seconds"] = max(0.0, aggregate_wall - loop_total)
    if baseline_runtime and total:
        derived["speedup_vs_baseline"] = baseline_runtime / total
    if len(samples) >= 2:
        mean = sum(samples) / len(samples)
        var = sum((x - mean) ** 2 for x in samples) / len(samples)
        derived["variance"] = var
        derived["variance_cv"] = (var ** 0.5) / mean if mean else 0.0
    return ResultIR(
        runtime_seconds=total,
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


def _select_directions_by_signal(
    direction_map: Dict[str, Dict[str, object]],
    profile_features: Dict[str, object],
    candidate_policy: Optional[Dict[str, object]],
    direction_top_k: int,
) -> tuple[List[str], Dict[str, float]]:
    tags = set(profile_features.get("bottleneck_tags") or [])
    metrics = profile_features.get("metrics", {}) or {}
    allow_skip = True
    if candidate_policy and isinstance(candidate_policy, dict):
        allow_skip = bool(candidate_policy.get("allow_skip_if_not_applicable", True))
    scores: Dict[str, float] = {}
    for direction_id, direction in direction_map.items():
        applies_when = set(direction.get("applies_when", []) or [])
        if allow_skip and applies_when and not (applies_when & tags):
            continue
        scores[direction_id] = _direction_score(direction_id, metrics, tags)
    if not scores:
        if "compute" in direction_map:
            return ["compute"], {"compute": 1.0}
        fallback = list(direction_map.keys())
        return (fallback[:1] if fallback else []), {}
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    selected = [item[0] for item in ordered[: max(1, direction_top_k)]]
    return selected, scores


def _filter_actions_by_direction(
    actions: List[ActionIR],
    direction_map: Dict[str, Dict[str, object]],
    selected_directions: List[str],
    candidate_policy: Optional[Dict[str, object]],
    profile_features: Dict[str, object],
) -> List[ActionIR]:
    if not direction_map or not selected_directions:
        return actions
    prefer_effects: List[str] = []
    tags = set(profile_features.get("bottleneck_tags") or [])
    if candidate_policy and isinstance(candidate_policy, dict):
        adjustments = candidate_policy.get("domain_adjustments", {}) or {}
        for tag in tags:
            entry = adjustments.get(tag, {})
            prefer_effects.extend(entry.get("prefer_effects", []) or [])
    selected: List[ActionIR] = []
    seen: set[str] = set()
    for direction_id in selected_directions:
        direction_cfg = direction_map.get(direction_id)
        if not direction_cfg:
            continue
        candidates = select_actions_for_direction(
            actions, direction_cfg, prefer_effects=prefer_effects or None
        )
        for action in candidates:
            if action.action_id in seen:
                continue
            selected.append(action)
            seen.add(action.action_id)
    return selected or actions


def _direction_score(direction_id: str, metrics: Dict[str, object], tags: set[str]) -> float:
    io_ratio = float(metrics.get("io_ratio") or 0.0)
    comm_ratio = float(metrics.get("comm_ratio") or 0.0)
    compute_ratio = float(metrics.get("compute_ratio") or 0.0)
    imbalance_ratio = float(metrics.get("imbalance_ratio") or 0.0)
    if direction_id == "io":
        return io_ratio
    if direction_id == "communication":
        return comm_ratio
    if direction_id == "compute":
        return compute_ratio
    if direction_id == "imbalance":
        return imbalance_ratio
    if direction_id == "memory":
        return 1.0 if "mem_bound" in tags else 0.0
    return 0.0


def _write_run_manifest(
    run_dir: Path,
    run_id: str,
    job: JobIR,
    action: Optional[ActionIR],
    base_run_id: Optional[str],
    base_action_id: Optional[str],
    base_job: Optional[JobIR],
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
    repro_script_path: Optional[str],
) -> None:
    manifest = _build_run_manifest(
        run_id=run_id,
        job=job,
        action=action,
        base_run_id=base_run_id,
        base_action_id=base_action_id,
        base_job=base_job,
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
        repro_script_path=repro_script_path,
    )
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _build_run_manifest(
    run_id: str,
    job: JobIR,
    action: Optional[ActionIR],
    base_run_id: Optional[str],
    base_action_id: Optional[str],
    base_job: Optional[JobIR],
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
    repro_script_path: Optional[str],
) -> Dict[str, object]:
    git_status = get_git_status(repo_root)
    lammps_hash = _sha256_file(job.app_bin)
    input_hash = _sha256_file(job.input_script) if job.input_script else None
    artifacts = {
        "stdout": result.logs[0] if result.logs else None,
        "stderr": result.logs[1] if len(result.logs) > 1 else None,
        "log": result.logs[2] if len(result.logs) > 2 else None,
        "time": run_output.time_output_path if run_output else None,
        "patch": patch_path,
        "repro_script": repro_script_path,
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
            "build_seconds": build_output.build_seconds,
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
        "base": {
            "run_id": base_run_id,
            "action_id": base_action_id,
            "env": base_job.env if base_job else None,
            "run_args": base_job.run_args if base_job else None,
        },
        "env_overrides": env_overrides,
        "env_final": job.env,
        "run_args": run_args,
        "code_version": {
            "git_commit_before": git_before,
            "git_commit_after": git_after,
            "dirty": git_status.get("dirty"),
        },
        "binary_version": {
            "path": job.app_bin,
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
    ]
    lines.extend(
        [
            "",
            "## Candidates",
        ]
    )
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
    ]
    lines.extend(["", "## 候选动作"])
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
    base_run_id: Optional[str],
    base_action_id: Optional[str],
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
    build_seconds: Optional[float] = None,
) -> ExperimentIR:
    timestamps = [time.strftime("%Y-%m-%dT%H:%M:%S")]
    return ExperimentIR(
        exp_id=exp_id,
        parent_exp_id=None,
        base_run_id=base_run_id,
        base_action_id=base_action_id,
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
        build_seconds=build_seconds,
    )


def _write_run_config_diff(
    run_dir: Path,
    base_job: JobIR,
    run_args: List[str],
    env_overrides: Dict[str, str],
    base_run_id: Optional[str],
    base_action_id: Optional[str],
) -> str:
    diff = {
        "base_run_id": base_run_id,
        "base_action_id": base_action_id,
        "env_before": base_job.env,
        "env_overrides": env_overrides,
        "env_after": {**base_job.env, **env_overrides},
        "run_args_before": base_job.run_args,
        "run_args_after": run_args,
    }
    diff_path = run_dir / "run_config.diff.json"
    diff_path.write_text(json.dumps(diff, indent=2), encoding="utf-8")
    return str(diff_path)


def _write_repro_script(
    run_dir: Path,
    workdir: Path,
    job: JobIR,
    run_args: List[str],
) -> str:
    import shlex

    repro_path = run_dir / "repro.sh"
    env_lines = []
    for key, value in sorted(job.env.items()):
        env_lines.append(f"export {key}={shlex.quote(str(value))}")
    args = " ".join(shlex.quote(str(arg)) for arg in run_args)
    lines = [
        "#!/bin/sh",
        "set -e",
        f"cd {shlex.quote(str(workdir))}",
        *env_lines,
        f"{shlex.quote(str(job.app_bin))} {args}",
        "",
    ]
    repro_path.write_text("\n".join(lines), encoding="utf-8")
    repro_path.chmod(0o755)
    return str(repro_path)


def _write_experiment(run_dir: Path, exp: ExperimentIR) -> None:
    path = run_dir / "experiment.json"
    path.write_text(json.dumps(exp.model_dump(), indent=2), encoding="utf-8")


def _read_experiment(path: Path) -> Optional[ExperimentIR]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
    try:
        return ExperimentIR.model_validate(data)
    except Exception:
        return None


def _write_best_state(
    artifacts_dir: Path,
    baseline_exp: ExperimentIR,
    best_exp: Optional[ExperimentIR],
    best_chain_exp: Optional[ExperimentIR],
    best_run_exp: Optional[ExperimentIR],
    best_build_exp: Optional[ExperimentIR],
    phase: str,
    frozen_run_id: Optional[str],
    frozen_build_id: Optional[str],
    opportunity_graph: Optional[Dict[str, object]] = None,
) -> Optional[str]:
    def exp_path(run_id: Optional[str]) -> Optional[str]:
        if not run_id:
            return None
        path = artifacts_dir / "runs" / run_id / "experiment.json"
        return str(path) if path.exists() else None

    payload = {
        "case_id": baseline_exp.job.case_id,
        "baseline_run_id": baseline_exp.run_id,
        "baseline_exp_path": exp_path(baseline_exp.run_id),
        "best_chain_run_id": best_chain_exp.run_id if best_chain_exp else None,
        "best_chain_exp_path": exp_path(best_chain_exp.run_id if best_chain_exp else None),
        "best_run_id": best_run_exp.run_id if best_run_exp else None,
        "best_run_exp_path": exp_path(best_run_exp.run_id if best_run_exp else None),
        "best_build_id": best_build_exp.run_id if best_build_exp else None,
        "best_build_exp_path": exp_path(best_build_exp.run_id if best_build_exp else None),
        "best_action_id": best_exp.action.action_id if best_exp and best_exp.action else None,
        "best_action": best_exp.action.model_dump() if best_exp and best_exp.action else None,
        "best_run_action": best_run_exp.action.model_dump() if best_run_exp and best_run_exp.action else None,
        "best_build_action": best_build_exp.action.model_dump()
        if best_build_exp and best_build_exp.action
        else None,
        "phase": phase,
        "frozen_run_id": frozen_run_id,
        "frozen_build_id": frozen_build_id,
        "opportunity_graph": opportunity_graph or {},
    }
    path = artifacts_dir / "best_state.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def _safe_read(path: Path) -> str:
    try:
        if path.is_dir():
            return ""
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _is_under_repo(path: Path, repo_root: Path) -> bool:
    try:
        path.resolve().relative_to(repo_root.resolve())
        return True
    except ValueError:
        return False
