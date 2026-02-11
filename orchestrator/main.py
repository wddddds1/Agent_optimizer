from __future__ import annotations

import argparse
import glob
import json
import shutil
import time
from pathlib import Path

import yaml

from orchestrator.console import ConsoleUI
from orchestrator.graph import run_optimization
from orchestrator.llm_client import LLMClient, LLMConfig
from orchestrator.router import load_action_space, load_direction_space, load_gates, load_policy
from schemas.job_ir import Budgets, JobIR


def _deep_merge_dicts(base: dict, overlay: dict) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_planner_cfg(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return {}
    defaults = raw.get("defaults") if isinstance(raw.get("defaults"), dict) else {}
    profiles = raw.get("profiles") if isinstance(raw.get("profiles"), dict) else {}
    active_profile = raw.get("active_profile")
    if isinstance(active_profile, str) and active_profile:
        profile_cfg = profiles.get(active_profile)
        if isinstance(profile_cfg, dict):
            return _deep_merge_dicts(defaults or {}, profile_cfg)
    return defaults or {}


def _migrate_legacy_artifacts(artifacts_root: Path) -> Path | None:
    sessions_dir = artifacts_root / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    legacy_source = artifacts_root / "runs"
    if not legacy_source.exists():
        return None
    session_id = time.strftime("%Y%m%d-%H%M%S")
    legacy_dir = sessions_dir / f"legacy-{session_id}"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    for entry in artifacts_root.iterdir():
        if entry.name in {"sessions", "latest", "latest.json"}:
            continue
        shutil.move(str(entry), str(legacy_dir / entry.name))
    return legacy_dir


def _init_artifacts_session(artifacts_root: Path) -> Path:
    sessions_dir = artifacts_root / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_id = time.strftime("%Y%m%d-%H%M%S")
    session_dir = sessions_dir / session_id
    suffix = 1
    while session_dir.exists():
        session_dir = sessions_dir / f"{session_id}-{suffix}"
        suffix += 1
    session_dir.mkdir(parents=True, exist_ok=True)

    latest_link = artifacts_root / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        if latest_link.is_dir() and not latest_link.is_symlink():
            shutil.rmtree(latest_link)
        else:
            latest_link.unlink()
    try:
        latest_link.symlink_to(session_dir.relative_to(artifacts_root))
    except OSError:
        pass

    latest_json = artifacts_root / "latest.json"
    latest_json.write_text(
        json.dumps(
            {
                "session_id": session_dir.name,
                "path": str(session_dir.resolve()),
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    return session_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="HPC agent platform MVP")
    parser.add_argument("--case", required=True, help="Case ID from configs/lammps_cases.yaml")
    parser.add_argument("--config-dir", default="configs", help="Config directory")
    parser.add_argument(
        "--ui",
        default="console",
        choices=["console", "quiet"],
        help="Console output mode",
    )
    parser.add_argument(
        "--ui-verbose",
        action="store_true",
        help="Enable verbose console output (show full details and raw previews)",
    )
    parser.add_argument(
        "--ui-no-raw",
        action="store_true",
        help="Disable raw stdout/stderr/log preview for each run",
    )
    parser.add_argument(
        "--ui-preview-bytes",
        type=int,
        default=2048,
        help="Max bytes per output preview when raw preview is enabled",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="Override max iterations for this run",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Override max runs for this run",
    )
    parser.add_argument(
        "--resume-best",
        action="store_true",
        help="Resume from artifacts/best_state.json and skip to start phase",
    )
    parser.add_argument(
        "--resume-state",
        default=None,
        help="Path to best_state.json for resuming a run",
    )
    parser.add_argument(
        "--start-phase",
        default=None,
        choices=["RUN_TUNE", "BUILD_TUNE", "RUN_RETUNE", "PATCH"],
        help="Force the starting phase when resuming",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Override max candidates per iteration",
    )
    parser.add_argument(
        "--baseline-repeats",
        type=int,
        default=None,
        help="Override baseline repeat count for this run",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Reuse baseline from resume state if available instead of rerunning it",
    )
    parser.add_argument(
        "--patch-debug-retries",
        type=int,
        default=2,
        help="Override max patch debug retries (default: 2)",
    )
    parser.add_argument(
        "--fixed-threads",
        type=int,
        default=None,
        help="Fix OMP thread count and skip thread sweep if set",
    )
    parser.add_argument(
        "--validate-top1-repeats",
        type=int,
        default=None,
        help="Override top-1 validation repeats for this run",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    env_cfg = yaml.safe_load((config_dir / "env.local.yaml").read_text(encoding="utf-8"))
    cases_cfg = yaml.safe_load((config_dir / "lammps_cases.yaml").read_text(encoding="utf-8"))
    case = cases_cfg["cases"].get(args.case)
    if not case:
        raise SystemExit(f"Unknown case_id: {args.case}")

    default_env = env_cfg.get("default_env", {})
    env = {**default_env, **case.get("env", {})}
    if args.fixed_threads is not None:
        env["OMP_NUM_THREADS"] = str(int(args.fixed_threads))
    budgets = Budgets(**case["budgets"])
    if args.max_iters is not None:
        budgets.max_iters = int(args.max_iters)
    if args.max_runs is not None:
        budgets.max_runs = int(args.max_runs)

    lammps_bin = case.get("lammps_bin") or env_cfg["lammps_bin"]
    lammps_bin_path = Path(lammps_bin)
    if not lammps_bin_path.is_absolute():
        lammps_bin_path = (config_dir.parent / lammps_bin_path).resolve()

    workdir = Path(case["workdir"])
    if not workdir.is_absolute():
        workdir = (config_dir.parent / workdir).resolve()
    input_script = Path(case["input_script"])
    if not input_script.is_absolute():
        input_script = (config_dir.parent / input_script).resolve()

    job = JobIR(
        case_id=args.case,
        workdir=str(workdir),
        lammps_bin=str(lammps_bin_path),
        input_script=str(input_script),
        env=env,
        run_args=case.get("run_args", []),
        budgets=budgets,
        tags=case.get("tags", []),
    )

    selection_mode = env_cfg.get("selection_mode", "action")
    actions = load_action_space(config_dir / "action_space.yaml")
    direction_space = []
    direction_path = env_cfg.get("direction_space", "configs/direction_space.yaml")
    direction_file = Path(direction_path)
    if not direction_file.is_absolute():
        direction_file = (config_dir.parent / direction_file).resolve()
    if direction_file.exists():
        direction_space = load_direction_space(direction_file)
    policy = load_policy(config_dir / "policy.yaml")
    gates = load_gates(config_dir / "gates.yaml")
    candidate_policy = None
    candidate_policy_path = config_dir / "candidate_policy.yaml"
    if candidate_policy_path.exists():
        candidate_policy = yaml.safe_load(candidate_policy_path.read_text(encoding="utf-8"))
    build_packs = None
    build_packs_path = config_dir / "build_packs.yaml"
    if build_packs_path.exists():
        build_packs = yaml.safe_load(build_packs_path.read_text(encoding="utf-8"))
    patch_families = None
    patch_families_path = config_dir / "patch_families.yaml"
    if patch_families_path.exists():
        patch_families = yaml.safe_load(patch_families_path.read_text(encoding="utf-8"))
    survey_guidance = None
    guidance_path = env_cfg.get("survey_guidance", "configs/code_survey_guidance.yaml")
    guidance_file = Path(guidance_path)
    if not guidance_file.is_absolute():
        guidance_file = (config_dir.parent / guidance_file).resolve()
    if guidance_file.exists():
        survey_guidance = yaml.safe_load(guidance_file.read_text(encoding="utf-8"))
    hierarchical_cfg = None
    hierarchical_path = config_dir / "hierarchical.yaml"
    if hierarchical_path.exists():
        hierarchical_cfg = yaml.safe_load(hierarchical_path.read_text(encoding="utf-8"))
    adapter_cfg = None
    adapter_dir = Path(env_cfg.get("adapter_dir", "configs/adapters"))
    if not adapter_dir.is_absolute():
        adapter_dir = (config_dir.parent / adapter_dir).resolve()
    adapter_path = adapter_dir / f"{job.app}.yaml"
    if adapter_path.exists():
        adapter_cfg = yaml.safe_load(adapter_path.read_text(encoding="utf-8"))
    if args.patch_debug_retries is not None:
        if not isinstance(adapter_cfg, dict):
            adapter_cfg = {}
        patch_rules = adapter_cfg.get("patch_rules")
        if not isinstance(patch_rules, dict):
            patch_rules = {}
        patch_rules["debug_max_attempts"] = int(args.patch_debug_retries)
        adapter_cfg["patch_rules"] = patch_rules
    planner_cfg: dict = {}
    planner_path = config_dir / "planner.yaml"
    if planner_path.exists():
        planner_cfg = _load_planner_cfg(planner_path)
    if args.max_candidates is not None:
        planner_cfg["max_candidates"] = int(args.max_candidates)

    llm_cfg_raw = env_cfg.get("llm", {})
    llm_config = LLMConfig(
        enabled=bool(llm_cfg_raw.get("enabled", False)),
        api_key_env=llm_cfg_raw.get("api_key_env", "DEEPSEEK_API_KEY"),
        base_url=llm_cfg_raw.get("base_url", "https://api.deepseek.com"),
        model=llm_cfg_raw.get("model", "deepseek-chat"),
        temperature=float(llm_cfg_raw.get("temperature", 0.0)),
        max_tokens=int(llm_cfg_raw.get("max_tokens", 512)),
    )
    llm_client = LLMClient(llm_config)

    artifacts_root = Path(env_cfg.get("artifacts_dir", "artifacts"))
    if not artifacts_root.is_absolute():
        artifacts_root = (config_dir.parent / artifacts_root).resolve()
    artifacts_root.mkdir(parents=True, exist_ok=True)
    legacy_dir = _migrate_legacy_artifacts(artifacts_root)

    resume_state = None
    resume_path = None
    if args.resume_state:
        resume_path = Path(args.resume_state)
        if not resume_path.is_absolute():
            resume_path = (config_dir.parent / resume_path).resolve()
    elif args.resume_best:
        latest_dir = artifacts_root / "latest"
        candidates = []
        if latest_dir.exists():
            candidates.append(latest_dir / "best_state.json")
        if legacy_dir:
            candidates.append(legacy_dir / "best_state.json")
        candidates.append(artifacts_root / "best_state.json")
        for candidate in candidates:
            if candidate.exists():
                resume_path = candidate
                break
    if resume_path and resume_path.exists():
        resume_payload = json.loads(resume_path.read_text(encoding="utf-8"))
        start_phase = args.start_phase or resume_payload.get("start_phase")
        if not start_phase and args.resume_best:
            start_phase = "PATCH"
        seed_paths = []
        for key in ("baseline_exp_path", "best_chain_exp_path", "best_run_exp_path", "best_build_exp_path"):
            value = resume_payload.get(key)
            if value:
                seed_paths.append(value)
        session_dir = resume_path.parent
        def _find_exp_path(run_id: str) -> Optional[Path]:
            exp_path = session_dir / "runs" / run_id / "experiment.json"
            if exp_path.exists():
                return exp_path
            # search across sessions if the run_id lives in another session
            if artifacts_root.exists():
                pattern = str(
                    artifacts_root / "sessions" / "*" / "runs" / run_id / "experiment.json"
                )
                matches = glob.glob(pattern)
                if matches:
                    return Path(matches[0])
            return None
        if "best_run_exp_path" not in resume_payload or not resume_payload.get("best_run_exp_path"):
            run_id = resume_payload.get("best_run_id")
            if run_id:
                exp_path = _find_exp_path(run_id)
                if exp_path:
                    seed_paths.append(str(exp_path))
        if "best_build_exp_path" not in resume_payload or not resume_payload.get("best_build_exp_path"):
            run_id = resume_payload.get("best_build_id")
            if run_id:
                exp_path = _find_exp_path(run_id)
                if exp_path:
                    seed_paths.append(str(exp_path))
        if not seed_paths:
            fallback = []
            for run_id_key in ("baseline_run_id", "best_run_id", "best_build_id"):
                run_id = resume_payload.get(run_id_key)
                if not run_id:
                    continue
                exp_path = _find_exp_path(run_id)
                if exp_path:
                    fallback.append(str(exp_path))
            if fallback:
                seed_paths = fallback
        if not seed_paths:
            exp_paths = list((session_dir / "runs").glob("*/experiment.json"))
            if exp_paths:
                seed_paths = [str(p) for p in exp_paths]
        resume_state = {
            "start_phase": start_phase,
            "frozen_run_id": resume_payload.get("best_run_id"),
            "frozen_build_id": resume_payload.get("best_build_id"),
            "best_chain_exp_path": resume_payload.get("best_chain_exp_path"),
            "best_chain_run_id": resume_payload.get("best_chain_run_id"),
            "seed_experiments": seed_paths,
        }
    artifacts_dir = _init_artifacts_session(artifacts_root)
    reporter = (
        ConsoleUI(
            enabled=args.ui == "console",
            verbose=bool(args.ui_verbose),
            show_output_preview=bool(args.ui_verbose) and not args.ui_no_raw,
            preview_bytes=args.ui_preview_bytes,
        )
        if args.ui
        else None
    )
    baseline_repeats = int(env_cfg.get("experiment", {}).get("baseline_repeats", 1))
    if args.baseline_repeats is not None:
        baseline_repeats = int(args.baseline_repeats)
    validate_top1_repeats = int(env_cfg.get("experiment", {}).get("validate_top1_repeats", 0))
    if args.validate_top1_repeats is not None:
        validate_top1_repeats = int(args.validate_top1_repeats)

    skip_baseline = bool(args.skip_baseline)
    if resume_state and not skip_baseline:
        if resume_state.get("seed_experiments"):
            skip_baseline = True
    if resume_state and args.start_phase and not args.skip_baseline:
        skip_baseline = True

    wrappers_cfg = env_cfg.get("wrappers")
    if not wrappers_cfg and env_cfg.get("tau"):
        tau_cfg = env_cfg.get("tau", {})
        wrappers_cfg = [
            {
                "id": "tau",
                "enabled": tau_cfg.get("enabled", False),
                "baseline_only": tau_cfg.get("baseline_only", True),
                "exec": tau_cfg.get("exec", "tau_exec"),
                "args": tau_cfg.get("args", []),
                "env": tau_cfg.get("env", {}),
                "profile_subdir": tau_cfg.get("profile_subdir", "tau"),
            }
        ]

    result = run_optimization(
        job=job,
        actions=actions,
        policy=policy,
        gates=gates,
        artifacts_dir=artifacts_dir,
        time_command=env_cfg.get("time_command"),
        wrappers_cfg=wrappers_cfg,
        min_delta_seconds=env_cfg.get("min_delta_seconds", 0.0),
        top_k=env_cfg.get("top_k", 5),
        selection_mode=selection_mode,
        direction_top_k=int(env_cfg.get("direction_top_k", env_cfg.get("top_k", 5))),
        direction_space=direction_space,
        llm_client=llm_client,
        candidate_policy=candidate_policy,
        build_packs=build_packs,
        patch_families=patch_families,
        survey_guidance=survey_guidance,
        hierarchical_cfg=hierarchical_cfg,
        adapter_cfg=adapter_cfg,
        planner_cfg=planner_cfg,
        reporter=reporter,
        build_cfg=env_cfg.get("build", {}),
        baseline_repeats=baseline_repeats,
        baseline_stat=env_cfg.get("experiment", {}).get("baseline_stat", "mean"),
        validate_top1_repeats=validate_top1_repeats,
        min_improvement_pct=float(env_cfg.get("experiment", {}).get("min_improvement_pct", 0.0)),
        resume_state=resume_state,
        fixed_threads=args.fixed_threads,
        skip_baseline=skip_baseline,
    )

    print(result["summary_table"])
    print(f"Report: {result['report_md']}")
    if "report_zh" in result:
        print(f"Report (ZH): {result['report_zh']}")


if __name__ == "__main__":
    main()
