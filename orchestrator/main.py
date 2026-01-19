from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from orchestrator.console import ConsoleUI
from orchestrator.graph import run_optimization
from orchestrator.llm_client import LLMClient, LLMConfig
from orchestrator.router import load_action_space, load_direction_space, load_gates, load_policy
from schemas.job_ir import Budgets, JobIR


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
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    env_cfg = yaml.safe_load((config_dir / "env.local.yaml").read_text(encoding="utf-8"))
    cases_cfg = yaml.safe_load((config_dir / "lammps_cases.yaml").read_text(encoding="utf-8"))
    case = cases_cfg["cases"].get(args.case)
    if not case:
        raise SystemExit(f"Unknown case_id: {args.case}")

    default_env = env_cfg.get("default_env", {})
    env = {**default_env, **case.get("env", {})}
    budgets = Budgets(**case["budgets"])

    lammps_bin = env_cfg["lammps_bin"]
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
    if selection_mode == "direction":
        direction_path = env_cfg.get("direction_space", "configs/direction_space.yaml")
        direction_file = Path(direction_path)
        if not direction_file.is_absolute():
            direction_file = (config_dir.parent / direction_file).resolve()
        actions = load_direction_space(direction_file)
    else:
        actions = load_action_space(config_dir / "action_space.yaml")
    policy = load_policy(config_dir / "policy.yaml")
    gates = load_gates(config_dir / "gates.yaml")
    candidate_policy = None
    candidate_policy_path = config_dir / "candidate_policy.yaml"
    if candidate_policy_path.exists():
        candidate_policy = yaml.safe_load(candidate_policy_path.read_text(encoding="utf-8"))
    adapter_cfg = None
    adapter_dir = Path(env_cfg.get("adapter_dir", "configs/adapters"))
    if not adapter_dir.is_absolute():
        adapter_dir = (config_dir.parent / adapter_dir).resolve()
    adapter_path = adapter_dir / f"{job.app}.yaml"
    if adapter_path.exists():
        adapter_cfg = yaml.safe_load(adapter_path.read_text(encoding="utf-8"))
    planner_cfg = {}
    planner_path = config_dir / "planner.yaml"
    if planner_path.exists():
        planner_cfg = yaml.safe_load(planner_path.read_text(encoding="utf-8")).get("defaults", {})

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

    artifacts_dir = Path(env_cfg.get("artifacts_dir", "artifacts"))
    if not artifacts_dir.is_absolute():
        artifacts_dir = (config_dir.parent / artifacts_dir).resolve()
    reporter = (
        ConsoleUI(
            enabled=args.ui == "console",
            show_output_preview=not args.ui_no_raw,
            preview_bytes=args.ui_preview_bytes,
        )
        if args.ui
        else None
    )
    result = run_optimization(
        job=job,
        actions=actions,
        policy=policy,
        gates=gates,
        artifacts_dir=artifacts_dir,
        time_command=env_cfg.get("time_command"),
        min_delta_seconds=env_cfg.get("min_delta_seconds", 0.0),
        top_k=env_cfg.get("top_k", 5),
        selection_mode=selection_mode,
        direction_top_k=int(env_cfg.get("direction_top_k", env_cfg.get("top_k", 5))),
        llm_client=llm_client,
        candidate_policy=candidate_policy,
        adapter_cfg=adapter_cfg,
        planner_cfg=planner_cfg,
        reporter=reporter,
        build_cfg=env_cfg.get("build", {}),
        baseline_repeats=int(env_cfg.get("experiment", {}).get("baseline_repeats", 1)),
        baseline_stat=env_cfg.get("experiment", {}).get("baseline_stat", "mean"),
        validate_top1_repeats=int(env_cfg.get("experiment", {}).get("validate_top1_repeats", 0)),
        min_improvement_pct=float(env_cfg.get("experiment", {}).get("min_improvement_pct", 0.0)),
    )

    print(result["summary_table"])
    print(f"Report: {result['report_md']}")
    if "report_zh" in result:
        print(f"Report (ZH): {result['report_zh']}")


if __name__ == "__main__":
    main()
