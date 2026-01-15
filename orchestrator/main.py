from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from orchestrator.graph import run_optimization
from orchestrator.llm_client import LLMClient, LLMConfig
from orchestrator.router import load_action_space, load_gates, load_policy
from schemas.job_ir import Budgets, JobIR


def main() -> None:
    parser = argparse.ArgumentParser(description="HPC agent platform MVP")
    parser.add_argument("--case", required=True, help="Case ID from configs/lammps_cases.yaml")
    parser.add_argument("--config-dir", default="configs", help="Config directory")
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

    actions = load_action_space(config_dir / "action_space.yaml")
    policy = load_policy(config_dir / "policy.yaml")
    gates = load_gates(config_dir / "gates.yaml")

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
    result = run_optimization(
        job=job,
        actions=actions,
        policy=policy,
        gates=gates,
        artifacts_dir=artifacts_dir,
        time_command=env_cfg.get("time_command"),
        min_delta_seconds=env_cfg.get("min_delta_seconds", 0.0),
        top_k=env_cfg.get("top_k", 5),
        llm_client=llm_client,
    )

    print(result["summary_table"])
    print(f"Report: {result['report_md']}")


if __name__ == "__main__":
    main()
