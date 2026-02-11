#!/usr/bin/env python3
"""Test script for the new agentic code optimizer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from orchestrator.agents.code_optimizer_agent import create_optimizer_agent
from schemas.action_ir import ActionIR
from schemas.profile_report import ProfileReport


def main():
    """Test the code optimizer agent."""
    print("=" * 60)
    print("Code Optimizer Agent Test")
    print("=" * 60)

    # Create the agent
    repo_root = project_root / "third_party" / "lammps"
    build_dir = project_root / "build" / "lammps-macos-omp"

    print(f"\nRepo root: {repo_root}")
    print(f"Build dir: {build_dir}")

    # Check if API key is set
    import os
    api_key_env = "DEEPSEEK_API_KEY"
    if not os.environ.get(api_key_env):
        print(f"\nError: {api_key_env} environment variable not set")
        print("Please set your API key to test the agent.")
        return

    agent = create_optimizer_agent(
        repo_root=repo_root,
        api_key_env=api_key_env,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        build_dir=build_dir,
    )

    # Create a test action
    action = ActionIR(
        action_id="source_patch.param_table_pack",
        family="source_patch",
        applies_to=["source_patch"],
        description="Pack LJ coefficients into cache-aligned struct",
        expected_effect="reduce_cache_misses",
        risk_level="medium",
        parameters={
            "patch_family": "param_table_pack",
            "target_file": "src/OPENMP/pair_lj_cut_omp.cpp",
        },
    )

    # Create mock profile data
    profile = ProfileReport(
        timing_breakdown={
            "Pair": {"time_seconds": 4.2, "percent": 84.0},
            "Neigh": {"time_seconds": 0.6, "percent": 12.0},
            "Comm": {"time_seconds": 0.1, "percent": 2.0},
            "Other": {"time_seconds": 0.1, "percent": 2.0},
        },
        system_metrics={
            "time_real_sec": 5.0,
            "time_user_sec": 4.8,
            "instructions_retired": 345_000_000_000,
            "cycles_elapsed": 170_000_000_000,
        },
        notes=["IPC = 2.03 (compute-bound)", "pair_lj_cut_omp::eval is the hotspot"],
    )

    print("\n" + "-" * 60)
    print("Running optimization...")
    print("-" * 60)

    result = agent.optimize(
        action=action,
        profile=profile,
        target_file="src/OPENMP/pair_lj_cut_omp.cpp",
        additional_context={
            "backend": "openmp",
            "case_id": "melt_xxlarge",
        },
    )

    print("\n" + "=" * 60)
    print("Result")
    print("=" * 60)
    print(f"Status: {result.status}")
    print(f"Confidence: {result.confidence}")
    print(f"Expected improvement: {result.expected_improvement}")
    print(f"Total turns: {result.total_turns}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"\nRationale:\n{result.rationale}")

    if result.patch_diff:
        print(f"\nPatch diff:\n{result.patch_diff[:1000]}")
        if len(result.patch_diff) > 1000:
            print("... (truncated)")

    # Save conversation log for debugging
    log_file = project_root / "artifacts" / "agent_test_log.json"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as f:
        json.dump({
            "status": result.status,
            "rationale": result.rationale,
            "confidence": result.confidence,
            "total_turns": result.total_turns,
            "conversation_log": result.conversation_log,
        }, f, indent=2)
    print(f"\nConversation log saved to: {log_file}")


if __name__ == "__main__":
    main()
