"""Agentic parameter exploration agent for Phase 1 of the two-phase workflow.

This agent uses tools (run_shell, read_file, get_profile, get_action_space, etc.)
to inspect the platform, understand the workload, and propose a set of parameter
candidates (ActionIR) for single-batch evaluation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestrator.agent_llm import AgentConfig, AgentLLMClient, AgentSession
from schemas.action_ir import ActionIR
from skills.parameter_explorer_tools import ParameterExplorerTools


SYSTEM_PROMPT = """\
You are an expert HPC performance engineer.  Your goal is to propose a diverse
set of **parameter candidates** (runtime config, build config, input tuning) to
optimise a LAMMPS molecular dynamics simulation on the current platform.

## Available Tools

- **run_shell**: Execute read-only shell commands to inspect the hardware and
  software environment.  Examples:
  - `uname -a` — OS and kernel
  - `sysctl hw.physicalcpu hw.logicalcpu` (macOS) / `nproc` (Linux) — CPU count
  - `sysctl hw.l1dcachesize hw.l2cachesize hw.l3cachesize` — cache sizes
  - `sysctl machdep.cpu.brand_string` — CPU model
  - `gcc --version` / `clang --version` — compiler version
  - `sw_vers` (macOS) / `lsb_release -a` (Linux) — OS version
  - `system_profiler SPHardwareDataType` (macOS) — hardware summary
  - `lscpu` (Linux) — CPU topology
  - `free -h` (Linux) — memory
  - `numactl --hardware` (Linux) — NUMA topology
  - `nvidia-smi -L` — GPU list (if any)
  Commands that don't exist on the current OS will return an error; just try
  a different command.

- **read_file**: Read source files, configs, CMake files, etc.
- **get_profile**: Get baseline timing breakdown (pair, neigh, comm, etc.)
- **get_action_space**: Get available parameter families and concrete actions.
  Use `family_filter` to focus on a specific family.
- **read_input_script**: Read the LAMMPS input script for the benchmark case.
- **search_experience**: Search past optimization results.

## Workflow

1. **Inspect the platform**:
   - CPU model, core/thread count, cache sizes, SIMD extensions
   - Memory size, NUMA topology (if applicable)
   - GPU presence
   - OS type, compiler version

2. **Understand the workload**:
   - Read the input script to understand simulation type, system size, pair style,
     neighbor settings, output frequency, run length
   - Look at the baseline profile to identify bottleneck categories

3. **Check available actions**:
   - Use `get_action_space` to see what parameter families exist
   - Use `search_experience` to see what has worked or failed before

4. **Propose candidates**:
   Based on your platform and workload analysis, propose 5-15 parameter
   candidates.  Prioritise diversity across families:
   - **parallel_omp**: Thread count and binding (match physical core count,
     try different bind/places combos)
   - **parallel_mpi**: MPI rank count (2, 4, 8 — match socket count for
     NUMA locality, use physical_cores/threads_per_rank for hybrid).
     Only propose if MPI runtime is available on the system.
   - **mpi_omp_hybrid**: Combined MPI+OpenMP (e.g. 2 ranks × 4 threads
     on 8-core system; adjust rank/thread ratio based on comm vs compute
     bottleneck — high comm → fewer ranks, high compute → more threads).
     Only propose if MPI runtime is available.
   - **neighbor_tune**: Skin distance (smaller = less neighbor overhead,
     larger = fewer rebuilds)
   - **wait_policy**: active vs passive (active is better for short idle,
     passive for long idle)
   - **sched_granularity**: static vs dynamic (static for uniform load,
     dynamic for unbalanced)
   - **build_config**: Optimisation level, LTO, fast-math
   - **runtime_lib**: Allocator tuning, KMP settings
   - **output_tune**: Reduce output frequency if it's a bottleneck
   - **lib_threading**: BLAS/FFT thread limits

## Output Format

When done, output a JSON object:
```json
{
  "status": "OK",
  "platform_summary": "brief hardware/OS summary",
  "workload_summary": "brief workload characterization",
  "bottleneck_analysis": "what the profile tells us",
  "rationale": "overall optimization strategy",
  "candidates": [
    {
      "action_id": "parallel_omp.t8_close_cores",
      "family": "parallel_omp",
      "description": "8 threads, close binding, cores placement",
      "applies_to": ["run_config"],
      "parameters": {
        "env": {
          "OMP_NUM_THREADS": "8",
          "OMP_PROC_BIND": "close",
          "OMP_PLACES": "cores",
          "OMP_DYNAMIC": "false"
        }
      },
      "expected_effect": ["compute_opt", "mem_locality"],
      "risk_level": "low"
    }
  ]
}
```

## Rules

- Each candidate MUST have a unique `action_id`.
- Use the exact `family` names from the action space.
- `applies_to` must be one of: "run_config", "input_script", "build_config".
- `parameters` must match what the executor expects for the family.
  For `parallel_omp`, `affinity_tune`, `wait_policy`, `sched_granularity`,
  `runtime_lib`, `lib_threading`: use `"env": {"VAR": "value"}`.
  For `neighbor_tune`: use `"neighbor_skin": float` or `"neighbor_every": int`.
  For `output_tune`: use `"output_thermo_every": int` or `"output_dump_every": int`.
  For `build_config`: use `"build_pack_id": "name"`.
- You may propose custom action_ids (e.g. "parallel_omp.t8_spread_threads")
  as long as the family and parameter structure is correct.
- For `parallel_mpi`: use `"launcher": {"type": "mpirun", "np": N}` in parameters.
- For `mpi_omp_hybrid`: combine `"launcher"` with `"env": {"OMP_NUM_THREADS": "N"}`
  and `"backend_enable": "mpi_omp"` and `"backend_threads": N`.
- Rank count should not exceed physical core count.
- For hybrid, np × OMP_NUM_THREADS should not exceed physical core count.
- Do NOT propose source_patch actions — those are handled in Phase 2.
- Use the provided Platform Probe as the primary hardware source.
- Only call `run_shell` if the Platform Probe is missing or clearly inconsistent.
- Prefer actions that the experience database shows have worked before.
- Avoid actions that the experience database shows have been harmful.
"""


@dataclass
class ExplorationResult:
    """Result of the parameter exploration phase."""
    status: str  # "OK" | "ERROR"
    candidates: List[ActionIR]
    platform_summary: str
    workload_summary: str
    bottleneck_analysis: str
    rationale: str
    conversation_log: List[Dict[str, Any]]
    total_turns: int
    total_tokens: int


class ParameterExplorerAgent:
    """Agentic parameter explorer that uses tools to inspect the platform
    and propose optimization candidates."""

    def __init__(
        self,
        config: AgentConfig,
        repo_root: Path,
        input_script_path: Optional[Path] = None,
        experience_db: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.repo_root = repo_root
        self.llm_client = AgentLLMClient(config)
        self.tools = ParameterExplorerTools(
            repo_root, input_script_path, experience_db
        )

    def explore(
        self,
        profile: Dict[str, Any],
        action_space: Dict[str, Any],
        job_context: Optional[Dict[str, Any]] = None,
    ) -> ExplorationResult:
        """Run the parameter exploration agent.

        Args:
            profile: Baseline performance profile dict.
            action_space: The full action_space.yaml content.
            job_context: Optional context (case_id, app, tags, etc.)

        Returns:
            ExplorationResult with proposed candidates.
        """
        self.tools.set_profile_data(profile)
        self.tools.set_action_space(action_space)

        platform_probe = self.tools.probe_platform()
        session = self.llm_client.create_session(SYSTEM_PROMPT)
        self.llm_client.register_tools(session, self.tools.get_all_tools())

        user_message = self._build_initial_message(profile, job_context, platform_probe)

        try:
            response = self.llm_client.chat(
                session, user_message, auto_execute_tools=True
            )
            return self._parse_result(response, session)
        except Exception as exc:
            return ExplorationResult(
                status="ERROR",
                candidates=[],
                platform_summary="",
                workload_summary="",
                bottleneck_analysis="",
                rationale=f"Agent error: {exc}",
                conversation_log=self._extract_conversation_log(session),
                total_turns=session.turn_count,
                total_tokens=session.total_tokens,
            )

    def _build_initial_message(
        self,
        profile: Dict[str, Any],
        job_context: Optional[Dict[str, Any]],
        platform_probe: Optional[Dict[str, str]] = None,
    ) -> str:
        parts = [
            "## Task: Explore Platform and Propose Parameter Candidates",
            "",
            "Inspect this machine's hardware, read the input script, analyse "
            "the baseline profile, and propose 5-15 diverse parameter candidates "
            "for optimising the LAMMPS benchmark.",
        ]

        if job_context:
            parts.extend([
                "",
                "## Job Context",
                f"```json",
                json.dumps(job_context, indent=2),
                "```",
            ])

        if platform_probe:
            parts.extend([
                "",
                "## Platform Probe (run_shell results)",
                "```json",
                json.dumps(platform_probe, indent=2),
                "```",
            ])

        parts.extend([
            "",
            "## Baseline Profile (summary)",
            "```json",
            json.dumps(profile, indent=2),
            "```",
            "",
            "## Instructions",
            "1. Use **run_shell** to inspect the platform (CPU, memory, caches, "
            "NUMA, GPU, compiler)",
            "2. Use **read_input_script** to understand the simulation",
            "3. Use **get_profile** for the full baseline data",
            "4. Use **get_action_space** to see available parameter families",
            "5. Use **search_experience** for past results",
            "6. Propose 5-15 candidates as JSON",
        ])

        return "\n".join(parts)

    def _parse_result(
        self, response: str, session: AgentSession
    ) -> ExplorationResult:
        conversation_log = self._extract_conversation_log(session)
        json_result = self._extract_json(response)

        # If the direct response didn't contain JSON (e.g. max turns reached),
        # scan the conversation history for a JSON block with candidates.
        if not json_result:
            for msg in reversed(session.messages):
                if msg.content:
                    json_result = self._extract_json(msg.content)
                    if json_result:
                        break

        if json_result:
            candidates = self._parse_candidates(
                json_result.get("candidates", [])
            )
            return ExplorationResult(
                status=json_result.get("status", "OK"),
                candidates=candidates,
                platform_summary=json_result.get("platform_summary", ""),
                workload_summary=json_result.get("workload_summary", ""),
                bottleneck_analysis=json_result.get("bottleneck_analysis", ""),
                rationale=json_result.get("rationale", ""),
                conversation_log=conversation_log,
                total_turns=session.turn_count,
                total_tokens=session.total_tokens,
            )

        return ExplorationResult(
            status="ERROR",
            candidates=[],
            platform_summary="",
            workload_summary="",
            bottleneck_analysis="",
            rationale=f"Failed to parse agent response: {response[:500]}",
            conversation_log=conversation_log,
            total_turns=session.turn_count,
            total_tokens=session.total_tokens,
        )

    def _parse_candidates(
        self, raw_candidates: List[Dict[str, Any]]
    ) -> List[ActionIR]:
        """Parse raw candidate dicts into ActionIR objects, skipping invalid."""
        result: List[ActionIR] = []
        for raw in raw_candidates:
            try:
                # Ensure required fields
                if not raw.get("action_id") or not raw.get("family"):
                    continue
                # Skip source_patch if the agent proposes them anyway
                if raw.get("family") == "source_patch":
                    continue
                action = ActionIR(
                    action_id=raw["action_id"],
                    family=raw["family"],
                    description=raw.get("description", ""),
                    applies_to=raw.get("applies_to", ["run_config"]),
                    parameters=raw.get("parameters", {}),
                    expected_effect=raw.get("expected_effect", []),
                    risk_level=raw.get("risk_level", "low"),
                )
                result.append(action)
            except Exception:
                continue
        return result

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text, handling markdown code blocks."""
        patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
            r"\{[\s\S]*\}",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    clean = match.strip()
                    if not clean.startswith("{"):
                        clean = "{" + clean.split("{", 1)[-1]
                    if not clean.endswith("}"):
                        clean = clean.rsplit("}", 1)[0] + "}"
                    parsed = json.loads(clean)
                    # Must contain candidates to be the right JSON
                    if "candidates" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    continue
        return None

    def _extract_conversation_log(
        self, session: AgentSession
    ) -> List[Dict[str, Any]]:
        log: List[Dict[str, Any]] = []
        for msg in session.messages:
            entry: Dict[str, Any] = {"role": msg.role}
            if msg.content:
                entry["content"] = (
                    msg.content[:500] + "..."
                    if len(msg.content) > 500
                    else msg.content
                )
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            log.append(entry)
        return log
