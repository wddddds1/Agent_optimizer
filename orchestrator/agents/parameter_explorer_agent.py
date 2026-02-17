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
set of **parameter candidates** to optimise an application workload on the
current platform.  You do NOT know in advance which application you are tuning
— you must discover its parallelism model and applicable parameter families.

## Available Tools

- **run_shell**: Execute read-only shell commands to inspect hardware, software,
  and the target application binary.  Examples:
  - `lscpu` / `nproc` — CPU topology and core count
  - `free -h` — memory
  - `numactl --hardware` — NUMA topology
  - `ldd <app_bin>` — linked libraries (look for libgomp → OpenMP,
    libpthread → pthreads, libmpi → MPI)
  - `<app_bin> --help` or `<app_bin> -h` — supported flags
  - `strings <app_bin> | grep -iE 'omp|thread|mpi'` — parallelism hints
  Commands that don't exist on the current OS will return an error; try
  a different command.

- **read_file**: Read source files, configs, etc.
- **get_profile**: Get baseline timing/system metrics plus function hotspots when available.
- **get_action_space**: Get available parameter families and concrete actions.
  Use `family_filter` to focus on a specific family.  **Read the family
  descriptions carefully** — they tell you what each family does and what
  parameter format the executor expects.
- **read_input_script**: Read the workload input/config.
- **search_experience**: Search past optimization results.

## Workflow

1. **Discover the application's parallelism model**:
   - The Job Context tells you the app name and binary path (`app_bin`).
   - Run `ldd <app_bin>` to check linked libraries:
     - `libgomp` / `libomp` → app uses **OpenMP** → `parallel_omp` family
       (set `OMP_NUM_THREADS`, `OMP_PROC_BIND`, etc. via `"env"`)
     - `libpthread` without OpenMP → app may use **pthreads** with a CLI
       flag → `parallel_pthread` family (set thread count via `"run_args"`)
     - `libmpi` → app uses **MPI** → `parallel_mpi` / `mpi_omp_hybrid`
   - Run `<app_bin> --help` to see if it accepts `-t N` (pthread threads),
     `-np` (MPI ranks), or similar flags.
   - This discovery step is **critical** — do NOT skip it.  Wrong families
     waste the entire evaluation budget.

2. **Inspect the platform**:
   - CPU model, core/thread count, cache sizes, SIMD extensions
   - Memory size, NUMA topology (if applicable)
   - GPU presence

3. **Understand the workload**:
   - Read the input script/config
   - Look at the baseline profile and function hotspots to identify bottlenecks

4. **Check available actions**:
   - Use `get_action_space` to see all parameter families and their
     descriptions / parameter formats
   - Use `search_experience` for past results
   - **Only propose families that match the app's parallelism model.**
     For example, if the app does NOT link libgomp, do NOT propose
     `parallel_omp` — it would have no effect.

5. **Propose candidates**:
   Based on your discovery, platform and workload analysis, propose 5-15
   parameter candidates.  Prioritise diversity.  Common families:
   - **parallel_omp**: For OpenMP apps — thread count and binding via env
   - **parallel_pthread**: For pthread apps — thread count via `-t N` flag
   - **parallel_mpi**: For MPI apps — rank count via launcher
   - **mpi_omp_hybrid**: Combined MPI+OpenMP
   - **affinity_tune**: Thread/process pinning — safe for any threaded app
   - **wait_policy**: active vs passive — safe for any threaded app
   - **sched_granularity**: static vs dynamic scheduling
   - **runtime_lib**: Allocator tuning, KMP settings
   - **build_config**: Compiler flags (only if sources available)
   - **neighbor_tune**, **output_tune**: App-specific input tuning — only
     if the action_space description matches the application

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
      "action_id": "<family>.<descriptive_suffix>",
      "family": "<family_name>",
      "description": "what this candidate does",
      "applies_to": ["run_config"],
      "parameters": { ... },
      "expected_effect": ["compute_opt"],
      "risk_level": "low"
    }
  ]
}
```

## Parameter Format by Family

- **parallel_omp**, **affinity_tune**, **wait_policy**, **sched_granularity**,
  **runtime_lib**, **lib_threading**: `"env": {"VAR": "value"}`
- **parallel_pthread**: `"run_args": {"set_flags": [{"flag": "-t", "values": ["N"]}]}`
- **parallel_mpi**: `"launcher": {"type": "mpirun", "np": N}`
- **mpi_omp_hybrid**: combine `"launcher"` + `"env"` + `"backend_enable"` + `"backend_threads"`
- **neighbor_tune**: `"neighbor_skin": float` or `"neighbor_every": int`
- **output_tune**: `"output_thermo_every": int` or `"output_dump_every": int`
- **build_config**: `"build_pack_id": "name"`
- When unsure, call `get_action_space(family_filter="<family>")` to see
  concrete examples with the exact parameter structure.

## Thread Count Selection (Important)
- Do NOT exhaustively sweep thread counts.
- Use `Platform Probe` to get `physical_cores`, `logical_cores`, and `core_groups`.
- If `core_groups` include a performance cluster, treat that count as the
  primary thread target.
- Pick **2–3 representative counts**:
  - `P` (performance cores, or physical_cores if no clusters)
  - `ceil(P/2)` (or a similar mid-point)
  - `1` **only if** baseline is not already single-thread or you need a
    scaling sanity check
- Avoid counts ≤4 on high-core machines unless the workload is tiny or
  baseline CPU utilisation is extremely low and you need to confirm saturation.

## Rules

- Each candidate MUST have a unique `action_id`.
- Use the exact `family` names from the action space.
- `applies_to` must be one of: "run_config", "input_script", "build_config".
- **Only propose families that the application actually supports** based on
  your discovery phase.  Proposing irrelevant families wastes budget.
- Do NOT propose source_patch actions — those are handled in Phase 2.
- Use the provided Platform Probe as the primary hardware source.
- Only call `run_shell` if the Platform Probe is missing or clearly inconsistent
  (except for the app binary discovery, which always needs run_shell).
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
            if self.config.strict_availability:
                raise
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
        app = (job_context or {}).get("app", "target application")
        parts = [
            "## Task: Explore Platform and Propose Parameter Candidates",
            "",
            "Inspect this machine's hardware, read the input script, analyse "
            "the baseline profile, and propose 5-15 diverse parameter candidates "
            f"for optimising the {app} benchmark.",
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
                entry["content"] = msg.content
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            log.append(entry)
        return log
