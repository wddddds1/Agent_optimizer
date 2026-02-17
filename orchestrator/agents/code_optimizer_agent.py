"""Agentic code optimization agent with tool use and multi-turn conversation."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestrator.agent_llm import (
    AgentConfig,
    AgentLLMClient,
    AgentSession,
    MAX_TURNS_SENTINEL,
    TOOL_REPAIR_EXHAUSTED_SENTINEL,
)
from schemas.action_ir import ActionIR
from schemas.patch_proposal_ir import PatchProposal
from schemas.profile_report import ProfileReport
from skills.agent_tools import CodeOptimizationTools
from skills.profile_payload import build_profile_payload


SYSTEM_PROMPT = """\
You are an expert HPC code optimizer specializing in molecular dynamics simulations.
Your role is to produce source-level optimizations that the compiler CANNOT do on its own.

## Golden Rule
**Diagnose first, optimize second.**  Never propose a patch without first checking
what the compiler already does.  Most "obvious" micro-optimizations (unrolling,
vectorization hints, prefetch intrinsics, constant hoisting) are already performed
by -O3 -march=native.  Your value lies in **algorithmic and structural changes**
that require semantic understanding beyond the compiler's reach.

## Available Tools

### Platform & Hardware Inspection
- **run_shell**: Execute read-only shell commands (uname, sysctl, lscpu, nproc,
  compiler --version, otool, objdump, etc.).  Use this to query hardware topology,
  SIMD capabilities, cache sizes, NUMA layout — whatever you need.

### Code Exploration
- **read_file**: Read source files
- **grep**: Search for patterns across the codebase
- **find_files**: Find files by glob pattern
- **get_file_outline**: Get a structural outline of a file

### Code Understanding
- **get_type_definition**: Look up struct/class definitions
- **get_type_layout**: Get memory layout (size, alignment, padding) of a type
- **get_include_chain**: Trace header inclusion paths
- **get_function_signature**: Find function declarations
- **get_callers**: Find all call sites of a function
- **get_macro_definition**: Look up preprocessor macros

### Build Context
- **get_compile_flags**: Get compilation flags from compile_commands.json
- **get_build_target_files**: List source files in a build target
- **resolve_backend**: Map high-level LAMMPS style to actual implementation file

### Compiler Analysis (CRITICAL — use these before designing any patch)
- **get_compiler_opt_report**: Compile a file and show which loops the compiler
  vectorized, unrolled, inlined — and which it MISSED and why.  This is your
  primary diagnostic tool.
- **compile_single**: Quick compile check; set with_opt_report=true for reports.

### Performance Analysis
- **get_profile**: Get timing breakdown (pair, neigh, comm, etc.)
- **get_assembly**: Get disassembly of a function.  Use this to see actual
  SIMD instructions, scatter/gather, branch patterns.

### Code Modification
- **create_patch**: Create a unified diff patch
- **preview_patch**: Preview what a patch would look like when applied
- **find_anchor_occurrences**: Find where an anchor string occurs
- **apply_patch_dry_run**: Test if a patch applies cleanly

Tool-call contract for `create_patch` (STRICT):
- `arguments` MUST be a JSON object.
- It MUST include both required fields: `file_path` (string) and `changes` (array).
- Never call `create_patch` with `{}` or missing fields.

### Verification
- **compile**: Full build with the patch applied
- **compile_single**: Compile just the modified file
- **get_last_build_errors**: Get detailed error messages from the last failed build
- **run_benchmark**: Run the benchmark and compare against baseline

### Knowledge
- **get_reference_implementation**: Look at OPT/INTEL/GPU reference implementations
- **search_experience**: Search past optimization results and lessons learned

## Diagnostic Workflow (follow this order)

### Phase 1: Understand the Environment
1. Use **get_profile** to see where time is spent.
2. Use **run_shell** to query platform specifics if needed (cache sizes, SIMD width,
   NUMA topology).

### Phase 2: Understand the Code
3. **read_file** the hotspot function.  Read surrounding context too.
4. Look up type definitions (**get_type_definition**, **get_type_layout**) —
   especially `dbl3_t`, param structs, neighbor list types.
5. Check **get_reference_implementation** for OPT/INTEL versions that may already
   implement the optimization you are considering.

### Phase 3: Diagnose Compiler Behavior (MANDATORY before any patch design)
6. **get_compiler_opt_report** on the target file.  Answer these questions:
   - Which inner loops are already vectorized?  What SIMD width?
   - Which loops FAILED to vectorize?  What is the reason?
     (aliasing? data dependence? non-unit stride? function calls?)
   - What was inlined?  What was not?
   - Any loop unrolling decisions?
7. **get_assembly** for the hotspot function.  Look for:
   - Scatter/gather instructions → AoS memory layout problem
   - Branch instructions inside tight loops → branchless opportunity
   - SIMD width actually used vs hardware maximum
   - Redundant loads/stores → aliasing preventing optimization

### Phase 4: Design a Targeted Optimization
8. Based on the diagnosis, identify what the compiler CANNOT do and design
   a patch that addresses that specific gap.
9. Use **search_experience** to check if similar patches have been tried before
   and what their results were.

### Phase 5: Implement and Verify
10. **create_patch** with the minimum set of changes.
11. **compile_single** the modified file.
12. If compilation fails, use **get_last_build_errors**, fix, and retry.
13. Output your final result as JSON.

## Optimization Categories

### PROHIBITED — compiler -O3 -march=native already handles these:
- Manual loop unrolling (#pragma unroll, hand-unrolled bodies)
- `#pragma omp simd` on loops the compiler already vectorizes
- `__builtin_prefetch` intrinsics (hardware prefetcher + compiler prefetch
  are almost always sufficient; software prefetch often hurts)
- Constant/invariant hoisting out of loops (LICM is standard at -O2+)
- Manual strength reduction (multiply → shift, etc.)
- Obvious branch hints (__builtin_expect) in non-critical paths

### EFFECTIVE — things the compiler cannot do:
- **Data layout transformation** (AoS → SoA) — reduces scatter/gather overhead
- **Algorithmic changes** — half vs full neighbor list, Newton third law toggling
- **Aliasing resolution** — adding `__restrict__` qualifiers where the compiler
  cannot prove non-aliasing, enabling vectorization of blocked loops
- **Branchless computation** — replacing if/else with arithmetic in inner loops
  where the compiler doesn't know the branch is predictable
- **Memory access pattern improvement** — sorting neighbor lists by spatial
  locality, blocking for cache, reducing indirect addressing depth
- **Redundant computation elimination** — only when it requires semantic knowledge
  the compiler lacks (e.g., symmetry in force computation, precomputing values
  that depend on simulation-level invariants)
- **Special-case fast paths** — splitting loops by mask/flag value when the
  common case avoids expensive operations (e.g., special_pair_split for bonds)
- **Thread-local accumulation** — restructuring to reduce false sharing or
  atomic contention in OpenMP parallel regions
- **SIMD-friendly data packing** — aligning coefficient tables, padding arrays

## Critical Coding Rules
- ALWAYS read the target file before modifying it
- ALWAYS look up type definitions before using them
- For OpenMP code (files ending in _omp.cpp): coordinates use `dbl3_t` struct.
  Access as `x[j].x`, `x[j].y`, `x[j].z` — NOT `x[j][0]`.
- Include necessary headers (#include <cstdlib> for malloc/free)
- Make minimal, focused changes — don't refactor unrelated code
- If compilation fails, analyze the error and fix it

## Output Format
When you have completed your optimization, output a JSON object:
```json
{
  "status": "OK" | "NEED_MORE_CONTEXT" | "NO_OPTIMIZATION_POSSIBLE",
  "patch_diff": "unified diff of the changes",
  "rationale": "explanation of what was optimized and why",
  "diagnosis": "what the compiler already does and what gap this patch fills",
  "confidence": 0.0-1.0,
  "expected_improvement": "estimated improvement percentage"
}
```

If after diagnosis you determine the compiler already handles what you were going
to do, set status to "NO_OPTIMIZATION_POSSIBLE" and explain why in the rationale.
It is better to report no optimization than to submit a useless or harmful patch.
"""


@dataclass
class OptimizationResult:
    """Result of an optimization attempt."""
    status: str  # "OK", "NEED_MORE_CONTEXT", "NO_OPTIMIZATION_POSSIBLE", "ERROR"
    patch_diff: str
    rationale: str
    diagnosis: str  # what the compiler already does and what gap was found
    confidence: float
    expected_improvement: str
    conversation_log: List[Dict[str, Any]]
    total_turns: int
    total_tokens: int


class CodeOptimizerAgent:
    """Agentic code optimizer with tool use capabilities."""

    def __init__(
        self,
        config: AgentConfig,
        repo_root: Path,
        build_dir: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.repo_root = repo_root
        self.build_dir = build_dir or repo_root / "build"

        self.llm_client = AgentLLMClient(config)
        self.tools = CodeOptimizationTools(repo_root, build_dir)

    def optimize(
        self,
        action: ActionIR,
        profile: ProfileReport,
        target_file: str,
        hotspot_code: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Run the optimization agent for a specific action.

        Args:
            action: The optimization action to perform
            profile: Performance profile data
            target_file: The file to optimize
            hotspot_code: Pre-extracted hotspot code (optional)
            additional_context: Additional context to provide

        Returns:
            OptimizationResult with the patch and metadata
        """
        # Clear per-action patch state so a failed action cannot reuse stale diffs.
        self.tools.reset_session_state()

        # Set up profile data for tools
        profile_data = build_profile_payload(profile)
        self.tools.set_profile_data(profile_data)

        # Create session with system prompt
        session = self.llm_client.create_session(SYSTEM_PROMPT)

        # Register all tools
        self.llm_client.register_tools(session, self.tools.get_all_tools())

        # Construct the initial user message
        user_message = self._build_initial_message(
            action, profile_data, target_file, hotspot_code, additional_context
        )

        # Run the agent loop
        try:
            response = self.llm_client.chat(session, user_message, auto_execute_tools=True)

            # Parse the final response
            result = self._parse_result(response, session)
            if (
                result.status == "NEED_MORE_CONTEXT"
                and response != MAX_TURNS_SENTINEL
                and not response.startswith(TOOL_REPAIR_EXHAUSTED_SENTINEL)
            ):
                retry_response = self._request_final_json_retry(session)
                if retry_response:
                    retry_result = self._parse_result(retry_response, session)
                    if retry_result.status != "NEED_MORE_CONTEXT":
                        return retry_result
            return result

        except Exception as e:
            if self.config.strict_availability:
                raise
            return OptimizationResult(
                status="ERROR",
                patch_diff="",
                rationale=f"Agent error: {e}",
                diagnosis="",
                confidence=0.0,
                expected_improvement="0%",
                conversation_log=self._extract_conversation_log(session),
                total_turns=session.turn_count,
                total_tokens=session.total_tokens,
            )

    def _build_initial_message(
        self,
        action: ActionIR,
        profile_data: Dict[str, Any],
        target_file: str,
        hotspot_code: Optional[str],
        additional_context: Optional[Dict[str, Any]],
    ) -> str:
        """Build the initial message for the agent."""
        params = action.parameters or {}

        patch_family = params.get('patch_family', 'general')
        message_parts = [
            "## Optimization Task",
            f"**Action**: {action.action_id}",
            f"**Target file**: {target_file}",
            f"**Optimization type**: {patch_family}",
            "",
            "## MANDATORY CONSTRAINT",
            f"You MUST focus exclusively on the **{patch_family}** optimization strategy.",
            f"The action ID is `{action.action_id}` — your patch must implement THIS "
            "specific optimization, not a different one.",
            "If you cannot implement this specific strategy, report "
            'status="NO_OPTIMIZATION_POSSIBLE" — do NOT substitute a different optimization.',
        ]

        if action.description:
            message_parts.extend([
                "",
                "## Optimization Hint",
                action.description,
            ])

        # Inject rich context from deep code analysis when available
        if params.get("origin") == "deep_code_analysis":
            _diagnosis = params.get("diagnosis", "")
            _compiler_gap = params.get("compiler_gap", "")
            _mechanism = params.get("mechanism", "")
            _asm_evidence = params.get("assembly_evidence", "")
            _ref_code = params.get("reference_code", "")
            if _diagnosis:
                message_parts.extend(["", "## Deep Analysis Diagnosis", _diagnosis])
            if _mechanism:
                message_parts.extend(["", "## Optimization Mechanism", _mechanism])
            if _compiler_gap:
                message_parts.extend([
                    "", "## Compiler Gap (why the compiler cannot do this)",
                    _compiler_gap,
                ])
            if _asm_evidence:
                message_parts.extend([
                    "", "## Assembly Evidence", "```", _asm_evidence, "```",
                ])
            if _ref_code:
                message_parts.extend([
                    "", "## Reference Implementation",
                    "```cpp", _ref_code, "```",
                    "Adapt this reference to the target code. Do not copy verbatim.",
                ])

        message_parts.extend([
            "",
            "## Performance Profile",
            "```json",
            json.dumps(profile_data, indent=2),
            "```",
        ])

        # Navigation hints: lightweight file pointers so the agent knows
        # where to start reading.  The agent uses read_file to get code.
        nav_hints = (additional_context or {}).get("navigation_hints", [])
        if nav_hints:
            message_parts.extend(["", "## Navigation Hints (use read_file to explore)"])
            for hint in nav_hints:
                parts = [f"- **{hint.get('path')}**"]
                if hint.get("total_lines"):
                    parts.append(f"({hint['total_lines']} lines)")
                if hint.get("function_signature"):
                    parts.append(f"— hotspot: `{hint['function_signature']}`")
                if hint.get("hotspot_line"):
                    parts.append(f"at line {hint['hotspot_line']}")
                if hint.get("function_start"):
                    parts.append(f"(function starts line {hint['function_start']})")
                message_parts.append(" ".join(parts))
            message_parts.extend([
                "",
                "Start by calling `read_file` on the target file to see the actual code.",
            ])
        elif hotspot_code:
            # Legacy fallback: pre-extracted code snippet
            message_parts.extend([
                "",
                "## Hotspot Code Preview",
                "```cpp",
                hotspot_code[:2000],
                "```",
            ])

        if additional_context:
            # Include failure feedback from previous iterations
            prev_failures = additional_context.get("previous_failures")
            if prev_failures:
                message_parts.extend([
                    "",
                    "## Previous Failed Patches (DO NOT repeat these)",
                    "The following source patches were tried in earlier iterations "
                    "and made performance WORSE.  Do NOT propose similar changes:",
                ])
                for failure in prev_failures:
                    message_parts.append(f"- {failure}")

            # Include other context (exclude navigation_hints — already rendered above)
            other_ctx = {
                k: v for k, v in additional_context.items()
                if k not in ("previous_failures", "navigation_hints")
            }
            if other_ctx:
                message_parts.extend([
                    "",
                    "## Additional Context",
                    json.dumps(other_ctx, indent=2),
                ])

        message_parts.extend([
            "",
            "## Instructions — Follow the Diagnostic Workflow",
            "1. **get_profile** — confirm where time is spent",
            "2. **read_file** — read the target file and understand the hotspot code",
            "3. **get_type_definition / get_type_layout** — look up key types "
            "(dbl3_t, param structs, etc.)",
            "4. **get_compiler_opt_report** on the target file — see what the "
            "compiler ALREADY optimizes and what it MISSES. This step is MANDATORY.",
            "5. **get_assembly** for the hotspot function — check actual SIMD usage, "
            "scatter/gather, branch patterns",
            "6. Identify the **specific gap** the compiler cannot fill",
            "7. Check **search_experience** for similar past patches and their results",
            "8. Design a targeted patch that addresses the identified gap",
            "9. **create_patch** to produce a concrete diff",
            "10. Run **compile_single** at most ONCE for sanity verification "
            "(skip repeated compile attempts)",
            "11. Output final result as JSON immediately after patch+sanity check",
            "",
            "## Tool Call Contract (STRICT)",
            "When calling `create_patch`, arguments MUST be a JSON object containing:",
            '- `file_path`: string',
            '- `changes`: array',
            "Do NOT call `create_patch` with `{}` or missing fields.",
            "If you receive a tool argument error, fix arguments and retry immediately.",
            "",
            "## Termination Rules (STRICT)",
            "- After the first successful `create_patch`, you MUST output final JSON "
            "within the next 2 assistant turns.",
            "- If `compile_single` reports 'no compile_commands.json' or fallback-mode "
            "success/failure, do NOT call `compile_single` again.",
            "- Do NOT keep exploring unrelated files after a valid patch is created. "
            "Finalize with JSON.",
            "",
            "IMPORTANT: If the compiler optimization report shows that your intended "
            "optimization is already performed, DO NOT create a patch. Instead, report "
            'status="NO_OPTIMIZATION_POSSIBLE" with an explanation.',
        ])

        return "\n".join(message_parts)

    def _parse_result(
        self,
        response: str,
        session: AgentSession,
    ) -> OptimizationResult:
        """Parse the agent's final response into an OptimizationResult."""
        conversation_log = self._extract_conversation_log(session)

        if response.startswith(TOOL_REPAIR_EXHAUSTED_SENTINEL):
            return OptimizationResult(
                status="NEED_MORE_CONTEXT",
                patch_diff="",
                rationale=(
                    "Agent produced repeated invalid tool calls; repair budget exhausted. "
                    + response
                ),
                diagnosis="",
                confidence=0.0,
                expected_improvement="0%",
                conversation_log=conversation_log,
                total_turns=session.turn_count,
                total_tokens=session.total_tokens,
            )

        # Strict mode: no fallback to tool-side patch state without final JSON.
        if response == MAX_TURNS_SENTINEL:
            return OptimizationResult(
                status="NEED_MORE_CONTEXT",
                patch_diff="",
                rationale=(
                    "Agent exhausted maximum turns before returning final JSON output."
                ),
                diagnosis="",
                confidence=0.0,
                expected_improvement="0%",
                conversation_log=conversation_log,
                total_turns=session.turn_count,
                total_tokens=session.total_tokens,
            )

        # Try to parse JSON from response
        json_result = self._extract_json(response)

        if json_result:
            _status = json_result.get("status", "NEED_MORE_CONTEXT")
            _patch = json_result.get("patch_diff", "")
            if _status == "OK" and not _patch:
                return OptimizationResult(
                    status="NEED_MORE_CONTEXT",
                    patch_diff="",
                    rationale="Invalid final JSON: status=OK but patch_diff is empty.",
                    diagnosis=json_result.get("diagnosis", ""),
                    confidence=0.0,
                    expected_improvement="0%",
                    conversation_log=conversation_log,
                    total_turns=session.turn_count,
                    total_tokens=session.total_tokens,
                )
            return OptimizationResult(
                status=_status,
                patch_diff=_patch,
                rationale=json_result.get("rationale", ""),
                diagnosis=json_result.get("diagnosis", ""),
                confidence=float(json_result.get("confidence", 0.5)),
                expected_improvement=json_result.get("expected_improvement", "unknown"),
                conversation_log=conversation_log,
                total_turns=session.turn_count,
                total_tokens=session.total_tokens,
            )

        return OptimizationResult(
            status="NEED_MORE_CONTEXT",
            patch_diff="",
            rationale=(
                "Invalid final response: missing machine-readable JSON result. "
                + response[:500]
            ),
            diagnosis="",
            confidence=0.0,
            expected_improvement="0%",
            conversation_log=conversation_log,
            total_turns=session.turn_count,
            total_tokens=session.total_tokens,
        )

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text, handling markdown code blocks."""
        if not text:
            return None
        payloads: List[Dict[str, Any]] = []
        candidates: List[str] = []
        for pattern in (
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
        ):
            candidates.extend(re.findall(pattern, text))
        candidates.append(text)
        for candidate in candidates:
            for parsed in self._iter_json_objects(candidate):
                if not isinstance(parsed, dict):
                    continue
                if {"status", "patch_diff", "rationale"} & set(parsed.keys()):
                    payloads.append(parsed)
        if not payloads:
            return None
        payloads.sort(key=self._result_payload_score, reverse=True)
        return payloads[0]

    def _iter_json_objects(self, text: str):
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(text):
            start = text.find("{", idx)
            if start < 0:
                break
            try:
                obj, end = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                idx = start + 1
                continue
            if isinstance(obj, dict):
                yield obj
            idx = start + max(1, end)

    def _result_payload_score(self, payload: Dict[str, Any]) -> float:
        score = 0.0
        status = str(payload.get("status", "")).upper()
        if status == "OK":
            score += 8.0
        if status == "NO_OPTIMIZATION_POSSIBLE":
            score += 5.0
        patch_diff = str(payload.get("patch_diff", ""))
        if patch_diff:
            score += 10.0 + min(len(patch_diff) / 5000.0, 4.0)
        if str(payload.get("rationale", "")).strip():
            score += 1.0
        if str(payload.get("diagnosis", "")).strip():
            score += 1.0
        return score

    def _request_final_json_retry(self, session: AgentSession) -> Optional[str]:
        if session.turn_count >= self.config.max_turns:
            return None
        try:
            return self.llm_client.chat(
                session,
                (
                    "Return exactly one final JSON object now. "
                    "No markdown, no tool calls. "
                    "Required keys: status, patch_diff, rationale, diagnosis, confidence, expected_improvement."
                ),
                auto_execute_tools=False,
            )
        except Exception:
            return None

    def _extract_conversation_log(self, session: AgentSession) -> List[Dict[str, Any]]:
        """Extract a simplified conversation log for debugging."""
        log = []
        for msg in session.messages:
            entry = {"role": msg.role}
            if msg.content:
                # Truncate long content
                entry["content"] = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "raw_arguments": tc.raw_arguments,
                        "parse_error": tc.parse_error,
                    }
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            log.append(entry)
        return log


def create_optimizer_agent(
    repo_root: Path,
    api_key_env: str = "DEEPSEEK_API_KEY",
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    build_dir: Optional[Path] = None,
) -> CodeOptimizerAgent:
    """Factory function to create a CodeOptimizerAgent with default config."""
    config = AgentConfig(
        enabled=True,
        api_key_env=api_key_env,
        base_url=base_url,
        model=model,
        temperature=0.2,
        max_tokens=4096,
        max_turns=15,
    )
    return CodeOptimizerAgent(config, repo_root, build_dir)
