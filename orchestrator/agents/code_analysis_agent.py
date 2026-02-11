"""Deep code analysis agent for Phase 2 initialization.

Runs ONCE at the start of Phase 2 to produce a ranked list of optimization
opportunities through interactive code exploration.  Unlike the single-shot
PatchPlanner, this agent uses tools to freely explore the codebase, study
reference implementations, check compiler reports, and examine assembly.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestrator.agent_llm import (
    AgentConfig,
    AgentLLMClient,
    AgentSession,
    MAX_TURNS_SENTINEL,
)
from schemas.code_analysis_ir import DeepCodeAnalysisResult
from skills.agent_tools import CodeOptimizationTools


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _build_system_prompt(
    patch_families: Optional[Dict[str, Any]] = None,
) -> str:
    families_section = ""
    if patch_families:
        lines: List[str] = []
        for item in (patch_families or {}).get("families", []):
            if not isinstance(item, dict):
                continue
            fid = item.get("id", "")
            desc = item.get("description", "")
            ref = item.get("reference_file", "")
            covered = item.get("compiler_covered", False)
            if covered:
                lines.append(f"  - {fid}: {desc} [COMPILER HANDLES THIS — skip]")
            elif ref:
                lines.append(f"  - {fid}: {desc} (ref: {ref})")
            else:
                lines.append(f"  - {fid}: {desc}")
        if lines:
            families_section = (
                "\n\n## Known Optimization Patterns (reference only)\n"
                "These are previously identified patterns.  You may reference them "
                "by setting `family_hint`, but you are NOT limited to these.  "
                "Discovering novel optimizations is equally valuable.\n"
                + "\n".join(lines)
            )

    return f"""\
You are a deep code analysis agent specialising in HPC performance optimisation.
Your task is to THOROUGHLY explore a codebase to discover ALL viable source-level
optimisation opportunities.  You are analysing LAMMPS, a molecular dynamics code.

## Your Mission

Produce a comprehensive, RANKED list of optimisation opportunities for the hotspot
code.  You have tools to:
- Read complete source files and trace call chains
- Examine type definitions, data layouts, and memory access patterns
- Study compiler optimisation reports to see what IS and IS NOT optimised
- Read disassembly to observe actual SIMD usage and branch patterns
- Find reference implementations (OPT/INTEL variants) for comparison
- Query past optimisation experience

You are NOT limited to predefined optimisation families.  Discover opportunities
organically through systematic exploration.

## Mandatory Analysis Workflow

Follow this workflow IN ORDER.  Use tools extensively — you have budget for 30+
tool calls.

### Phase 1: Understand the Architecture (3-4 tool calls)
1. **get_profile** — Where is time spent?  Note pair/neigh/comm ratios.
2. **read_file** the main hotspot file(s) identified in the profile.
3. **get_file_outline** for each hotspot file — overall structure.
4. **get_callers** for the main compute/eval function — call chain from
   the simulation loop to the hot inner loop.

### Phase 2: Map Data Structures (3-5 tool calls)
5. **get_type_definition** and **get_type_layout** for ALL types in the hot path:
   - Coordinate types (dbl3_t, vec3_t, double**)
   - Coefficient types (cutsq, lj1, lj2 — how are they stored?)
   - Neighbour list types (NeighList, firstneigh, numneigh)
   - Force accumulation types (f[i] patterns, fxtmp/fytmp/fztmp)
6. Document the access pattern for each: AoS vs SoA?  How many cache lines
   per access?  Indirection (pointer-to-pointer)?

### Phase 3: Study Reference Implementations (2-3 tool calls)
7. **get_reference_implementation** — find OPT / INTEL versions of the hotspot.
8. **read_file** the reference implementation fully.  Identify EVERY optimisation
   technique it uses and whether each could apply to the current code.
9. **grep** for specific patterns from the reference (struct definitions,
   specialised loop structures, etc.)

### Phase 4: Compiler Baseline — CRITICAL (2-4 tool calls)
10. **get_compiler_opt_report** on the hotspot file — see what the compiler DOES.
    For each optimisation you are considering, check: does the compiler already
    do it?
11. **get_assembly** for the hotspot function.  Look for:
    - SIMD instructions (vmulpd, vfmadd = vectorised; mulsd = scalar)
    - Scatter/gather (vgatherdpd = indirect access problem)
    - Branches inside tight loops (jne, je)
    - Redundant loads (same address loaded multiple times)
12. **get_compile_flags** — optimisation level, target arch, enabled packages.

### Phase 5: Deep Pattern Mining (3-6 tool calls)
13. **Compare** current code against the reference OPT/INTEL version line-by-line.
    What techniques does the reference use that the current code does not?
14. For each potential optimisation, VERIFY with the compiler report and assembly
    that it is NOT already being done.
15. **search_experience** — has this been tried before?  What was the result?
16. For novel patterns beyond the reference, examine:
    - Unnecessary memory indirections that could be flattened
    - Branches in the inner loop that could be eliminated
    - Data dependencies preventing vectorisation
    - Memory access patterns that could be improved
    - Redundant computations visible only with semantic understanding

### Phase 6: Rank and Output (final turn)
17. Produce your final JSON with ALL discovered opportunities, ranked by
    estimated impact, confidence, risk, and implementation complexity.

## Evidence Requirements

For EACH opportunity you MUST have:
- **compiler_gap**: A specific explanation of why the compiler cannot do this.
  If you cannot articulate the gap, the optimisation is likely already handled.
- **code_context**: Actual source code from the hotspot (copy from read_file).
- At least ONE of: assembly_evidence, compiler_report_evidence, or reference_code.

## What Makes a GOOD Opportunity
1. The compiler cannot do it: data layout changes, algorithm changes, semantic
   optimisations requiring domain knowledge.
2. Evidence-based: you saw it in the assembly, compiler report, or reference impl.
3. Targeted: you know the exact file, function, and lines to modify.
4. Measurable: you can estimate impact from the profile data.

## What Makes a BAD Opportunity
1. Compiler already does it: loop unrolling, constant hoisting, simple branch hints
   — these are done by -O3 -march=native.
2. No evidence: "this might help" without checking assembly or compiler report.
3. Vague target: "optimise the inner loop" without specifying the mechanism.
4. Previously failed: search_experience shows it was tried and failed.
{families_section}

## Output Format

When analysis is complete, output a single JSON object:
```json
{{
  "status": "OK",
  "architecture_summary": "...",
  "call_chain": [
    {{"function_name": "eval", "file_path": "src/OPENMP/pair_lj_cut_omp.cpp",
      "line_number": 75, "time_share_pct": 84.0, "description": "Hot inner loop"}}
  ],
  "data_structures": [
    {{"type_name": "dbl3_t", "defined_in": "src/OPENMP/omp_compat.h",
      "access_pattern": "AoS", "hotspot_usage": "x[j].x/y/z coordinate access",
      "cache_behavior": "24 bytes, fits in one cache line",
      "optimization_relevance": "Determines vectorisation pattern"}}
  ],
  "compiler_insights": [
    {{"file_path": "...", "vectorized_loops": ["..."], "missed_optimizations": ["..."],
      "simd_width_used": "...", "aliasing_issues": ["..."], "inlining_decisions": ["..."]}}
  ],
  "compiler_baseline_summary": "...",
  "bottleneck_diagnosis": "...",
  "hotspot_files": ["..."],
  "opportunities": [
    {{
      "opportunity_id": "param_table_pack_1",
      "title": "Pack coefficient arrays into cache-aligned struct",
      "category": "data_layout",
      "family_hint": "param_table_pack",
      "target_files": ["src/OPENMP/pair_lj_cut_omp.cpp"],
      "target_functions": ["eval"],
      "diagnosis": "Six separate 2D arrays each require separate cache line loads...",
      "mechanism": "Pack into single 64-byte aligned struct like OPT reference...",
      "compiler_gap": "Compiler cannot merge accesses across independently allocated arrays.",
      "evidence": ["Assembly shows 6 separate load sequences per iteration"],
      "code_context": "const double * _noalias const cutsqi = cutsq[itype];...",
      "reference_code": "typedef struct {{ double cutsq,lj1,lj2,lj3,lj4,offset; double _pad[2]; }} fast_alpha_t;",
      "assembly_evidence": "Multiple movsd from different base addresses per neighbour",
      "compiler_report_evidence": "",
      "estimated_impact": "high",
      "confidence": 0.8,
      "risk_level": "medium",
      "expected_effect": ["mem_locality", "compute_opt"],
      "depends_on": [],
      "conflicts_with": [],
      "composable_with": ["special_pair_split_1"],
      "implementation_complexity": "medium",
      "lines_of_change_estimate": 40,
      "priority_rank": 1
    }}
  ],
  "recommended_sequence": ["param_table_pack_1", "special_pair_split_1"],
  "strategy_rationale": "Start with param_table_pack for highest impact...",
  "total_files_explored": 5,
  "total_functions_analyzed": 3,
  "exploration_notes": []
}}
```

IMPORTANT: Do NOT stop after finding one opportunity.  Continue through ALL phases.
A thorough analysis typically finds 3-8 opportunities.

IMPORTANT: If running low on turns, IMMEDIATELY output your current findings as
JSON.  Partial results are better than no results.
"""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DeepAnalysisResult:
    """Result of the deep code analysis phase."""

    status: str  # "OK" | "PARTIAL" | "ERROR"
    analysis: Optional[DeepCodeAnalysisResult]
    conversation_log: List[Dict[str, Any]] = field(default_factory=list)
    total_turns: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Tool names to include (analysis-only subset of CodeOptimizationTools)
# ---------------------------------------------------------------------------

_ANALYSIS_TOOL_NAMES = {
    # Code Exploration
    "read_file",
    "grep",
    "find_files",
    "get_file_outline",
    # Code Understanding
    "get_type_definition",
    "get_type_layout",
    "get_include_chain",
    "get_function_signature",
    "get_callers",
    "get_macro_definition",
    # Build Context
    "get_compile_flags",
    "get_build_target_files",
    "resolve_backend",
    # Performance Analysis
    "get_profile",
    "get_assembly",
    # Compiler Analysis
    "get_compiler_opt_report",
    "compile_single",
    # Knowledge
    "get_reference_implementation",
    "search_experience",
    # Platform
    "run_shell",
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DeepCodeAnalysisAgent:
    """Agentic deep code analysis for Phase 2 initialisation.

    Runs ONCE at the start of Phase 2 to produce a ranked list of
    optimisation opportunities through interactive code exploration.
    """

    def __init__(
        self,
        config: AgentConfig,
        repo_root: Path,
        build_dir: Optional[Path] = None,
        experience_db: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.repo_root = repo_root
        self.build_dir = build_dir or repo_root / "build"
        self.llm_client = AgentLLMClient(config)
        self.tools = CodeOptimizationTools(repo_root, build_dir, experience_db)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        profile: Dict[str, Any],
        hotspot_files: List[str],
        system_caps: Dict[str, Any],
        patch_families: Optional[Dict[str, Any]] = None,
        experience_hints: Optional[List[Dict[str, Any]]] = None,
        backend_variant: Optional[str] = None,
        input_script_path: Optional[str] = None,
    ) -> DeepAnalysisResult:
        """Run the deep code analysis.

        Args:
            profile: Baseline performance profile dict (timing_breakdown, etc.).
            hotspot_files: Paths (relative to repo_root) of performance-critical files.
            system_caps: Hardware topology and capabilities.
            patch_families: Optional known patch family definitions (reference only).
            experience_hints: Historical optimisation results.
            backend_variant: e.g., "omp".
            input_script_path: Path to the LAMMPS input script.

        Returns:
            DeepAnalysisResult with ranked optimisation opportunities.
        """
        self.tools.set_profile_data(profile)
        if input_script_path:
            self.tools.set_benchmark_input(input_script_path)

        system_prompt = _build_system_prompt(patch_families)
        session = self.llm_client.create_session(system_prompt)

        analysis_tools = self._get_analysis_tools()
        self.llm_client.register_tools(session, analysis_tools)

        user_message = self._build_initial_message(
            profile=profile,
            hotspot_files=hotspot_files,
            system_caps=system_caps,
            experience_hints=experience_hints or [],
            backend_variant=backend_variant,
        )

        try:
            response = self.llm_client.chat(
                session, user_message, auto_execute_tools=True
            )
            return self._parse_result(response, session)
        except Exception as exc:
            return DeepAnalysisResult(
                status="ERROR",
                analysis=None,
                conversation_log=self._extract_conversation_log(session),
                total_turns=session.turn_count,
                total_tokens=session.total_tokens,
            )

    # ------------------------------------------------------------------
    # Tool subset selection
    # ------------------------------------------------------------------

    def _get_analysis_tools(self) -> List:
        """Return the analysis-only subset of CodeOptimizationTools.

        Excludes code modification tools (create_patch, preview_patch, etc.)
        since this agent only analyses — it does not modify code.
        """
        return [
            t
            for t in self.tools.get_all_tools()
            if t.name in _ANALYSIS_TOOL_NAMES
        ]

    # ------------------------------------------------------------------
    # Initial message
    # ------------------------------------------------------------------

    def _build_initial_message(
        self,
        profile: Dict[str, Any],
        hotspot_files: List[str],
        system_caps: Dict[str, Any],
        experience_hints: List[Dict[str, Any]],
        backend_variant: Optional[str],
    ) -> str:
        parts = [
            "## Task: Deep Code Analysis for Optimisation Discovery",
            "",
            "Explore the hotspot code thoroughly and produce a ranked list of "
            "optimisation opportunities.  Follow the mandatory analysis workflow "
            "in order.  Use tools extensively — you have budget for 30+ tool calls.",
            "",
            "## Baseline Profile",
            "```json",
            json.dumps(profile, indent=2),
            "```",
            "",
            "## Hotspot Files to Analyse",
        ]
        for f in hotspot_files:
            parts.append(f"- `{f}`")

        parts.extend([
            "",
            "## Platform",
            "```json",
            json.dumps(system_caps, indent=2),
            "```",
        ])

        if backend_variant:
            parts.extend([
                "",
                f"## Backend: `{backend_variant}`",
                "The code uses OpenMP parallelisation.  Coordinates are accessed "
                "via `dbl3_t` struct (x[j].x, x[j].y, x[j].z).",
            ])

        if experience_hints:
            parts.extend([
                "",
                "## Historical Optimisation Experience",
                "```json",
                json.dumps(experience_hints[:10], indent=2),
                "```",
                "Use `search_experience` tool for more details on specific families.",
            ])

        parts.extend([
            "",
            "## Begin Analysis",
            "Start with Phase 1: call `get_profile` and `read_file` for the first "
            "hotspot file.  Work through all 6 phases systematically.",
        ])
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------

    def _parse_result(
        self, response: str, session: AgentSession
    ) -> DeepAnalysisResult:
        conversation_log = self._extract_conversation_log(session)

        # Handle max turns sentinel
        if response == MAX_TURNS_SENTINEL:
            partial = self._extract_partial_from_conversation(session)
            return DeepAnalysisResult(
                status="PARTIAL" if partial else "ERROR",
                analysis=partial,
                conversation_log=conversation_log,
                total_turns=session.turn_count,
                total_tokens=session.total_tokens,
            )

        # Try parsing the final response
        json_result = self._extract_json(response)
        if json_result:
            try:
                analysis = DeepCodeAnalysisResult(**json_result)
                return DeepAnalysisResult(
                    status="OK",
                    analysis=analysis,
                    conversation_log=conversation_log,
                    total_turns=session.turn_count,
                    total_tokens=session.total_tokens,
                )
            except Exception:
                pass

        # Fallback: try from unstructured response
        partial = self._extract_partial_from_conversation(session)
        return DeepAnalysisResult(
            status="PARTIAL" if partial else "ERROR",
            analysis=partial,
            conversation_log=conversation_log,
            total_turns=session.turn_count,
            total_tokens=session.total_tokens,
        )

    def _extract_partial_from_conversation(
        self, session: AgentSession
    ) -> Optional[DeepCodeAnalysisResult]:
        """Scan conversation history backward for any JSON with opportunities."""
        for msg in reversed(session.messages):
            if msg.role == "assistant" and msg.content:
                json_result = self._extract_json(msg.content)
                if json_result and "opportunities" in json_result:
                    try:
                        return DeepCodeAnalysisResult(**json_result)
                    except Exception:
                        continue
        return None

    # ------------------------------------------------------------------
    # JSON extraction (reused from ParameterExplorerAgent pattern)
    # ------------------------------------------------------------------

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
                    if "opportunities" in parsed:
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
