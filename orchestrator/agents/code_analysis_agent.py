"""Deep code analysis agent for Phase 2 initialization.

Runs ONCE at the start of Phase 2 to produce a ranked list of optimization
opportunities through interactive code exploration.  Unlike the single-shot
PatchPlanner, this agent uses tools to freely explore the codebase, study
reference implementations, check compiler reports, and examine assembly.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
import hashlib
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
from schemas.opportunity_graph import (
    MACRO_MECHANISMS,
    Composability,
    ExpectedGain,
    HotspotEvidence,
    OpportunityGraph,
    OpportunityGraphResult,
    OpportunityMechanism,
    OpportunityNode,
    OpportunityStatus,
    SelectedOpportunities,
    SelectedOpportunity,
    ValidationPlan,
    validate_graph,
)
from skills.agent_tools import CodeOptimizationTools


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _build_system_prompt(
    patch_families: Optional[Dict[str, Any]] = None,
    algorithm_preanalysis: Optional[Dict[str, Any]] = None,
    domain_knowledge: Optional[Dict[str, Any]] = None,
    bottleneck_classification: Optional[Dict[str, Any]] = None,
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
            hints = item.get("detection_hints")
            if hints and not covered:
                patterns = hints.get("patterns", [])
                if patterns:
                    lines.append(f"    Look for: {'; '.join(patterns[:2])}")
        if lines:
            families_section = (
                "\n\n## Known Optimization Patterns (reference only)\n"
                "These are previously identified patterns.  You may reference them "
                "by setting `family_hint`, but you are NOT limited to these.  "
                "Discovering novel optimizations is equally valuable.\n"
                + "\n".join(lines)
            )

    # Bottleneck classification section
    bottleneck_section = ""
    if bottleneck_classification and isinstance(bottleneck_classification, dict):
        bn_type = bottleneck_classification.get("bottleneck_type", "mixed")
        bn_ipc = bottleneck_classification.get("ipc", 0)
        effective = bottleneck_classification.get("effective_directions", [])
        ineffective = bottleneck_classification.get("ineffective_directions", [])
        rationale = bottleneck_classification.get("rationale", "")
        bn_parts: List[str] = [
            f"\n\n## Bottleneck Classification: {bn_type}",
            f"IPC: {bn_ipc:.2f}" if bn_ipc else "",
            f"Analysis: {rationale}" if rationale else "",
        ]
        if effective:
            bn_parts.append(f"Prioritize: {', '.join(effective)}")
        if ineffective:
            bn_parts.append(f"Deprioritize (likely ineffective): {', '.join(ineffective)}")
        bottleneck_section = "\n".join(p for p in bn_parts if p)

    # Algorithm-level pre-analysis insights
    algo_section = ""
    if algorithm_preanalysis and isinstance(algorithm_preanalysis, dict):
        algo_parts: List[str] = ["\n\n## Algorithm-Level Insights (from pre-analysis)"]
        opps = algorithm_preanalysis.get("algorithm_opportunities", [])
        if opps:
            algo_parts.append("Pre-identified algorithm-level opportunities:")
            for opp in opps[:6]:
                if isinstance(opp, dict):
                    title = opp.get("title", "")
                    rationale = opp.get("rationale", "")
                    impact = opp.get("estimated_impact", "")
                    algo_parts.append(f"  - {title} [{impact}]: {rationale}")
        ds_obs = algorithm_preanalysis.get("data_structure_observations", "")
        if ds_obs:
            algo_parts.append(f"Data structure observations: {ds_obs}")
        comm = algorithm_preanalysis.get("communication_pattern", "")
        if comm:
            algo_parts.append(f"Communication pattern: {comm}")
        if len(algo_parts) > 1:
            algo_section = "\n".join(algo_parts)

    # Domain knowledge section
    domain_section = ""
    if domain_knowledge and isinstance(domain_knowledge, dict):
        dk_parts: List[str] = ["\n\n## Domain Knowledge"]
        app_type = domain_knowledge.get("application_type", "")
        if app_type:
            dk_parts.append(f"Application type: {app_type}")
        kernels = domain_knowledge.get("kernel_semantics", [])
        if kernels:
            dk_parts.append("Key kernels:")
            for k in kernels[:5]:
                if isinstance(k, dict):
                    name = k.get("name", "")
                    desc = k.get("description", "")
                    dk_parts.append(f"  - {name}: {desc}")
                    for opt in (k.get("known_optimizations") or [])[:3]:
                        dk_parts.append(f"    * {opt}")
        strategies = domain_knowledge.get("effective_strategies", [])
        if strategies:
            dk_parts.append("Known effective strategies:")
            for s in strategies[:4]:
                if isinstance(s, dict):
                    cat = s.get("category", "")
                    desc = s.get("description", "")
                    gain = s.get("typical_gain", "")
                    dk_parts.append(f"  - [{cat}] {desc} (typical: {gain})")
        pitfalls = domain_knowledge.get("common_pitfalls", [])
        if pitfalls:
            dk_parts.append("Common pitfalls to avoid:")
            for p in pitfalls[:4]:
                dk_parts.append(f"  - {p}")
        if len(dk_parts) > 1:
            domain_section = "\n".join(dk_parts)

    return f"""\
You are an HPC deep-analysis agent.
Goal: discover high-impact source-level optimization opportunities on real hotspots.

Rules:
1. Prioritize macro mechanisms first: data_layout, memory_path, vectorization, algorithmic.
2. Only keep opportunities with concrete evidence and clear compiler gap.
3. Avoid generic micro-opts unless macro paths are weak or exhausted.
4. If context/profile is insufficient, return explicit status and missing items.
5. If status=OK, output 8-12 opportunities when feasible; prefer quality over fixed quota.
6. Favor structural transforms (data layout, memory traffic shaping, vector path redesign, algorithm path changes), not only loop micro-tweaks.
{families_section}{bottleneck_section}{algo_section}{domain_section}

Tool workflow (compact):
- get_profile -> read_file/get_file_outline/get_callers on top hotspots
- get_type_definition/get_type_layout on hot-path data
- get_reference_implementation + grep for comparable optimized patterns
- get_compiler_opt_report + get_structured_compiler_analysis to verify compiler gap
- get_assembly + get_compile_flags for detailed analysis
- search_experience to avoid repeating known failures

Output:
- Return ONE valid JSON object only (no markdown), with keys:
  status, hotspot_files, opportunities, recommended_sequence, strategy_rationale
- status must be one of: OK, NEED_MORE_CONTEXT, NEED_MORE_PROFILE, NO_ACTIONABLE
- Each opportunity must include:
  opportunity_id, title, category, target_files, target_functions,
  diagnosis, mechanism, compiler_gap, evidence, code_context,
  estimated_impact, confidence, risk_level.
- For each opportunity include one falsifiable performance hypothesis in `diagnosis`.
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
    error_reason: str = ""


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
    "get_structured_compiler_analysis",
    # Knowledge
    "get_reference_implementation",
    "search_experience",
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
        self.last_discovery_run: Optional[DeepAnalysisResult] = None

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
        supplemental_context: Optional[Dict[str, str]] = None,
    ) -> DeepAnalysisResult:
        """Legacy compatibility wrapper.

        The new contract is:
        1) discover_opportunity_graph(...)
        2) select_topk_from_graph(...)
        """
        return self._run_analysis_session(
            profile=profile,
            hotspot_files=hotspot_files,
            system_caps=system_caps,
            patch_families=patch_families,
            experience_hints=experience_hints,
            backend_variant=backend_variant,
            input_script_path=input_script_path,
            supplemental_context=supplemental_context,
        )

    def discover_opportunity_graph(
        self,
        profile: Dict[str, Any],
        hotspot_files: List[str],
        system_caps: Dict[str, Any],
        patch_families: Optional[Dict[str, Any]] = None,
        experience_hints: Optional[List[Dict[str, Any]]] = None,
        backend_variant: Optional[str] = None,
        input_script_path: Optional[str] = None,
        supplemental_context: Optional[Dict[str, str]] = None,
        algorithm_preanalysis: Optional[Dict[str, Any]] = None,
        domain_knowledge: Optional[Dict[str, Any]] = None,
        bottleneck_classification: Optional[Dict[str, Any]] = None,
    ) -> OpportunityGraphResult:
        run = self._run_analysis_session(
            profile=profile,
            hotspot_files=hotspot_files,
            system_caps=system_caps,
            patch_families=patch_families,
            experience_hints=experience_hints,
            backend_variant=backend_variant,
            input_script_path=input_script_path,
            supplemental_context=supplemental_context,
            algorithm_preanalysis=algorithm_preanalysis,
            domain_knowledge=domain_knowledge,
            bottleneck_classification=bottleneck_classification,
        )
        self.last_discovery_run = run
        analysis = run.analysis
        if analysis and analysis.opportunities:
            graph = self._analysis_to_opportunity_graph(
                analysis=analysis,
                profile=profile,
                fallback_hotspot_files=hotspot_files,
            )
            graph = validate_graph(graph)
            valid_nodes = [n for n in graph.opportunities if not n.invalid]
            if valid_nodes:
                return OpportunityGraphResult(
                    status=OpportunityStatus.OK,
                    graph=graph,
                    rationale="discovery complete",
                    suggestions=[],
                )

        missing = self._infer_missing_context(run.conversation_log, run.error_reason)
        needs_profile = self._infer_missing_profile(run.conversation_log, run.error_reason, profile)
        if missing:
            return OpportunityGraphResult(
                status=OpportunityStatus.NEED_MORE_CONTEXT,
                graph=None,
                missing=missing,
                needs_profile=[],
                rationale=run.error_reason or "missing code context",
                suggestions=[
                    "collect missing files/functions and rerun discovery once",
                ],
            )
        if needs_profile:
            return OpportunityGraphResult(
                status=OpportunityStatus.NEED_MORE_PROFILE,
                graph=None,
                missing=[],
                needs_profile=needs_profile,
                rationale=run.error_reason or "insufficient profiling evidence",
                suggestions=[
                    "collect requested profile data and rerun discovery",
                ],
            )
        return OpportunityGraphResult(
            status=OpportunityStatus.NO_ACTIONABLE,
            graph=None,
            missing=[],
            needs_profile=[],
            rationale=run.error_reason or "no actionable opportunities",
            suggestions=[
                "re-run deep analysis with broader hotspot scope",
                "collect compiler optimization report and branch/memory counters",
            ],
        )

    def select_topk_from_graph(
        self,
        graph: OpportunityGraph,
        k: int,
        budget: Optional[Dict[str, Any]] = None,
        experience_hints: Optional[List[Dict[str, Any]]] = None,
        selection_policy: Optional[Dict[str, Any]] = None,
    ) -> SelectedOpportunities:
        validated = validate_graph(graph)
        hints = experience_hints or []
        policy = selection_policy or {}
        tested_ids: set[str] = set()
        blocked_ids: set[str] = set()
        for hint in hints:
            if not isinstance(hint, dict):
                continue
            candidate_id = str(hint.get("opportunity_id") or hint.get("action_id") or "").strip()
            if not candidate_id:
                continue
            outcome = str(hint.get("outcome") or hint.get("status") or "").upper()
            if outcome in {"PASS", "FAIL", "SKIP"}:
                tested_ids.add(candidate_id)
            if outcome in {"FAIL", "INFEASIBLE"}:
                blocked_ids.add(candidate_id)

        budget_penalty = 1.0
        if isinstance(budget, dict):
            # Prefer lower implementation cost when run/iteration budget is tight.
            max_iters = float(budget.get("max_iters") or 0.0)
            max_runs = float(budget.get("max_runs") or 0.0)
            if max_iters > 0 and max_iters <= 3:
                budget_penalty *= 1.1
            if max_runs > 0 and max_runs <= 20:
                budget_penalty *= 1.2

        # Default policy favors deeper opportunities by reducing pure cost penalty
        # and applying soft mechanism priorities (not hard quotas).
        cost_penalty_power = max(0.3, min(1.0, float(policy.get("cost_penalty_power", 0.65) or 0.65)))
        cost_penalty_weight = max(0.1, min(1.0, float(policy.get("cost_penalty_weight", 0.7) or 0.7)))
        macro_priority_bonus = max(0.0, min(1.5, float(policy.get("macro_priority_bonus", 0.28) or 0.28)))
        algorithmic_priority_bonus = max(
            0.0, min(1.5, float(policy.get("algorithmic_priority_bonus", 0.14) or 0.14))
        )
        micro_penalty = max(0.0, min(1.0, float(policy.get("micro_penalty", 0.12) or 0.12)))
        untested_bonus = max(0.0, min(0.5, float(policy.get("untested_bonus", 0.06) or 0.06)))
        retested_penalty = max(0.0, min(0.5, float(policy.get("retested_penalty", 0.05) or 0.05)))
        macro_guard_top_n = max(0, int(policy.get("macro_guard_top_n", 2) or 2))
        macro_guard_min = max(0, int(policy.get("macro_guard_min", 2) or 2))
        macro_guard_max_gap = max(
            0.0, min(1.0, float(policy.get("macro_guard_max_relative_gap", 0.5) or 0.5))
        )

        scored: List[SelectedOpportunity] = []
        dropped: List[Dict[str, object]] = []
        for node in validated.opportunities:
            if node.invalid:
                dropped.append(
                    {
                        "opportunity_id": node.opportunity_id,
                        "reason": "invalid",
                        "details": list(node.invalid_reasons),
                    }
                )
                continue
            if node.opportunity_id in blocked_ids:
                dropped.append(
                    {
                        "opportunity_id": node.opportunity_id,
                        "reason": "historically_infeasible",
                    }
                )
                continue
            expected_speedup = self._normalize_speedup(node.expected_gain.p50)
            composability_score = float(node.composability.score)
            raw_cost = max(float(node.implementation_cost), 1.0)
            # Use sublinear cost growth so medium/complex opportunities are not
            # over-penalized versus micro tweaks.
            effective_cost = 1.0 + cost_penalty_weight * ((raw_cost ** cost_penalty_power) - 1.0)
            implementation_cost = effective_cost * budget_penalty
            objective = (
                expected_speedup
                * float(node.success_prob)
                * composability_score
                / max(implementation_cost, 1.0e-6)
            )
            mechanism_bonus = 0.0
            if node.mechanism in MACRO_MECHANISMS:
                mechanism_bonus += macro_priority_bonus
            if node.mechanism == OpportunityMechanism.ALGORITHMIC:
                mechanism_bonus += algorithmic_priority_bonus
            if node.mechanism == OpportunityMechanism.MICRO_OPT:
                mechanism_bonus -= micro_penalty
            if node.opportunity_id not in tested_ids:
                mechanism_bonus += untested_bonus
            else:
                mechanism_bonus -= retested_penalty
            weighted_objective = objective * max(0.05, 1.0 + mechanism_bonus)
            value_density = expected_speedup / max(effective_cost, 1.0e-6)
            node.score = weighted_objective
            node.value_density = value_density
            scored.append(
                SelectedOpportunity(
                    opportunity=node,
                    expected_speedup=expected_speedup,
                    success_prob=float(node.success_prob),
                    composability=composability_score,
                    implementation_cost=effective_cost,
                    objective_score=weighted_objective,
                    value_density=value_density,
                )
            )

        scored.sort(
            key=lambda item: (
                -item.objective_score,
                -item.value_density,
                item.opportunity.opportunity_id,
            )
        )
        target_k = max(0, int(k or 0))
        selected: List[SelectedOpportunity] = list(scored[:target_k])
        macro_rule_applied = False
        if target_k > 0 and macro_guard_top_n > 0 and selected:
            top_n = min(target_k, macro_guard_top_n, len(selected))
            macro_pool = [item for item in scored if item.opportunity.mechanism in MACRO_MECHANISMS]
            selected_ids = {item.opportunity.opportunity_id for item in selected}
            selected_top_macro = [
                item for item in selected[:top_n]
                if item.opportunity.mechanism in MACRO_MECHANISMS
            ]
            target_macro = min(macro_guard_min, top_n, len(macro_pool))
            needed = max(0, target_macro - len(selected_top_macro))
            if needed > 0:
                insert_pool = [
                    item for item in macro_pool
                    if item.opportunity.opportunity_id not in selected_ids
                ]
                for candidate in insert_pool:
                    if needed <= 0:
                        break
                    replace_idx = -1
                    for idx in range(top_n - 1, -1, -1):
                        if selected[idx].opportunity.mechanism not in MACRO_MECHANISMS:
                            replace_idx = idx
                            break
                    if replace_idx < 0:
                        break
                    replaced = selected[replace_idx]
                    denom = max(abs(replaced.objective_score), 1.0e-9)
                    rel_gap = max(0.0, (replaced.objective_score - candidate.objective_score) / denom)
                    if rel_gap <= macro_guard_max_gap:
                        selected_ids.discard(replaced.opportunity.opportunity_id)
                        selected_ids.add(candidate.opportunity.opportunity_id)
                        selected[replace_idx] = candidate
                        needed -= 1
                        macro_rule_applied = True

        return SelectedOpportunities(
            selected=selected,
            dropped=dropped,
            macro_rule_applied=macro_rule_applied,
            ranking_rationale=(
                "score = expected_speedup * success_prob * composability / effective_cost, "
                "with soft mechanism priorities and optional macro guard"
            ),
        )

    def _run_analysis_session(
        self,
        profile: Dict[str, Any],
        hotspot_files: List[str],
        system_caps: Dict[str, Any],
        patch_families: Optional[Dict[str, Any]] = None,
        experience_hints: Optional[List[Dict[str, Any]]] = None,
        backend_variant: Optional[str] = None,
        input_script_path: Optional[str] = None,
        supplemental_context: Optional[Dict[str, str]] = None,
        algorithm_preanalysis: Optional[Dict[str, Any]] = None,
        domain_knowledge: Optional[Dict[str, Any]] = None,
        bottleneck_classification: Optional[Dict[str, Any]] = None,
    ) -> DeepAnalysisResult:
        self.tools.set_profile_data(profile)
        if input_script_path:
            self.tools.set_benchmark_input(input_script_path)

        system_prompt = _build_system_prompt(
            patch_families,
            algorithm_preanalysis=algorithm_preanalysis,
            domain_knowledge=domain_knowledge,
            bottleneck_classification=bottleneck_classification,
        )
        session = self.llm_client.create_session(system_prompt)

        analysis_tools = self._get_analysis_tools()
        self.llm_client.register_tools(session, analysis_tools)

        user_message = self._build_initial_message(
            profile=profile,
            hotspot_files=hotspot_files,
            system_caps=system_caps,
            experience_hints=experience_hints or [],
            backend_variant=backend_variant,
            supplemental_context=supplemental_context or {},
        )

        try:
            response = self.llm_client.chat(
                session, user_message, auto_execute_tools=True
            )
            return self._parse_result(response, session, hotspot_files)
        except Exception as exc:
            if self.config.strict_availability:
                raise
            return DeepAnalysisResult(
                status="ERROR",
                analysis=None,
                conversation_log=self._extract_conversation_log(session),
                total_turns=session.turn_count,
                total_tokens=session.total_tokens,
                error_reason=str(exc),
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
        supplemental_context: Dict[str, str],
    ) -> str:
        profile_payload = self._compact_profile_payload(profile)
        caps_payload = self._compact_system_caps(system_caps)
        hotspot_preview = hotspot_files[:12]
        parts = [
            "## Task: Deep Code Analysis for Optimisation Discovery",
            "",
            "Explore hotspot code and produce a ranked list of actionable source-level opportunities.",
            "Work efficiently: keep tool calls focused on hottest functions and strongest evidence.",
            "When actionable, provide at least 6 opportunities (target 8-12), with macro-priority coverage.",
            "",
            "## Baseline Profile",
            "```json",
            json.dumps(profile_payload, indent=2),
            "```",
            "",
            "## Hotspot Files to Analyse",
        ]
        for f in hotspot_preview:
            parts.append(f"- `{f}`")
        if len(hotspot_files) > len(hotspot_preview):
            parts.append(f"- ... ({len(hotspot_files) - len(hotspot_preview)} more)")

        # TAU hotspot data (when available)
        tau_hotspots = profile.get("tau_hotspots", [])
        if tau_hotspots:
            parts.extend([
                "",
                "## TAU Function-Level Hotspots",
                "The following functions were identified by TAU sampling profiling.",
                "Focus on functions with the highest `exclusive_us`.",
                "```json",
                json.dumps(tau_hotspots[:12], indent=2),
                "```",
            ])
        portrait = profile_payload.get("bottleneck_portrait")
        if isinstance(portrait, dict) and portrait:
            parts.extend([
                "",
                "## Stage Bottleneck Portrait (per-round)",
                "```json",
                json.dumps(portrait, indent=2),
                "```",
                "Derive structural goals for the next patch round from this portrait, not from fixed templates.",
            ])

        parts.extend([
            "",
            "## Platform",
            "```json",
            json.dumps(caps_payload, indent=2),
            "```",
        ])

        if backend_variant:
            parts.extend([
                "",
                f"## Backend: `{backend_variant}`",
                "The application uses this backend for parallelisation.  "
                "Examine the source code to understand data structures and access patterns.",
            ])

        if experience_hints:
            parts.extend([
                "",
                "## Historical Optimisation Experience",
                "```json",
                json.dumps(experience_hints[:6], indent=2),
                "```",
                "Use `search_experience` tool for more details on specific families.",
            ])

        if supplemental_context:
            parts.extend(["", "## Additional Context (补上下文重试数据)"])
            for path, snippet in list(supplemental_context.items())[:8]:
                snippet_text = str(snippet or "").strip()
                if not snippet_text:
                    continue
                parts.extend(
                    [
                        f"### {path}",
                        "```text",
                        snippet_text[:4000],
                        "```",
                    ]
                )

        parts.extend([
            "",
            "## Begin Analysis",
            "Start with `get_profile`, then inspect top hotspot function(s) first.",
            "",
            "## Final Output Constraints",
            "Final answer must be ONE compact JSON object only (no markdown fences).",
            "Keep `reference_code` and evidence snippets concise; avoid dumping long code blocks.",
        ])
        return "\n".join(parts)

    def _compact_profile_payload(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if not isinstance(profile, dict):
            return payload
        timing = profile.get("timing_breakdown")
        if isinstance(timing, dict):
            top_timing = sorted(
                ((str(k), float(v)) for k, v in timing.items() if isinstance(v, (int, float))),
                key=lambda item: item[1],
                reverse=True,
            )[:12]
            payload["timing_breakdown_top"] = [
                {"metric": key, "value": value} for key, value in top_timing
            ]
        metrics = profile.get("system_metrics")
        if isinstance(metrics, dict):
            payload["system_metrics"] = {
                str(k): v for k, v in metrics.items() if isinstance(v, (int, float, str))
            }
        notes = profile.get("notes")
        if isinstance(notes, list):
            payload["notes"] = [str(item)[:240] for item in notes[:8]]
        tau = profile.get("tau_hotspots")
        if isinstance(tau, list):
            payload["tau_hotspots"] = tau[:12]
        return payload

    def _compact_system_caps(self, system_caps: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(system_caps, dict):
            return {}
        keep_keys = {
            "platform",
            "os",
            "cpu_model",
            "cpu_model_name",
            "cpu_cores_logical",
            "cpu_cores_physical",
            "memory_gb",
            "compiler",
            "mpi_launcher",
        }
        compact: Dict[str, Any] = {}
        for key in keep_keys:
            if key in system_caps:
                compact[key] = system_caps[key]
        return compact

    # ------------------------------------------------------------------
    # OpportunityGraph conversion + selection helpers
    # ------------------------------------------------------------------

    def _analysis_to_opportunity_graph(
        self,
        analysis: DeepCodeAnalysisResult,
        profile: Dict[str, Any],
        fallback_hotspot_files: List[str],
    ) -> OpportunityGraph:
        graph_id = f"opp-graph-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        evidence_catalog: Dict[str, Dict[str, object]] = {}
        nodes: List[OpportunityNode] = []
        tau_hotspots = profile.get("tau_hotspots", []) if isinstance(profile, dict) else []

        for idx, opp in enumerate(analysis.opportunities, start=1):
            hotspot = self._resolve_hotspot_evidence(
                opportunity=opp,
                tau_hotspots=tau_hotspots,
                fallback_hotspot_files=analysis.hotspot_files or fallback_hotspot_files,
            )
            evidence_ids: List[str] = []
            raw_evidence = list(opp.evidence or [])
            if opp.assembly_evidence:
                raw_evidence.append(opp.assembly_evidence)
            if opp.compiler_report_evidence:
                raw_evidence.append(opp.compiler_report_evidence)
            if opp.code_context:
                raw_evidence.append(opp.code_context[:240])
            for eidx, item in enumerate(raw_evidence, start=1):
                snippet = str(item or "").strip()
                if not snippet:
                    continue
                eid = self._evidence_id(opp.opportunity_id, eidx, snippet)
                evidence_catalog[eid] = {
                    "type": "analysis_evidence",
                    "opportunity_id": opp.opportunity_id,
                    "snippet": snippet[:800],
                }
                evidence_ids.append(eid)

            gain = self._impact_to_gain(
                opp.estimated_impact, opp.confidence, hotspot.share
            )
            mechanism = self._to_mechanism(opp.category, opp.mechanism, opp.family_hint)
            success_prob = max(0.05, min(0.95, float(opp.confidence)))
            implementation_cost = float(self._complexity_to_cost(opp.implementation_complexity))
            composability_score = self._composability_score(opp)
            node = OpportunityNode(
                opportunity_id=opp.opportunity_id or f"opportunity_{idx}",
                title=opp.title or f"Opportunity {idx}",
                hotspot=hotspot,
                mechanism=mechanism,
                evidence_ids=evidence_ids,
                hypothesis=self._build_hypothesis(opp),
                expected_gain=ExpectedGain(p50=gain["p50"], p90=gain["p90"]),
                success_prob=success_prob,
                implementation_cost=implementation_cost,
                composability=Composability(
                    score=composability_score,
                    depends_on=list(opp.depends_on or []),
                    conflicts_with=list(opp.conflicts_with or []),
                ),
                validation_plan=ValidationPlan(
                    benchmark="re-run baseline command + patched command on same case",
                    metrics=["runtime_seconds", "speedup_vs_baseline", "correctness_gate"],
                    acceptance=(
                        "PASS correctness + runtime improves over best_chain by >= 1%"
                    ),
                ),
                target_files=list(opp.target_files or []),
                target_functions=list(opp.target_functions or []),
                family_hint=str(opp.family_hint or ""),
                notes=str(opp.mechanism or ""),
                meta={
                    "diagnosis": opp.diagnosis,
                    "compiler_gap": opp.compiler_gap,
                    "category": opp.category,
                },
            )
            nodes.append(node)
        return OpportunityGraph(
            graph_id=graph_id,
            opportunities=nodes,
            evidence_catalog=evidence_catalog,
            invalid_nodes=[],
            ranking_notes=[],
        )

    def _resolve_hotspot_evidence(
        self,
        opportunity: Any,
        tau_hotspots: List[Dict[str, Any]],
        fallback_hotspot_files: List[str],
    ) -> HotspotEvidence:
        target_file = ""
        if getattr(opportunity, "target_files", None):
            target_file = str(opportunity.target_files[0] or "")
        elif fallback_hotspot_files:
            target_file = str(fallback_hotspot_files[0] or "")
        target_fn = ""
        if getattr(opportunity, "target_functions", None):
            target_fn = str(opportunity.target_functions[0] or "")

        share = 0.01
        line = "0-0"
        if tau_hotspots:
            matched = None
            for item in tau_hotspots:
                if not isinstance(item, dict):
                    continue
                fn = str(item.get("name") or "")
                file_path = str(item.get("file") or "")
                if target_fn and fn == target_fn:
                    matched = item
                    break
                if target_file and file_path and file_path.endswith(target_file):
                    matched = item
                    break
            if matched is None:
                matched = tau_hotspots[0] if isinstance(tau_hotspots[0], dict) else None
            if matched:
                target_file = str(matched.get("file") or target_file)
                if not target_fn:
                    target_fn = str(matched.get("name") or "")
                line_no = int(matched.get("line", 0) or 0)
                line = f"{line_no}-{line_no}" if line_no > 0 else "0-0"
                share = float(matched.get("exclusive_share", 0.0) or 0.0)
                if share <= 0:
                    excl = float(matched.get("exclusive_us", 0.0) or 0.0)
                    total_excl = sum(
                        float(item.get("exclusive_us", 0.0) or 0.0)
                        for item in tau_hotspots
                        if isinstance(item, dict)
                    )
                    if total_excl > 0 and excl > 0:
                        share = excl / total_excl
        if not target_file:
            target_file = "unknown"
        if not target_fn:
            target_fn = "unknown"
        return HotspotEvidence(
            file=target_file,
            function=target_fn,
            line_range=line,
            share=max(share, 0.01),
        )

    def _to_mechanism(
        self,
        category: str,
        mechanism: str,
        family_hint: Optional[str],
    ) -> OpportunityMechanism:
        text = " ".join(
            [
                str(category or "").lower(),
                str(mechanism or "").lower(),
                str(family_hint or "").lower(),
            ]
        )
        if any(key in text for key in ("layout", "pack", "soa", "aos")):
            return OpportunityMechanism.DATA_LAYOUT
        if any(key in text for key in ("memory", "cache", "prefetch", "indirect")):
            return OpportunityMechanism.MEMORY_PATH
        if any(key in text for key in ("vector", "simd", "restrict")):
            return OpportunityMechanism.VECTORIZATION
        if any(key in text for key in ("algorithm", "lookup", "branch split", "search")):
            return OpportunityMechanism.ALGORITHMIC
        if any(key in text for key in ("lock", "sync", "barrier")):
            return OpportunityMechanism.SYNC
        if any(key in text for key in ("io", "output", "log")):
            return OpportunityMechanism.IO
        if any(key in text for key in ("alloc", "arena", "malloc")):
            return OpportunityMechanism.ALLOCATION
        return OpportunityMechanism.MICRO_OPT

    def _impact_to_gain(
        self, impact: str, confidence: float, hotspot_share: float = 1.0
    ) -> Dict[str, float]:
        table = {
            "high": (0.18, 0.35),
            "medium": (0.08, 0.18),
            "low": (0.03, 0.08),
        }
        p50, p90 = table.get(str(impact or "").lower(), (0.06, 0.14))
        scale = max(0.6, min(1.2, 0.6 + float(confidence)))
        p50 = p50 * scale
        p90 = p90 * scale
        # Amdahl's Law: gain is bounded by the hotspot's share of total
        # runtime.  A "high" impact on a 1% hotspot yields ~1% max gain,
        # not the 18% that the raw table suggests.
        if 0 < hotspot_share < 1.0:
            p50 = min(p50, hotspot_share)
            p90 = min(p90, hotspot_share)
        return {
            "p50": round(p50, 4),
            "p90": round(p90, 4),
        }

    def _complexity_to_cost(self, complexity: str) -> float:
        mapping = {
            "trivial": 1.0,
            "simple": 2.0,
            "medium": 3.0,
            "complex": 4.0,
            "very_complex": 5.0,
            "high": 4.0,
            "low": 2.0,
        }
        return mapping.get(str(complexity or "").lower(), 3.0)

    def _composability_score(self, opportunity: Any) -> float:
        deps = len(list(getattr(opportunity, "depends_on", []) or []))
        conflicts = len(list(getattr(opportunity, "conflicts_with", []) or []))
        base = 0.75 - 0.08 * deps - 0.10 * conflicts
        return max(0.15, min(0.95, base))

    def _build_hypothesis(self, opportunity: Any) -> str:
        mechanism = str(getattr(opportunity, "mechanism", "") or "").strip()
        diagnosis = str(getattr(opportunity, "diagnosis", "") or "").strip()
        if mechanism and diagnosis:
            return f"{mechanism}; expected bottleneck reduction: {diagnosis[:180]}"
        if mechanism:
            return mechanism
        return "Reduce bottleneck in hotspot function with source-level restructuring."

    def _normalize_speedup(self, gain: float) -> float:
        if gain > 1.0:
            return max(gain / 100.0, 0.0)
        return max(float(gain), 0.0)

    def _evidence_id(self, opportunity_id: str, index: int, snippet: str) -> str:
        key = f"{opportunity_id}|{index}|{snippet[:200]}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        return f"ev_{opportunity_id}_{index}_{digest}"

    def _infer_missing_context(
        self,
        conversation_log: List[Dict[str, Any]],
        error_reason: str,
    ) -> List[str]:
        text = " ".join(
            [str(error_reason or "")]
            + [str(item.get("content") or "") for item in conversation_log]
        )
        patterns = [
            r"File not found:\s*([^\s\n]+)",
            r"missing include[:\s]+([^\s\n]+)",
            r"undefined reference to ([^\s\n]+)",
        ]
        missing: List[str] = []
        for pattern in patterns:
            for match in re.findall(pattern, text, flags=re.IGNORECASE):
                token = str(match).strip().strip("`\"',.;:()[]{}")
                # compile_commands is useful but non-blocking for analysis tools
                # because fallback compile/report paths are available.
                if "compile_commands.json" in token.lower():
                    continue
                if token and token not in missing:
                    missing.append(token)
        return missing

    def _infer_missing_profile(
        self,
        conversation_log: List[Dict[str, Any]],
        error_reason: str,
        profile: Dict[str, Any],
    ) -> List[str]:
        text = " ".join(
            [str(error_reason or "")]
            + [str(item.get("content") or "") for item in conversation_log]
        ).lower()
        needs: List[str] = []
        if not profile.get("tau_hotspots"):
            needs.append("tau_hotspots_or_xctrace_hotspots")
        if "compiler report" in text or "vectorization report" in text:
            needs.append("compiler_optimization_report")
        if "branch" in text and "miss" in text:
            needs.append("branch_miss_profile")
        if "l1" in text or "cache miss" in text:
            needs.append("cache_miss_profile")
        deduped: List[str] = []
        for item in needs:
            if item not in deduped:
                deduped.append(item)
        return deduped

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------

    def _parse_result(
        self,
        response: str,
        session: AgentSession,
        hotspot_files: Optional[List[str]] = None,
    ) -> DeepAnalysisResult:
        conversation_log = self._extract_conversation_log(session)
        fallback_hotspots = hotspot_files or []

        # Handle max turns sentinel
        if response == MAX_TURNS_SENTINEL:
            partial = self._extract_partial_from_conversation(session)
            if not partial:
                partial = self._repair_result_from_conversation(
                    conversation_log=conversation_log,
                    final_response="",
                    hotspot_files=fallback_hotspots,
                )
            return DeepAnalysisResult(
                status="PARTIAL" if partial else "ERROR",
                analysis=partial,
                conversation_log=conversation_log,
                total_turns=session.turn_count,
                total_tokens=session.total_tokens,
                error_reason=(
                    "max_turns_exhausted"
                    if partial is None
                    else "max_turns_exhausted_salvaged"
                ),
            )

        # Try parsing the final response
        json_result = self._extract_json(response)
        if json_result:
            analysis = self._coerce_analysis_result(json_result, fallback_hotspots)
            if analysis:
                return DeepAnalysisResult(
                    status="OK" if analysis.opportunities else "PARTIAL",
                    analysis=analysis,
                    conversation_log=conversation_log,
                    total_turns=session.turn_count,
                    total_tokens=session.total_tokens,
                    error_reason="parsed_final_response",
                )

        retry_payload = self._request_compact_final_json(session, fallback_hotspots)
        if retry_payload:
            retry_analysis = self._coerce_analysis_result(retry_payload, fallback_hotspots)
            if retry_analysis:
                return DeepAnalysisResult(
                    status="OK" if retry_analysis.opportunities else "PARTIAL",
                    analysis=retry_analysis,
                    conversation_log=self._extract_conversation_log(session),
                    total_turns=session.turn_count,
                    total_tokens=session.total_tokens,
                    error_reason="parsed_retry_response",
                )

        repaired = self._repair_result_from_conversation(
            conversation_log=conversation_log,
            final_response=response or "",
            hotspot_files=fallback_hotspots,
        )
        if repaired:
            return DeepAnalysisResult(
                status="PARTIAL",
                analysis=repaired,
                conversation_log=conversation_log,
                total_turns=session.turn_count,
                total_tokens=session.total_tokens,
                error_reason="repaired_from_conversation",
            )

        # Fallback: try from unstructured response
        partial = self._extract_partial_from_conversation(session)
        partial_reason = "recovered_from_partial_assistant_message" if partial else ""
        return DeepAnalysisResult(
            status="PARTIAL" if partial else "ERROR",
            analysis=partial,
            conversation_log=conversation_log,
            total_turns=session.turn_count,
            total_tokens=session.total_tokens,
            error_reason=(
                "no_parseable_result"
                if partial is None
                else partial_reason
            ),
        )

    def _extract_partial_from_conversation(
        self, session: AgentSession
    ) -> Optional[DeepCodeAnalysisResult]:
        """Scan conversation history backward for any JSON with opportunities."""
        best: Optional[DeepCodeAnalysisResult] = None
        best_score = -1.0
        for msg in reversed(session.messages):
            if msg.role == "assistant" and msg.content:
                json_result = self._extract_json(msg.content)
                if json_result:
                    analysis = self._coerce_analysis_result(json_result, [])
                    if analysis:
                        score = self._analysis_payload_score(json_result)
                        if score > best_score:
                            best = analysis
                            best_score = score
        return best

    # ------------------------------------------------------------------
    # JSON extraction (reused from ParameterExplorerAgent pattern)
    # ------------------------------------------------------------------

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract an analysis-like JSON object from free-form text."""
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
                if self._is_analysis_like_payload(parsed):
                    payloads.append(parsed)
        if not payloads:
            return None
        payloads.sort(key=self._analysis_payload_score, reverse=True)
        return payloads[0]

    def _iter_json_objects(self, text: str):
        decoder = json.JSONDecoder()
        idx = 0
        length = len(text)
        while idx < length:
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

    def _is_analysis_like_payload(self, parsed: Dict[str, Any]) -> bool:
        keys = set(parsed.keys())
        return bool(
            {"opportunities", "hotspot_files", "architecture_summary", "status"} & keys
        )

    def _analysis_payload_score(self, payload: Dict[str, Any]) -> float:
        if not isinstance(payload, dict):
            return -1.0
        score = 0.0
        opps = payload.get("opportunities")
        if isinstance(opps, list):
            score += 10.0 * len(opps)
        elif isinstance(opps, dict):
            score += 8.0 * len(opps)
        hotspot_files = payload.get("hotspot_files")
        if isinstance(hotspot_files, list):
            score += float(min(len(hotspot_files), 10))
        if isinstance(payload.get("recommended_sequence"), list):
            score += 5.0
        for key in ("architecture_summary", "bottleneck_diagnosis", "strategy_rationale"):
            if str(payload.get(key, "")).strip():
                score += 2.0
        return score

    def _request_compact_final_json(
        self,
        session: AgentSession,
        fallback_hotspots: List[str],
    ) -> Optional[Dict[str, Any]]:
        if session.turn_count >= self.config.max_turns:
            return None
        try:
            retry = self.llm_client.chat(
                session,
                (
                    "Return exactly ONE compact JSON object now. "
                    "No markdown, no tools. Required keys: status, hotspot_files, opportunities. "
                    f"Use these hotspot files when relevant: {json.dumps(fallback_hotspots[:10])}"
                ),
                auto_execute_tools=False,
            )
        except Exception:
            return None
        parsed = self._extract_json(retry or "")
        if isinstance(parsed, dict):
            return parsed
        try:
            loaded = json.loads((retry or "").strip())
        except Exception:
            return None
        return loaded if isinstance(loaded, dict) else None

    def _coerce_analysis_result(
        self,
        payload: Dict[str, Any],
        fallback_hotspots: List[str],
    ) -> Optional[DeepCodeAnalysisResult]:
        if not isinstance(payload, dict):
            return None
        status_raw = str(payload.get("status", "NEED_MORE_CONTEXT")).strip().upper()
        allowed_status = {"OK", "NEED_MORE_EVIDENCE", "NEED_MORE_CONTEXT", "NEED_MORE_PROFILE", "NO_ACTIONABLE"}
        status = status_raw if status_raw in allowed_status else "NEED_MORE_CONTEXT"
        status_model = status if status in {"OK", "NEED_MORE_EVIDENCE", "NEED_MORE_CONTEXT"} else "NEED_MORE_CONTEXT"
        hotspot_files = self._to_str_list(payload.get("hotspot_files"))
        if not hotspot_files:
            hotspot_files = self._to_str_list(fallback_hotspots)

        opportunities = self._coerce_opportunities(
            payload.get("opportunities"),
            hotspot_files=hotspot_files,
        )

        normalized: Dict[str, Any] = {
            "status": status_model,
            "architecture_summary": str(payload.get("architecture_summary", "") or ""),
            "bottleneck_diagnosis": str(payload.get("bottleneck_diagnosis", "") or ""),
            "hotspot_files": hotspot_files,
            "opportunities": opportunities,
            "recommended_sequence": self._to_str_list(payload.get("recommended_sequence")),
            "strategy_rationale": str(payload.get("strategy_rationale", "") or ""),
            "exploration_notes": self._to_str_list(payload.get("exploration_notes")),
        }
        try:
            return DeepCodeAnalysisResult(**normalized)
        except Exception:
            return None

    def _coerce_opportunities(
        self,
        raw_opps: Any,
        hotspot_files: List[str],
    ) -> List[Dict[str, Any]]:
        if isinstance(raw_opps, dict):
            items = list(raw_opps.values())
        elif isinstance(raw_opps, list):
            items = raw_opps
        else:
            items = []
        coerced: List[Dict[str, Any]] = []
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            op_id = str(
                item.get("opportunity_id")
                or item.get("id")
                or item.get("name")
                or f"opportunity_{idx}"
            ).strip()
            title = str(item.get("title") or item.get("name") or op_id).strip()
            category = str(item.get("category") or item.get("type") or "novel").strip()
            target_files = self._to_str_list(item.get("target_files") or item.get("files"))
            if not target_files and hotspot_files:
                target_files = [hotspot_files[min(idx - 1, len(hotspot_files) - 1)]]
            target_functions = self._to_str_list(
                item.get("target_functions") or item.get("functions")
            )
            diagnosis = str(item.get("diagnosis") or item.get("problem") or "").strip()
            mechanism = str(item.get("mechanism") or item.get("approach") or "").strip()
            compiler_gap = str(
                item.get("compiler_gap")
                or item.get("gap")
                or "Requires semantic/source-level transformation beyond compiler heuristics."
            ).strip()
            evidence = self._to_str_list(item.get("evidence"))
            if not evidence:
                asm = str(item.get("assembly_evidence") or "").strip()
                rpt = str(item.get("compiler_report_evidence") or "").strip()
                if asm:
                    evidence.append(asm[:300])
                if rpt:
                    evidence.append(rpt[:300])
            risk = str(item.get("risk_level") or "medium").strip().lower()
            if risk not in {"low", "medium", "high"}:
                risk = "medium"
            impact = str(item.get("estimated_impact") or "medium").strip().lower()
            if impact not in {"low", "medium", "high"}:
                impact = "medium"
            try:
                confidence = float(item.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))
            opp: Dict[str, Any] = {
                "opportunity_id": op_id or f"opportunity_{idx}",
                "title": title or f"Opportunity {idx}",
                "category": category or "novel",
                "family_hint": item.get("family_hint"),
                "target_files": target_files,
                "target_functions": target_functions,
                "diagnosis": diagnosis,
                "mechanism": mechanism,
                "compiler_gap": compiler_gap,
                "evidence": evidence,
                "code_context": str(item.get("code_context") or "").strip(),
                "reference_code": str(item.get("reference_code") or "").strip(),
                "assembly_evidence": str(item.get("assembly_evidence") or "").strip(),
                "compiler_report_evidence": str(
                    item.get("compiler_report_evidence") or ""
                ).strip(),
                "estimated_impact": impact,
                "confidence": confidence,
                "risk_level": risk,
                "expected_effect": self._to_str_list(item.get("expected_effect")),
                "depends_on": self._to_str_list(item.get("depends_on")),
                "conflicts_with": self._to_str_list(item.get("conflicts_with")),
                "composable_with": self._to_str_list(item.get("composable_with")),
                "implementation_complexity": str(
                    item.get("implementation_complexity") or "medium"
                ).strip(),
                "lines_of_change_estimate": self._to_int(
                    item.get("lines_of_change_estimate"), default=0
                ),
                "priority_rank": self._to_int(item.get("priority_rank"), default=idx),
            }
            coerced.append(opp)
        return coerced

    def _to_str_list(self, value: Any) -> List[str]:
        if isinstance(value, list):
            out: List[str] = []
            for item in value:
                s = str(item).strip()
                if s and s not in out:
                    out.append(s)
            return out
        if value is None:
            return []
        s = str(value).strip()
        return [s] if s else []

    def _to_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _repair_result_from_conversation(
        self,
        conversation_log: List[Dict[str, Any]],
        final_response: str,
        hotspot_files: List[str],
    ) -> Optional[DeepCodeAnalysisResult]:
        context = self._build_repair_context(
            conversation_log=conversation_log,
            final_response=final_response,
            max_chars=24000,
        )
        if not context:
            return None
        repair_system = (
            "You are a strict JSON formatter for HPC code analysis results. "
            "Output exactly one JSON object matching DeepCodeAnalysisResult. "
            "No markdown, no prose."
        )
        repair_user = (
            "Convert the analysis transcript into a valid DeepCodeAnalysisResult JSON.\n"
            "Rules:\n"
            "1) Return JSON only.\n"
            "2) Include 'status', 'hotspot_files', and 'opportunities'.\n"
            "3) If evidence is weak, keep opportunities concise and lower confidence.\n"
            f"4) Preserve these hotspot files when possible: {json.dumps(hotspot_files[:10])}\n\n"
            "Transcript excerpt:\n"
            f"{context}\n"
        )
        repair_session = self.llm_client.create_session(repair_system)
        try:
            response = self.llm_client.chat(
                repair_session, repair_user, auto_execute_tools=False
            )
        except Exception:
            return None
        parsed = self._extract_json(response or "")
        if not parsed:
            try:
                parsed = json.loads((response or "").strip())
            except Exception:
                parsed = None
        if not isinstance(parsed, dict):
            try:
                retry = self.llm_client.chat(
                    repair_session,
                    "Return one valid JSON object only. No markdown. Include keys: "
                    "status, hotspot_files, opportunities.",
                    auto_execute_tools=False,
                )
            except Exception:
                retry = ""
            parsed = self._extract_json(retry or "")
            if not parsed:
                try:
                    parsed = json.loads((retry or "").strip())
                except Exception:
                    parsed = None
        if not isinstance(parsed, dict):
            return None
        return self._coerce_analysis_result(parsed, hotspot_files)

    def _build_repair_context(
        self,
        conversation_log: List[Dict[str, Any]],
        final_response: str,
        max_chars: int = 24000,
    ) -> str:
        chunks: List[str] = []
        total = 0
        if final_response:
            frag = final_response.strip()
            if frag:
                frag = frag[:1200]
                chunks.append(f"[assistant-final]\n{frag}")
                total += len(chunks[-1])
        for entry in reversed(conversation_log):
            role = str(entry.get("role", ""))
            if role not in {"assistant", "tool"}:
                continue
            content = str(entry.get("content") or "").strip()
            if not content:
                continue
            lower = content.lower()
            if (
                role == "tool"
                and not any(
                    k in lower
                    for k in (
                        "hotspot",
                        "compiler",
                        "assembly",
                        "vector",
                        "opportun",
                        "cache",
                        "bottleneck",
                        "missed",
                    )
                )
            ):
                continue
            snippet = content[:2000]
            block = f"[{role}]\n{snippet}"
            if total + len(block) + 2 > max_chars:
                break
            chunks.append(block)
            total += len(block) + 2
        chunks.reverse()
        return "\n\n".join(chunks)

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
