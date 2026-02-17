# OpportunityGraph Patch Pipeline (Strict Mode)

## 1) Phase 2 Init: Discovery Contract

`DeepCodeAnalysisAgent.discover_opportunity_graph(...)` produces exactly one explicit status:

- `OK`: returns a validated `OpportunityGraph`.
- `NEED_MORE_CONTEXT`: returns `missing` path/symbol list.
- `NEED_MORE_PROFILE`: returns required profile artifacts.
- `NO_ACTIONABLE`: returns rationale + suggestions.

No silent fallback is allowed in strict mode.

## 2) Status Handling in Orchestrator

`orchestrator/graph.py` routes status with explicit branches:

- `OK` -> continue to selection.
- `NEED_MORE_CONTEXT` -> collect missing file snippets and retry once.
- `NEED_MORE_PROFILE` -> explicit error (stop).
- `NO_ACTIONABLE` -> explicit error (stop).

All branches are written to `agent_trace.json` (`deep_analysis_discovery` events).

## 3) Selection Contract (Top-K)

`DeepCodeAnalysisAgent.select_topk_from_graph(...)` computes unified objective:

`score = expected_speedup * success_prob * composability / implementation_cost`

Each node must carry:

- hotspot evidence
- mechanism enum
- evidence ids
- falsifiable hypothesis
- expected gain (`p50`, `p90`)
- success probability
- implementation cost
- composability + depends/conflicts
- validation plan

Invalid nodes are retained with `invalid=true` and reasons for audit.

## 4) Macro-First Hard Rule

When untested macro mechanisms exist (`data_layout`, `memory_path`, `vectorization`, `algorithmic`):

- selection and ranking push macro to the front,
- `micro_opt` cannot remain in Top-2.

Rule is implemented in code, not prompt-only.

## 5) Patch Planning Contract

PATCH actions are generated from selected graph nodes only:

- `PatchPlannerAgent.plan_from_opportunity_selection(...)`
- action parameters include `opportunity_id`, `evidence_ids`, ROI fields, validation plan.

In graph-driven mode, legacy free-form PATCH proposal path is disabled.

## 6) Queue Semantics

Graph-derived actions are treated as a deterministic queue:

- an action is considered pending until actually tested,
- tested actions are skipped on subsequent iterations,
- when no untested graph actions remain, PATCH phase exits explicitly (`opportunity_graph_exhausted`).

## 7) Audit Artifacts

Session artifacts include:

- `agent_trace.json`: full graph + selection + invalid reasons + evidence ids.
- `opportunity_graph.json`: full graph dump.
- `opportunity_selection.json`: selected opportunities and score decomposition.
- `best_state.json`: compact `opportunity_graph` summary.

## 8) Config Knobs (planner.yaml)

- `two_phase.deep_analysis.enabled`
- `two_phase.deep_analysis.strict_required`
- `two_phase.deep_analysis.top_k`
- `two_phase.deep_analysis.max_context_retry`
- `ranking.value_density_min`
- `ranking.macro_first.enabled`
- `ranking.macro_first.protect_top_n`

These are designed to be app-agnostic (BWA/LAMMPS/other HPC apps).
