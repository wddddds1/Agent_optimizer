# HPC Agent Platform Architecture (7 Agents + Reporter)

## 0. Goals & Non-Goals

### Goals
- Provide an auditable, reproducible optimization loop for HPC applications (starting with LAMMPS) that supports:
  1) runtime/run_config tuning
  2) build_config compilation strategy tuning
  3) localized source_patch optimization
- Enforce "controlled action families" with policy constraints and deterministic orchestration.
- Preserve complete evidence for every attempt: inputs, patches, build provenance, runtime artifacts, metrics, and decisions.

### Non-Goals
- Orchestrator does not "understand" application source code semantics.
- Agents never directly execute tools; all side effects go through Skills.
- The system is not a generic autonomous codebase refactoring agent; only constrained, localized transforms are allowed.

---

## 1. Core Separation of Concerns

### Control Plane (Deterministic)
- **Orchestrator**: decides WHEN and WHAT to execute, manages budgets, schedules agents, enforces policy, persists artifacts, and applies stop/fuse rules.
- **Judge/Gates (deterministic)**: validates build/runtime/performance/correctness. LLM cannot overrule gates.

### Reasoning Plane (LLM-assisted but constrained)
- **Agents**: propose, classify, plan, and explain; output must conform to typed IR (Pydantic/JSON schema).
- Agent outputs are treated as *suggestions*; Orchestrator and Policy decide whether to execute.

### Execution Plane (Side effects)
- **Skills**: the only modules allowed to modify repo/worktrees, build binaries, run cases, and write artifacts.

---

## 2. High-Level Loop (One Iteration)

1. Baseline (if needed): run N times, compute median/variance; store BaselineStats.
2. ProfilerAgent: parse artifacts -> ProfileReport.
3. AnalystAgent: ProfileReport + History -> AnalysisResult (allowed families/transforms).
4. PlannerAgent: AnalysisResult + Budget + Cost model -> PlanIR (which family, how many candidates, evaluation strategy).
5. OptimizerAgent: PlanIR + Policy + (optional code context) -> CandidateList[ActionIR] (2–5 representative actions).
6. RouterRankerAgent: policy-filter + dedupe + score -> RankedActions (top-K).
7. For each action in RankedActions (until budget):
   a) ExecutorAgent: apply action via skills (patch/build/run/collect) -> RunResult.
   b) VerifierAgent: deterministic gates -> VerificationResult.
   c) TriageAgent (only on FAIL/exception): classify failure -> FailureSummary; may trigger debug mode/fuse rules.
   d) Ledger append: store outcome and evidence pointers.
8. ReporterAgent (out-of-loop, end of run or end of iteration): synthesize report.md/report.json.

---

## 3. Agents (8 Total)

### Shared Contract for ALL Agents
- Input: a structured `ContextBundle` assembled by Orchestrator (see §4).
- Output: a single IR object (Pydantic model) per call.
- No tool execution. No filesystem writes except returning text blobs that Orchestrator may persist.
- Must include `confidence` and `assumptions` when reasoning is uncertain.
- Must not invent paths, commands, or metrics not present in inputs.

---

### 3.1 ProfilerAgent (Deterministic)
**Purpose**: Extract structured performance evidence from run artifacts.

**When called**: after baseline and/or after each candidate run (optional; typically baseline + best run).

**Inputs**
- `CaseSpec`, `RunArtifactsIndex` (paths to logs/time/metrics files)
- `BaselineStats` (optional)
- `EnvProvenance` (optional)

**Output IR: `ProfileReport`**
Required fields:
- `run_id: str`
- `case_id: str`
- `bottleneck: Literal["compute","memory","io","comm","imbalance","unknown"]`
- `hotspots: list[Hotspot]` (even if coarse)
- `signals: ProfileSignals`
- `confidence: float` (0..1)
- `notes: list[str]`

Hotspot:
- `file: Optional[str]`
- `function: Optional[str]`
- `percent_time: Optional[float]`
- `evidence: list[str]` (e.g., "LAMMPS timing breakdown: Pair=62%")

ProfileSignals:
- `runtime_s: float`
- `step_time_s: Optional[float]`
- `cpu_util_pct: Optional[float]`
- `rss_mb: Optional[float]`
- `omp_threads: Optional[int]`
- `mpi_ranks: Optional[int]`
- `timing_breakdown: dict[str,float]` (best-effort)

**Boundary**
- Does not propose actions. Does not read source code beyond file/function names in profiler outputs.

**Prompt requirements**
- None by default. (If LLM used for human-readable summary, output must still be `ProfileReport`.)

---

### 3.2 AnalystAgent (LLM-assisted, constrained)
**Purpose**: Convert evidence into allowed optimization directions and risk constraints.

**When called**: every iteration after ProfileReport.

**Inputs**
- `ProfileReport`
- `HistorySummary` (recent gains/fails/cost by family)
- `PolicySnapshot` (available families, hard constraints)
- `CaseSuiteSpec` (primary/guard, tags)

**Output IR: `AnalysisResult`**
Required fields:
- `bottleneck: ...` (same enum as ProfileReport)
- `allowed_families: list[FamilyId]` (subset of ["run_config","build_config","source_patch"])
- `allowed_transforms: list[str]` (for source_patch; e.g., "hoist_invariant","math_simplify")
- `forbidden_transforms: list[str]`
- `risk_overrides: dict[str,Any]` (e.g., {"fastmath": "forbid", "lto": "allow"})
- `confidence: float`
- `rationale: str` (short, grounded in evidence)

**Boundary**
- Must not output specific actions/patches. Only “what directions are allowed now”.

**Prompt must include**
- Role & scope
- Evidence-only rule
- Output schema (JSON)
- Conservative behavior under low confidence

---

### 3.3 PlannerAgent (Deterministic decision; optional LLM explanation)
**Purpose**: Choose *one* optimization family (or two max) and allocate budget/evaluation strategy.

**When called**: after AnalysisResult.

**Inputs**
- `AnalysisResult`
- `BudgetState` (remaining builds/runs/time)
- `CostModel` (avg build time, avg run time, expected retries)
- `HistorySummary` (success rate, median gain per family)

**Output IR: `PlanIR`**
Required fields:
- `iteration_id: int`
- `chosen_families: list[FamilyId]` (1–2)
- `max_candidates: int` (typically 2–5)
- `evaluation: EvaluationPlan` (baseline repeats, halving stages, top1 reruns)
- `enable_debug_mode: bool`
- `fuse_rules: FusePlan` (e.g., disable source_patch after N compile fails)
- `stop_condition: StopPlan` (e.g., stop if improvement < eps for K rounds)
- `reason: str` (short, references history + cost)

EvaluationPlan:
- `baseline_repeats: int`
- `candidate_repeats_stage0: int` (short run)
- `candidate_repeats_stage1: int` (full run)
- `top1_validation_repeats: int`
- `use_successive_halving: bool`

FusePlan:
- `max_compile_fails: int`
- `max_runtime_fails: int`
- `cooldown_rounds: int`
- `fallback_family: FamilyId`

StopPlan:
- `max_iterations: int`
- `min_relative_gain: float`
- `patience_rounds: int`

**Boundary**
- Planner does not generate actions. It only chooses direction/budget.

**Prompt requirements**
- If LLM is used, it must only produce an explanation string; the PlanIR must remain deterministic and policy-driven.

---

### 3.4 OptimizerAgent (LLM-assisted, generates candidates)
**Purpose**: Generate *representative* candidates within the chosen family, not exhaustive search.

**When called**: after PlanIR; once per chosen family.

**Inputs**
- `PlanIR`
- `PolicySnapshot`
- `CaseSuiteSpec`
- `ProfileReport` (hotspot hints)
- `BuildProvenanceSummary` (toolchain constraints)
- For `source_patch`: `CodeContextPack` (whitelisted files/snippets only)

**Output IR: `CandidateList`**
Required fields:
- `family: FamilyId`
- `candidates: list[ActionIR]` (2–5; may be 1 for source_patch)
- `assumptions: list[str]`
- `confidence: float`

**ActionIR (common)**
- `action_id: str`
- `family: FamilyId`
- `risk: Literal["very_low","low","medium","high"]`
- `expected_cost: Literal["low","medium","high"]`
- `expected_gain: Literal["low","medium","high"]`
- `preconditions: list[str]` (e.g., "OPENMP enabled")
- `explanation: str` (grounded)

Family-specific payload:
1) run_config payload `RunConfigChange`
- env vars (OMP_NUM_THREADS, OMP_PROC_BIND, OMP_PLACES, etc.)
- cli args (e.g., -sf omp, -pk omp N)

2) build_config payload `BuildConfigChange`
- compiler choice (appleclang/llvmclang) if available
- cmake flags pack (O3/native/LTO/PGO)
- link/runtime libs selection (libomp path)

3) source_patch payload `SourcePatchChange`
- `patch_format: Literal["unified_diff","patch_file_ref"]`
- `patch_text: str` (unified diff) OR `patch_path: str`
- `patch_root: str` (e.g., "third_party/lammps")
- `touched_files: list[str]`
- `diff_stats: {files:int, lines_added:int, lines_removed:int}`

**Boundary**
- Must obey policy constraints: allowed files only, max diff lines, allowed transforms.
- Must not propose broad refactors, API changes, or multi-file rewrites unless explicitly allowed.

**Prompt must include**
- Strict JSON output requirement (no prose outside JSON)
- Candidate count limits
- Action-family constraints
- For source_patch: only modify code within provided snippet boundaries; if insufficient context, return NEED_MORE_CONTEXT.

---

### 3.5 RouterRankerAgent (Deterministic filter + ranking; LLM optional explain)
**Purpose**: Convert CandidateList into an executable ranked queue under policy + cost/risk.

**When called**: after CandidateList.

**Inputs**
- `CandidateList`
- `PolicySnapshot`
- `HistorySummary`
- `BudgetState`
- `CostModel`

**Output IR: `RankedActions`**
Required fields:
- `ranked: list[RankedAction]` (top-K)
- `rejected: list[Rejection]` (with reasons)
- `scoring_notes: str`

RankedAction:
- `action: ActionIR`
- `score: float`
- `score_breakdown: dict[str,float]` (gain_prior, cost_penalty, risk_penalty, novelty)

Rejection:
- `action_id: str`
- `reason: str`

**Boundary**
- No new candidates generated.
- Deterministic ranking strongly preferred for reproducibility.

**Prompt**
- Not required (deterministic). If LLM used, only for `scoring_notes` explanation.

---

### 3.6 ExecutorAgent (Deterministic; uses Skills)
**Purpose**: Apply one action in an isolated workspace, produce full artifacts, never pollute baseline.

**When called**: for each ranked action attempt.

**Inputs**
- `ActionIR`
- `EnvSpec` (paths, toolchain, build dirs)
- `CaseSuiteSpec`
- `ExecutionPlan` (from PlanIR)

**Output IR: `RunResult`**
Required fields:
- `run_id: str`
- `action_id: str`
- `status: Literal["OK","FAIL"]`
- `phase: Literal["patch","build","run_primary","run_guard"]` (where it failed)
- `artifacts: RunArtifactsIndex` (paths)
- `provenance: RunProvenance` (binary hash, cmake cache pointer)
- `timings: TimingSummary` (build_s, run_primary_s, run_guard_s)

**Boundary**
- Does not decide pass/fail; just executes and records.
- Must always produce artifacts and a repro script (if enabled).

---

### 3.7 VerifierAgent (Deterministic Judge)
**Purpose**: Decide PASS/FAIL based on gates: runtime, performance, variance, correctness.

**When called**: after ExecutorAgent returns.

**Inputs**
- `RunResult`
- `BaselineStats`
- `GatesConfig`
- `CaseSuiteSpec`

**Output IR: `VerificationResult`**
Required fields:
- `run_id: str`
- `action_id: str`
- `verdict: Literal["PASS","FAIL"]`
- `runtime_gate: GateOutcome`
- `performance_gate: GateOutcome`
- `correctness_gate: GateOutcome`
- `metrics: VerifiedMetrics` (median runtime, rel gain, variance)
- `reasons: list[str]`

GateOutcome:
- `passed: bool`
- `details: dict[str,Any]`

VerifiedMetrics:
- `baseline_median_s: float`
- `candidate_median_s: float`
- `relative_improvement: float`
- `variance_cv: Optional[float]`

**Boundary**
- No tool execution. No candidate generation.
- Must remain deterministic.

---

### 3.8 TriageAgent (LLM-assisted failure classifier)
**Purpose**: Summarize failures to save human debugging time and to drive fuse rules.

**When called**: when ExecutorAgent fails or VerifierAgent FAIL with errors.

**Inputs**
- `RunResult` + relevant log excerpts (stderr/build.log tail)
- `HistorySummary` (recent similar failures)
- `PolicySnapshot` (what was attempted)

**Output IR: `FailureSummary`**
Required fields:
- `run_id: str`
- `action_id: str`
- `category: Literal["ENV","BUILD","RUNTIME","CORRECTNESS","PERF_NOISE","UNKNOWN"]`
- `signature: str` (stable fingerprint)
- `top_causes: list[str]` (grounded)
- `next_steps: list[NextStep]`
- `suggest_debug_mode: bool`
- `suggest_disable_family: Optional[FamilyId]`
- `confidence: float`

NextStep:
- `type: Literal["collect_more_logs","try_smaller_patch","fix_toolchain","adjust_case","retry","stop_family"]`
- `detail: str`

**Boundary**
- Must not propose new code patches.
- May suggest enabling debug mode or disabling a family.

**Prompt must include**
- Extract-only behavior from logs
- JSON output only
- Conservative recommendations

---

### 3.9 ReporterAgent (LLM-assisted report writer; out-of-loop)
**Purpose**: Convert ledger + artifacts into a clear optimization report for humans and paper writing.

**When called**: end of run or on demand.

**Inputs**
- `Ledger` (all attempts)
- `BestResultSummary`
- `EnvProvenance`, `BuildProvenance`
- `CaseSuiteSpec`
- `PolicySnapshot`

**Output IR: `ReportBundle`**
Required fields:
- `report_md: str`
- `report_json: dict`
- `tables: dict[str, list[dict]]` (e.g., attempts table)
- `figures: list[FigureSpec]` (optional placeholders)
- `key_takeaways: list[str]`

**Boundary**
- No execution, no changes. Pure summarization.
- Must cite run_ids and artifact paths as evidence pointers.

**Prompt must include**
- Mandatory structure: Overview, Setup, Method, Attempts Table, Best Patch, Validation, Limitations, Repro steps.
- Must not invent results; only use ledger entries.

---

## 4. Context Bundle (Inputs to Agents)

Orchestrator constructs a `ContextBundle` with:
- `iteration_id`
- `case_suite: CaseSuiteSpec`
- `policy: PolicySnapshot`
- `budget: BudgetState`
- `history: HistorySummary`
- `baseline: BaselineStats` (if available)
- `profile: ProfileReport` (when available)
- `env_provenance: EnvProvenanceSummary`
- `build_provenance: BuildProvenanceSummary`
- `code_context: CodeContextPack` (source_patch only; whitelisted snippets)

**Important**: code_context must be minimal and scoped: only relevant files + bounded line ranges.

---

## 5. Skills (Tooling Modules)

Skills are the only side-effect modules. They must:
- Accept typed inputs
- Produce typed outputs
- Always write artifacts under `artifacts/runs/<run_id>/`
- Never mutate main branch; use git worktrees

### Minimum skill set
1) `skills/patch_apply.py`
- apply unified diff to repo/worktree (supports submodule root)
- enforce: allowed files, max diff lines, forbid touching build system unless allowed
- output: PatchApplyResult + saved patch file

2) `skills/build.py`
- configure/build in per-run build dir
- capture: build.log, CMakeCache, compile_commands, tool versions, binary hash, linked libs
- output: BuildResult

3) `skills/run_case.py`
- run primary/guard with configured repeats (short/full)
- capture: stdout/stderr/log/time and metrics parse inputs
- output: RunCaseResult

4) `skills/metrics_parse.py`
- parse app logs into canonical metrics (runtime, step time, energy series, etc.)
- output: MetricsReport

5) `skills/gates.py`
- runtime gate, performance gate (median/CV), correctness gate (series/drift/hash)
- output: GateResults

6) `skills/repro.py`
- generate `repro.sh` for each run_id with exact env, binary, args, working dirs

7) `skills/ledger.py`
- append/read ledger entries
- produce HistorySummary aggregates

---

## 6. Policy & Action Families (Controlled Action Space)

Families:
- `run_config`: runtime environment + CLI args
- `build_config`: toolchain/flags strategy packs
- `source_patch`: localized code modifications (diff) with strict constraints

Policy constraints must be explicit:
- allowed families by case tag
- allowed files for source_patch
- max diff lines
- allowed transform types
- forbiddens (CMakeLists, public headers, API changes) unless allowed

---

## 7. Reproducibility & Audit Requirements

For every run_id:
- `manifest.json`: action, env, policy snapshot hash, case specs, tool versions
- `experiment.json`: iteration, decisions, evaluation plan
- `agent_trace.json`: agent inputs/outputs (sanitized), scores, reasons
- `build/` with provenance (if build happened)
- `repro.sh`
- logs: stdout/stderr/app log/time log
- metrics json
- verdict json

---

## 8. Default File Layout

- `prompts/system.md`
- `prompts/agents/{profiler,analyst,planner,optimizer,ranker,executor,verifier,triage,reporter}.md`
- `schemas/` Pydantic models for all IRs listed above
- `orchestrator/graph.py` deterministic state machine
- `orchestrator/agents/*.py` wrappers that call LLM or deterministic logic
- `skills/*.py`
- `configs/` policy.yaml, action_space.yaml, gates.yaml, planner.yaml, case_suites.yaml
- `artifacts/` generated outputs

---

## 9. Minimal LLM Usage Policy

- LLM-required agents: AnalystAgent (optional), OptimizerAgent (required), TriageAgent (optional), ReporterAgent (optional)
- LLM-prohibited for decision authority: Orchestrator, VerifierAgent, skills, gates
- All LLM outputs must be JSON that validates against IR schema; otherwise treat as FAIL and fallback or request more context.
