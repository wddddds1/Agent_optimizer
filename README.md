# HPC Agent Platform (MVP)

Local-only (macOS) multi-agent optimization control plane for HPC applications. The MVP focuses on LAMMPS but is
application-agnostic via an Action Grammar and IR schemas. All actions are auditable, reversible, and stored under
`artifacts/`.

## Prerequisites
- Python 3.11+
- LAMMPS binary on macOS (MPI optional)
- Git installed (required for patch/apply/revert workflow)

## Quick Start
1) Create a Git repo and baseline commit:
```
git init
git add .
git commit -m "baseline"
```

2) Install dependencies:
```
pip install -e .
```

3) Initialize LAMMPS submodule:
```
git submodule update --init --recursive
```

4) Configure paths in `configs/env.local.yaml` and add a case in `configs/lammps_cases.yaml`.
   - For direction-based selection, set `selection_mode: "direction"` and edit `configs/direction_space.yaml`.

5) Run:
```
python -m orchestrator.main --case <case_id>
```

Artifacts (logs, patches, JSON) are written to `artifacts/`.

## Adding a Case
Edit `configs/lammps_cases.yaml` and provide:
- `workdir` (directory containing input files)
- `input_script` (path to LAMMPS input)
- `run_args` (e.g. `["-in", "in.lj", "-log", "log.lammps"]`)
- optional `tags` for policy routing

## Inspecting Results
- `artifacts/runs/<run_id>/experiment.json` contains the full ExperimentIR.
- `artifacts/runs/<run_id>/manifest.json` records build provenance, binary hashes, and verification summary.
- `artifacts/runs/<run_id>/agent_trace.json` contains per-run agent events.
- `artifacts/report.json` and `artifacts/report.md` summarize the run.
- `artifacts/report_zh.md` contains a Chinese summary of the full optimization.
- `artifacts/runs/<run_id>/patch.diff` (if input was modified) or `run_config.diff.json` (if env/args changed).
- `artifacts/ledger/run_index.jsonl` provides a one-line index of all runs.
- `artifacts/ledger/iteration_###_summary_zh.md` provides a Chinese summary per iteration.

## Tests
```
pytest
```

## LAMMPS Build (macOS)
Out-of-source build layout:
- `third_party/lammps/` (submodule)
- `build/lammps-macos/` (build output)

Example build:
```
cmake -S third_party/lammps/cmake -B build/lammps-macos -D BUILD_MPI=off -D BUILD_OMP=on -D PKG_OMP=on -D CMAKE_BUILD_TYPE=Release
cmake --build build/lammps-macos -j 4
```

## Notes
- The system does not require an LLM to run. Optional LLM hooks are stubbed for future use.
- For run-config-only actions, correctness checks may be skipped by policy and explicitly recorded.
- Build-config/source-patch actions trigger isolated builds under `artifacts/runs/<run_id>/build/` with
  `build.log`, `CMakeCache.txt`, and `compile_commands.json` recorded in the manifest.

## Action Grammar
Actions are defined in `configs/action_space.yaml` using a small grammar:
- `preconditions` and `constraints` are machine-checkable rules (e.g. `input_contains`, `args_not_contains`)
- `parameters` describe deterministic edits (env, run_args flags, or input_script replacements)

Direction-based selection (recommended for parameter tuning) uses `configs/direction_space.yaml` to define
optimization directions with multi-parameter presets (e.g., threads + suffix + affinity in one action). Some
directions (comm/io/accuracy/runtime) are gated by case tags such as `allow_comm`, `allow_io`, `allow_accuracy`,
and `allow_runtime_lib`.

## Optional LLM Ranking (DeepSeek)
To enable LLM-based action ranking:
1) Set `llm.enabled: true` in `configs/env.local.yaml`.
2) Export `DEEPSEEK_API_KEY` (or change `api_key_env`).

The LLM is only used to rank actions; all executions remain deterministic via skills.
