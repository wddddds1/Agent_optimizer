"""Tool implementations for the parameter exploration agent (Phase 1).

Provides a lightweight tool set focused on platform inspection, profile data,
action-space querying, input-script reading, and experience search.  Source-code
modification tools are intentionally excluded — this agent only *proposes*
parameter candidates.
"""
from __future__ import annotations

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestrator.agent_llm import ToolDefinition
from skills.shell_tool import ShellTool
from skills.system_probe import probe_system_info


class ParameterExplorerTools:
    """Tools available to the ParameterExplorerAgent."""

    def __init__(
        self,
        repo_root: Path,
        input_script_path: Optional[Path] = None,
        experience_db: Optional[Any] = None,
    ) -> None:
        self.repo_root = repo_root
        self.input_script_path = input_script_path
        self.experience_db = experience_db
        self._shell_tool = ShellTool(cwd=repo_root)
        self._profile_data: Optional[Dict[str, Any]] = None
        self._action_space: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Setters (called by the orchestrator before the agent loop)
    # ------------------------------------------------------------------

    def set_profile_data(self, profile: Dict[str, Any]) -> None:
        self._profile_data = profile

    def set_action_space(self, action_space: Dict[str, Any]) -> None:
        self._action_space = action_space

    # ------------------------------------------------------------------
    # platform probe (direct, deterministic)
    # ------------------------------------------------------------------

    def probe_platform(self) -> Dict[str, str]:
        """Collect a small platform snapshot for the agent prompt."""
        return probe_system_info()

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def get_all_tools(self) -> List[ToolDefinition]:
        return [
            # Platform / hardware inspection
            self._shell_tool.get_tool_definition(),

            # File reading (read-only, repo-scoped)
            self._tool_read_file(),

            # Performance profile
            self._tool_get_profile(),

            # Action space
            self._tool_get_action_space(),

            # Input script
            self._tool_read_input_script(),

            # Experience
            self._tool_search_experience(),
        ]

    # ------------------------------------------------------------------
    # read_file
    # ------------------------------------------------------------------

    def _tool_read_file(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_file",
            description=(
                "Read a file from the repository (source code, configs, etc.). "
                "Path is relative to the repo root."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to repository root",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Start line (1-indexed, optional)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "End line (1-indexed, optional)",
                    },
                },
                "required": ["path"],
            },
            handler=self._handle_read_file,
        )

    def _handle_read_file(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        file_path = self.repo_root / path
        if not file_path.exists():
            return f"Error: File not found: {path}"
        if not file_path.is_file():
            return f"Error: Not a file: {path}"
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            if start_line is not None or end_line is not None:
                start = (start_line or 1) - 1
                end = end_line or len(lines)
                lines = lines[start:end]
                numbered = [
                    f"{i + start + 1:4d} | {line}"
                    for i, line in enumerate(lines)
                ]
                return "\n".join(numbered)
            if len(lines) > 500:
                numbered = [f"{i+1:4d} | {l}" for i, l in enumerate(lines[:500])]
                return "\n".join(numbered) + f"\n... ({len(lines)} total lines, truncated)"
            numbered = [f"{i+1:4d} | {l}" for i, l in enumerate(lines)]
            return "\n".join(numbered)
        except Exception as exc:
            return f"Error: {exc}"

    # ------------------------------------------------------------------
    # get_profile
    # ------------------------------------------------------------------

    def _tool_get_profile(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_profile",
            description=(
                "Get the baseline performance profile including timing breakdown "
                "(pair, neigh, comm, output, modify, other), system metrics "
                "(CPU utilisation, RSS), and any notes."
            ),
            parameters={"type": "object", "properties": {}},
            handler=self._handle_get_profile,
        )

    def _handle_get_profile(self) -> str:
        if not self._profile_data:
            return "No profile data available."
        return json.dumps(self._profile_data, indent=2)

    # ------------------------------------------------------------------
    # get_action_space
    # ------------------------------------------------------------------

    def _tool_get_action_space(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_action_space",
            description=(
                "Get the available parameter action families and their templates. "
                "Returns families (with descriptions and expected effects) and "
                "concrete actions you can propose as candidates. "
                "Only parameter families are shown (no source_patch)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "family_filter": {
                        "type": "string",
                        "description": (
                            "Optional: filter to a specific family id "
                            "(e.g. 'parallel_omp', 'neighbor_tune'). "
                            "Leave empty for all families."
                        ),
                    },
                },
            },
            handler=self._handle_get_action_space,
        )

    def _handle_get_action_space(self, family_filter: str = "") -> str:
        if not self._action_space:
            return "No action space loaded."

        families = self._action_space.get("families", [])
        actions = self._action_space.get("actions", [])

        # Exclude source_patch — that's for Phase 2
        param_families = [
            f for f in families if f.get("id") != "source_patch"
        ]
        param_actions = [
            a for a in actions
            if a.get("family") != "source_patch"
        ]

        if family_filter:
            param_families = [
                f for f in param_families if f.get("id") == family_filter
            ]
            param_actions = [
                a for a in param_actions if a.get("family") == family_filter
            ]

        result = {
            "families": param_families,
            "actions_count": len(param_actions),
            "actions": param_actions,
        }
        text = json.dumps(result, indent=2)
        if len(text) > 30000:
            # Truncate actions list if too large
            result["actions"] = param_actions[:30]
            result["note"] = (
                f"Showing first 30 of {len(param_actions)} actions. "
                "Use family_filter to narrow."
            )
            text = json.dumps(result, indent=2)
        return text

    # ------------------------------------------------------------------
    # read_input_script
    # ------------------------------------------------------------------

    def _tool_read_input_script(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_input_script",
            description=(
                "Read the LAMMPS input script for the current benchmark case. "
                "This shows the simulation setup: atom style, pair style, "
                "neighbor settings, thermo output, run length, etc."
            ),
            parameters={"type": "object", "properties": {}},
            handler=self._handle_read_input_script,
        )

    def _handle_read_input_script(self) -> str:
        if not self.input_script_path:
            return "No input script path configured."
        path = Path(self.input_script_path)
        if not path.exists():
            return f"Error: Input script not found: {path}"
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            numbered = [f"{i+1:3d} | {l}" for i, l in enumerate(lines)]
            return "\n".join(numbered)
        except Exception as exc:
            return f"Error: {exc}"

    # ------------------------------------------------------------------
    # search_experience
    # ------------------------------------------------------------------

    def _tool_search_experience(self) -> ToolDefinition:
        return ToolDefinition(
            name="search_experience",
            description=(
                "Search the experience database for past optimization results. "
                "Returns records of previously tried actions with their "
                "improvement percentages and verdicts. Use this to avoid "
                "repeating unsuccessful configurations and to build on "
                "successful ones."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "family": {
                        "type": "string",
                        "description": (
                            "Filter by action family "
                            "(e.g. 'parallel_omp', 'neighbor_tune'). "
                            "Leave empty for all."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max records to return (default 20)",
                    },
                },
            },
            handler=self._handle_search_experience,
        )

    def _handle_search_experience(
        self, family: str = "", limit: int = 20
    ) -> str:
        if not self.experience_db:
            return "No experience database available."

        records = self.experience_db.records
        if not records:
            return "Experience database is empty (no past optimization records)."

        if family:
            records = [r for r in records if r.family == family]

        # Sort by improvement (best first)
        records = sorted(
            records,
            key=lambda r: r.improvement_pct if r.improvement_pct is not None else -999,
            reverse=True,
        )
        records = records[:limit]

        if not records:
            return f"No experience records found" + (f" for family '{family}'" if family else "") + "."

        results = []
        for r in records:
            entry = {
                "action_id": r.action_id,
                "family": r.family,
                "improvement_pct": round(r.improvement_pct, 2) if r.improvement_pct is not None else None,
                "speedup": round(r.speedup, 3) if r.speedup is not None else None,
                "verdict": r.verdict,
                "strength": r.strength,
                "app": r.app,
                "case_id": r.case_id,
            }
            if r.patch_family:
                entry["patch_family"] = r.patch_family
            results.append(entry)

        return json.dumps(results, indent=2)
