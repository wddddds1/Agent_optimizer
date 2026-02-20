"""Bridge adapter to use the new agentic code optimizer with the existing orchestration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from orchestrator.agent_llm import AgentConfig, MAX_TURNS_SENTINEL
from orchestrator.agents.code_optimizer_agent import CodeOptimizerAgent
from schemas.action_ir import ActionIR
from schemas.patch_proposal_ir import PatchProposal
from schemas.profile_report import ProfileReport


class AgenticCodePatchAgent:
    """Drop-in replacement for CodePatchAgent using the new agentic approach.

    This adapter maintains the same interface as CodePatchAgent but uses
    the new multi-turn, tool-using agent internally.
    """

    def __init__(
        self,
        repo_root: Path,
        build_dir: Optional[Path] = None,
        api_key_env: str = "DEEPSEEK_API_KEY",
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        enabled: bool = True,
        max_turns: int = 25,
        max_tool_calls_per_turn: int = 5,
        max_invalid_tool_calls_total: int = 5,
        max_invalid_tool_calls_per_tool: int = 2,
        request_timeout_sec: float = 120.0,
        api_timeout_retries: int = 2,
    ) -> None:
        self.repo_root = repo_root
        self.enabled = enabled
        self.last_trace: Optional[Dict[str, Any]] = None

        if not enabled:
            self.agent = None
            return

        config = AgentConfig(
            enabled=True,
            api_key_env=api_key_env,
            base_url=base_url,
            model=model,
            temperature=0.2,
            max_tokens=4096,
            max_turns=max_turns,
            max_tool_calls_per_turn=max_tool_calls_per_turn,
            max_invalid_tool_calls_total=max_invalid_tool_calls_total,
            max_invalid_tool_calls_per_tool=max_invalid_tool_calls_per_tool,
            request_timeout_sec=request_timeout_sec,
            api_timeout_retries=api_timeout_retries,
        )

        self.agent = CodeOptimizerAgent(config, repo_root, build_dir)

    def _normalize_target_file(self, target_file: str, orchestration_repo_root: Path) -> str:
        """Normalize target path to this agent's repo_root when possible."""
        if not target_file:
            return target_file
        raw = Path(target_file)
        try:
            if raw.is_absolute():
                resolved = raw.resolve()
            else:
                local_candidate = (self.repo_root / raw).resolve()
                if local_candidate.exists():
                    return raw.as_posix()
                global_candidate = (orchestration_repo_root / raw).resolve()
                if not global_candidate.exists():
                    return raw.as_posix()
                resolved = global_candidate
            return resolved.relative_to(self.repo_root.resolve()).as_posix()
        except Exception:
            return raw.as_posix()

    def _normalize_allowed_files(
        self,
        allowed_files: List[str],
        orchestration_repo_root: Path,
    ) -> List[str]:
        normalized: List[str] = []
        for item in allowed_files:
            if not isinstance(item, str) or not item:
                continue
            value = self._normalize_target_file(item, orchestration_repo_root)
            if value and value not in normalized:
                normalized.append(value)
        return normalized

    def _normalize_patch_rules(
        self,
        patch_rules: Dict[str, object],
        orchestration_repo_root: Path,
    ) -> Dict[str, object]:
        if not isinstance(patch_rules, dict):
            return {}
        normalized_rules = dict(patch_rules)
        # Keep patch rules as guidance only; do not pass hard file-scope limits.
        normalized_rules.pop("allowed_globs", None)
        normalized_rules.pop("patch_root", None)
        return normalized_rules

    def propose(
        self,
        action: ActionIR,
        profile: ProfileReport,
        patch_rules: Dict[str, object],
        allowed_files: List[str],
        code_snippets: List[Dict[str, object]],
        repo_root: Path,
        feedback: Optional[str] = None,
        backend_variant: Optional[str] = None,
        reference_template: Optional[Dict[str, str]] = None,
        navigation_hints: Optional[List[Dict[str, object]]] = None,
    ) -> Optional[PatchProposal]:
        """Generate a patch proposal using the agentic approach.

        When ``navigation_hints`` are provided the agent receives lightweight
        file pointers (path, hotspot line, function signature) instead of
        pre-extracted code snippets.  The agent uses its own ``read_file``
        and ``grep`` tools to explore code autonomously.
        """
        if not self.agent or not self.enabled:
            return None

        normalized_allowed_files = self._normalize_allowed_files(allowed_files, repo_root)

        # Extract target file from action parameters or navigation hints
        params = action.parameters or {}
        target_file = params.get("target_file", "")

        if not target_file:
            # Derive from navigation hints first, then allowed_files
            if navigation_hints:
                target_file = str(navigation_hints[0].get("path", ""))
            elif normalized_allowed_files:
                patch_files = params.get("patch_files", [])
                if patch_files:
                    target_file = patch_files[0]
                else:
                    target_file = normalized_allowed_files[0]
        if isinstance(target_file, str) and target_file:
            target_file = self._normalize_target_file(target_file, repo_root)

        # Build additional context
        additional_context: Dict[str, Any] = {
            "backend": backend_variant or "unknown",
            "allowed_files": normalized_allowed_files,
            "patch_rules": self._normalize_patch_rules(patch_rules, repo_root),
        }

        if navigation_hints:
            normalized_hints: List[Dict[str, object]] = []
            for hint in navigation_hints:
                if not isinstance(hint, dict):
                    continue
                normalized = dict(hint)
                path = normalized.get("path")
                if isinstance(path, str) and path:
                    normalized["path"] = self._normalize_target_file(path, repo_root)
                normalized_hints.append(normalized)
            if normalized_hints:
                additional_context["navigation_hints"] = normalized_hints

        if feedback:
            additional_context["previous_feedback"] = feedback

        if reference_template:
            additional_context["reference_template"] = reference_template

        # Run the agentic optimization — no pre-extracted hotspot code;
        # the agent reads files itself using its read_file tool.
        result = self.agent.optimize(
            action=action,
            profile=profile,
            target_file=target_file,
            hotspot_code=None,
            additional_context=additional_context,
        )

        # Store trace for debugging
        self.last_trace = {
            "total_turns": result.total_turns,
            "total_tokens": result.total_tokens,
            "diagnosis": result.diagnosis,
            "conversation_log": result.conversation_log,
        }

        # Persist conversation log to disk for post-hoc analysis
        self._save_conversation_log(action.action_id, result)

        # Convert to PatchProposal format
        # Guard: reject results where the agent ran out of turns without a patch
        if MAX_TURNS_SENTINEL in (result.rationale or "") and not result.patch_diff:
            return PatchProposal(
                status="NEED_MORE_CONTEXT",
                patch_diff="",
                touched_files=[],
                rationale=result.rationale,
                assumptions=[],
                confidence=0.0,
                missing_fields=["Agent exhausted max turns without producing a patch"],
            )

        if result.status == "ERROR":
            return PatchProposal(
                status="NEED_MORE_CONTEXT",
                patch_diff="",
                touched_files=[],
                rationale=result.rationale,
                assumptions=[],
                confidence=0.0,
                missing_fields=[f"Agent error: {result.rationale}"],
            )

        if result.status == "NO_OPTIMIZATION_POSSIBLE":
            return PatchProposal(
                status="NEED_MORE_CONTEXT",
                patch_diff="",
                touched_files=[],
                rationale=result.rationale,
                assumptions=[],
                confidence=result.confidence,
                missing_fields=["No optimization possible for this code"],
            )

        if not result.patch_diff:
            return PatchProposal(
                status="NEED_MORE_CONTEXT",
                patch_diff="",
                touched_files=[],
                rationale=result.rationale,
                assumptions=[],
                confidence=result.confidence,
                missing_fields=["Agent did not produce a patch"],
            )

        # Extract touched files from patch diff
        touched_files = self._extract_touched_files(result.patch_diff)

        # AST post-validation: check structural correctness before returning
        ast_warnings: List[str] = []
        try:
            from skills.code_structure import is_available as _ts_avail, validate_patch_structure
            if _ts_avail() and touched_files:
                for tf in touched_files:
                    full_path = self.repo_root / tf
                    if not full_path.is_file():
                        continue
                    # Apply patch to get patched content for validation
                    patched_content = self._simulate_patch(full_path, result.patch_diff)
                    if patched_content is not None:
                        validation = validate_patch_structure(str(full_path), patched_content)
                        if not validation.valid:
                            ast_warnings.extend(
                                [f"[AST] {tf}: {e}" for e in validation.errors]
                            )
                        ast_warnings.extend(
                            [f"[AST warning] {tf}: {w}" for w in validation.warnings]
                        )
        except Exception:
            pass  # AST validation is best-effort

        rationale = result.rationale or ""
        if ast_warnings:
            rationale += "\n\nAST validation notes:\n" + "\n".join(ast_warnings[:5])

        return PatchProposal(
            status="OK",
            patch_diff=result.patch_diff,
            touched_files=touched_files,
            rationale=rationale,
            assumptions=[],
            confidence=result.confidence,
            missing_fields=[],
        )

    def _save_conversation_log(self, action_id: str, result: Any) -> None:
        """Save the full conversation log to a JSON file next to the repo root."""
        try:
            log_dir = self.repo_root / "artifacts" / "agentic_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            import time as _time
            ts = _time.strftime("%Y%m%d-%H%M%S")
            safe_id = action_id.replace("/", "_").replace(" ", "_")
            log_path = log_dir / f"{ts}_{safe_id}.json"
            log_data = {
                "action_id": action_id,
                "total_turns": result.total_turns,
                "total_tokens": result.total_tokens,
                "status": result.status,
                "diagnosis": result.diagnosis,
                "rationale": result.rationale,
                "confidence": result.confidence,
                "conversation_log": result.conversation_log,
            }
            log_path.write_text(json.dumps(log_data, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass  # Best-effort — don't fail the pipeline for logging

    def _simulate_patch(self, file_path: Path, patch_diff: str) -> Optional[str]:
        """Apply a unified diff to file content in memory for AST validation.

        Returns the patched file content, or None if simulation fails.
        """
        try:
            import subprocess
            original = file_path.read_text(encoding="utf-8", errors="replace")
            # Use subprocess with `patch --dry-run` to simulate
            proc = subprocess.run(
                ["patch", "--dry-run", "-p1", "-o", "-"],
                input=patch_diff,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(self.repo_root),
            )
            if proc.returncode == 0 and proc.stdout:
                return proc.stdout
        except Exception:
            pass
        return None

    def _extract_touched_files(self, patch_diff: str) -> List[str]:
        """Extract file paths from a unified diff."""
        import re
        files = []
        for line in patch_diff.split("\n"):
            if line.startswith("+++ b/") or line.startswith("--- a/"):
                path = line[6:]  # Remove "+++ b/" or "--- a/"
                if path and path not in files:
                    files.append(path)
        return files


def create_agentic_code_patch_agent(
    repo_root: Path,
    build_dir: Optional[Path] = None,
    llm_config: Optional[Dict[str, Any]] = None,
) -> AgenticCodePatchAgent:
    """Factory function to create an AgenticCodePatchAgent.

    Args:
        repo_root: Path to the repository root
        build_dir: Path to the build directory (optional)
        llm_config: LLM configuration dict with keys:
            - api_key_env: Environment variable name for API key
            - base_url: API base URL
            - model: Model name
            - enabled: Whether the agent is enabled
            - max_turns: Maximum conversation turns (default 25)
            - max_tool_calls_per_turn: Maximum tool calls per turn (default 5)
            - max_invalid_tool_calls_total: Max invalid tool calls before hard fail
            - max_invalid_tool_calls_per_tool: Max consecutive invalid calls per tool
            - request_timeout_sec: Per-request timeout in seconds (default 120)
            - api_timeout_retries: Extra retries for retryable timeout/network errors

    Returns:
        Configured AgenticCodePatchAgent
    """
    config = llm_config or {}

    return AgenticCodePatchAgent(
        repo_root=repo_root,
        build_dir=build_dir,
        api_key_env=config.get("api_key_env", "DEEPSEEK_API_KEY"),
        base_url=config.get("base_url", "https://api.deepseek.com"),
        model=config.get("model", "deepseek-chat"),
        enabled=config.get("enabled", True),
        max_turns=int(config.get("max_turns", 25)),
        max_tool_calls_per_turn=int(config.get("max_tool_calls_per_turn", 5)),
        max_invalid_tool_calls_total=int(config.get("max_invalid_tool_calls_total", 5)),
        max_invalid_tool_calls_per_tool=int(config.get("max_invalid_tool_calls_per_tool", 2)),
        request_timeout_sec=float(config.get("request_timeout_sec", 120.0)),
        api_timeout_retries=int(config.get("api_timeout_retries", 2)),
    )
