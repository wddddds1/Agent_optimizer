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
        )

        self.agent = CodeOptimizerAgent(config, repo_root, build_dir)

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

        # Extract target file from action parameters or navigation hints
        params = action.parameters or {}
        target_file = params.get("target_file", "")

        if not target_file:
            # Derive from navigation hints first, then allowed_files
            if navigation_hints:
                target_file = str(navigation_hints[0].get("path", ""))
            elif allowed_files:
                patch_files = params.get("patch_files", [])
                if patch_files:
                    target_file = patch_files[0]
                else:
                    target_file = allowed_files[0]

        # Build additional context
        additional_context: Dict[str, Any] = {
            "backend": backend_variant or "unknown",
            "allowed_files": allowed_files,
            "patch_rules": patch_rules,
        }

        if navigation_hints:
            additional_context["navigation_hints"] = navigation_hints

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

        return PatchProposal(
            status="OK",
            patch_diff=result.patch_diff,
            touched_files=touched_files,
            rationale=result.rationale,
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
    )
