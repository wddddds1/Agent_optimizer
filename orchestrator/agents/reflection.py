from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from pydantic import ValidationError

from orchestrator.llm_client import LLMClient
from schemas.reflection_ir import ReflectionResult


class ReflectionAgent:
    """Lightweight single-LLM-call agent that reflects on batch results.

    Analyses which opportunities succeeded / failed in the last iteration,
    identifies category-level patterns, and reprioritises the remaining
    opportunity queue for the next iteration.
    """

    def __init__(self, llm_client: Optional[LLMClient]) -> None:
        self.llm_client = llm_client
        self.last_llm_trace: Optional[Dict[str, object]] = None

    def reflect(self, payload: Dict[str, object]) -> Optional[ReflectionResult]:
        """Run a single reflection call.  Returns *None* on any failure."""
        if not self.llm_client or not self.llm_client.config.enabled:
            return None
        prompt = _load_prompt("reflection")
        data = self.llm_client.request_json(prompt, payload)
        self.last_llm_trace = {"payload": payload, "response": data}
        if not data:
            return None
        try:
            result = ReflectionResult(**data)
        except ValidationError:
            return None
        if result.status != "OK":
            return None
        return result


def _load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "agents" / f"{name}.md"
    return path.read_text(encoding="utf-8")
