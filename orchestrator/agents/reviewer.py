from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from pydantic import ValidationError

from orchestrator.llm_client import LLMClient
from schemas.review_ir import ReviewDecision


class ReviewerAgent:
    def __init__(self, llm_client: Optional[LLMClient]) -> None:
        self.llm_client = llm_client
        self.last_llm_trace: Optional[Dict[str, object]] = None

    def review(self, payload: Dict[str, object]) -> ReviewDecision:
        if not self.llm_client or not self.llm_client.config.enabled:
            return ReviewDecision(
                should_stop=False,
                confidence=0.0,
                reason="LLM disabled; continue by default.",
                evidence={"fallback": True},
                suggested_next_step="continue",
            )
        prompt = _load_prompt("reviewer")
        data = self.llm_client.request_json(prompt, payload)
        self.last_llm_trace = {"payload": payload, "response": data}
        if not data:
            return ReviewDecision(
                should_stop=False,
                confidence=0.0,
                reason="LLM returned empty response; continue by default.",
                evidence={"fallback": True},
                suggested_next_step="continue",
            )
        try:
            decision = ReviewDecision(**data)
        except ValidationError:
            return ReviewDecision(
                should_stop=False,
                confidence=0.0,
                reason="LLM response invalid; continue by default.",
                evidence={"fallback": True},
                suggested_next_step="continue",
            )
        if decision.status != "OK":
            return ReviewDecision(
                should_stop=False,
                confidence=0.0,
                reason="LLM returned non-OK status; continue by default.",
                evidence={"fallback": True, "status": decision.status},
                suggested_next_step="continue",
            )
        return decision


def _load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "agents" / f"{name}.md"
    return path.read_text(encoding="utf-8")
