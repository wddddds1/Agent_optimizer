from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from pydantic import ValidationError

from orchestrator.llm_client import LLMClient
from schemas.review_ir import ReviewDecision


class ReviewerAgent:
    # After this many consecutive fallback responses, force a stop to avoid
    # infinite loops when the LLM is consistently failing.
    MAX_CONSECUTIVE_FALLBACKS = 3

    def __init__(self, llm_client: Optional[LLMClient]) -> None:
        self.llm_client = llm_client
        self.last_llm_trace: Optional[Dict[str, object]] = None
        self._consecutive_fallbacks = 0

    def review(self, payload: Dict[str, object]) -> ReviewDecision:
        if not self.llm_client or not self.llm_client.config.enabled:
            return self._fallback("LLM disabled; continue by default.")
        prompt = _load_prompt("reviewer")
        data = self.llm_client.request_json(prompt, payload)
        self.last_llm_trace = {"payload": payload, "response": data}
        if not data:
            return self._fallback("LLM returned empty response; continue by default.")
        try:
            decision = ReviewDecision(**data)
        except ValidationError:
            return self._fallback("LLM response invalid; continue by default.")
        if decision.status != "OK":
            return self._fallback(
                "LLM returned non-OK status; continue by default.",
                extra_evidence={"status": decision.status},
            )
        # Valid response — reset fallback counter
        self._consecutive_fallbacks = 0
        return decision

    def _fallback(
        self,
        reason: str,
        extra_evidence: Optional[Dict[str, object]] = None,
    ) -> ReviewDecision:
        """Return a fallback decision, stopping if too many consecutive fallbacks."""
        self._consecutive_fallbacks += 1
        evidence: Dict[str, object] = {"fallback": True, "consecutive_fallbacks": self._consecutive_fallbacks}
        if extra_evidence:
            evidence.update(extra_evidence)
        if self._consecutive_fallbacks >= self.MAX_CONSECUTIVE_FALLBACKS:
            return ReviewDecision(
                should_stop=True,
                confidence=0.0,
                reason=f"{reason} ({self._consecutive_fallbacks} consecutive fallbacks — forcing stop)",
                evidence=evidence,
                suggested_next_step="stop",
            )
        return ReviewDecision(
            should_stop=False,
            confidence=0.0,
            reason=reason,
            evidence=evidence,
            suggested_next_step="continue",
        )


def _load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "agents" / f"{name}.md"
    return path.read_text(encoding="utf-8")
