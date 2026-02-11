from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from pydantic import ValidationError

from orchestrator.llm_client import LLMClient
from schemas.patch_review_ir import PatchReview


class PatchReviewAgent:
    def __init__(self, llm_client: Optional[LLMClient]) -> None:
        self.llm_client = llm_client
        self.last_trace: Optional[Dict[str, object]] = None

    def review(
        self,
        patch_diff: str,
        patch_rules: Dict[str, object],
        context: Dict[str, object],
    ) -> Optional[PatchReview]:
        if not self.llm_client or not self.llm_client.config.enabled:
            return None
        prompt = _load_prompt("patch_review")
        payload = {
            "patch_diff": patch_diff,
            "patch_rules": patch_rules,
            "context": context,
        }
        data = self.llm_client.request_json(prompt, payload)
        self.last_trace = {"payload": payload, "response": data}
        if not data:
            return None
        try:
            review = PatchReview(**data)
        except ValidationError:
            return None
        return review


def _load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "agents" / f"{name}.md"
    return path.read_text(encoding="utf-8")
